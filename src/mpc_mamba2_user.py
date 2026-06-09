import torch
import crypten
import time
import json
from typing import TypeAlias
from mpc_mamba2 import Mamba2LMHeadModel,Mamba2Config
from transformers import AutoTokenizer

Device: TypeAlias = str | torch.device | None

generation_config = dict(
    max_new_length=200,
    temperature=1.0,
    top_k=30,
    top_p=1.0,
)

# Hugging Faceから事前学習済みのMamba2の内部パラメータをおろす
def from_pretrained(huggingface_model_id: str, device: Device = None):
    from transformers.utils import CONFIG_NAME, WEIGHTS_NAME
    from transformers.utils.hub import cached_file

    config_path = cached_file(huggingface_model_id, CONFIG_NAME)
    assert config_path, "Failed to get huggingface config file"
    state_dict_path = cached_file(huggingface_model_id, WEIGHTS_NAME)
    assert state_dict_path, "Failed to get huggingface state dict file"
    config = json.load(open(config_path))

    # モデル構造に関わるパラメータをおろす
    args = Mamba2Config(
        d_model=config["d_model"],
        n_layer=config["n_layer"],
        vocab_size=config["vocab_size"],
        pad_vocab_size_multiple=config["pad_vocab_size_multiple"],
    )
    map_location = "cpu" if device is None else device
    state_dict = torch.load(
        state_dict_path, weights_only=True, map_location=map_location, mmap=True # Hugging Faceのpath、重みのみダウンロード、CPU or GPU、メモリマッピングON
    )
    # モデル定義
    model = Mamba2LMHeadModel(args, device=device)
    # ダウンロードした重みをロード
    model.load_state_dict(state_dict)

    return model

def client_load_and_encrypt_model(model_id,device: Device = None):
    """
    ユーザーの手元（安全な環境）で平文モデルを読み込み、
    それを暗号化テンソル（シェア）に変換したもので新たなモデルを作成する関数
    """

    plain_model = from_pretrained(model_id,device) 

    encrypted_weights = {}
    for name, param in plain_model.state_dict().items():
        encrypted_weights[name] = crypten.cryptensor(param)
    
    mpc_model = Mamba2LMHeadModel(plain_model.args) 
    # 先ほど暗号化した重みを流し込む
    mpc_model.load_encrypted_state_dict(encrypted_weights)

    mpc_model.eval()
        
    return mpc_model

def user_generate(
    mpc_model: Mamba2LMHeadModel, 
    prompt: str, 
    tokenizer, 
    generation_config: dict=generation_config, 
    show_perf=True
):
    """
    ユーザーの手元（クライアントサイド）で実行するメイン生成ループ関数。
    """
    
    # プロンプトを平文のトークンIDに変換
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)[0]
    vocab_size = mpc_model.args.vocab_size
    
    print(prompt, end="")

    start = time.process_time()
    n_generated = 0
    h = None

    # 設定の読み込み
    max_new_length = generation_config.get("max_new_length", 20)
    temperature = generation_config.get("temperature", 1.0)
    top_k = generation_config.get("top_k", 50)
    top_p = generation_config.get("top_p", 1.0)
    eos_token_id = generation_config.get("eos_token_id", 0)

    # 最初のトークンIDをセット（プロンプトの最後のトークン）
    prefix = input_ids[:-1]
    
    h = None
    if prefix.shape[0] > 0:
        # 残してある通常の forward 関数を呼び出し、プロンプト文脈を反映した初期キャッシュ h を作成する
        # forward 内部で入力を受け取った直後に自動で crypten.cryptensor() に変換されます
        with torch.no_grad():
            _, h = mpc_model(prefix.unsqueeze(0), None)
    
    current_token_id = input_ids[-1].item()

    # 生成ループの開始
    for i in range(max_new_length):
        
        # 現在のトークンIDを平文のOne-hot表現にし、暗号化（シェア化）してサーバーに送る形式にする
        onehot = torch.zeros(1, 1, vocab_size, device=device)
        onehot[0, 0, current_token_id] = 1.0
        onehot_mpc = crypten.cryptensor(onehot)  # 暗号化
        
        # サーバーの秘密計算関数を呼び出し、次のロジット（シェア）とキャッシュを受け取る
        with torch.no_grad():
            logits_mpc, h = mpc_model.predict_next_logit_mpc(onehot_mpc, h)
            
        # サーバーから返ってきた「ロジットのシェア」をユーザーの手元だけで復号する
        # これにより logits_plain は (vocab_size,) の普通のPyTorchテンソル（平文）になる
        logits_plain = logits_mpc.get_plain_text().squeeze(0).squeeze(0)

        # 「平文」の処理
        if temperature != 1.0:
            logits_plain = logits_plain / temperature
            
        # Top-K フィルター
        if top_k > 0:
            indices_to_remove = logits_plain < torch.topk(logits_plain, k=top_k)[0][-1]
            logits_plain[indices_to_remove] = -float('inf')
            
        # Top-P フィルター
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits_plain, descending=True)
            cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cum_probs > top_p
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = False
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits_plain[indices_to_remove] = -float('inf')
            
        # 次のトークンを確率分布からサンプリング
        probs = torch.softmax(logits_plain, dim=-1)
        next_token_tensor = torch.multinomial(probs, num_samples=1)
        current_token_id = next_token_tensor.item()

        # 終了トークンならループを抜ける
        if current_token_id == eos_token_id:
            break

        # デコードして画面に出力
        token_str = tokenizer.decode([current_token_id])
        print(token_str, end="", flush=True)

        # 性能測定用のカウンター
        if i == 0:
            now = time.process_time()
            prompt_eval_elapsed, start = now - start, now
        else:
            n_generated += 1

    # 性能の表示
    if show_perf and n_generated > 0:
        elapsed = time.process_time() - start
        print('\n\n---')
        print(f'Prompt eval | tokens: {input_ids.shape[0]} | elapsed: {prompt_eval_elapsed:.2f}s | tok/s: {input_ids.shape[0] / prompt_eval_elapsed:.2f}')
        print(f'Generation | tokens: {n_generated} | elapsed: {elapsed:.2f}s | tok/s: {n_generated / elapsed:.2f}')

if __name__ == "__main__":
    crypten.init()


    device=None
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    
    model = client_load_and_encrypt_model("state-spaces/mamba2-1.3b", device=device)
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    tokenizer.pad_token_id = tokenizer.eos_token_id

    user_generate(model,"Mamba-2 with MPC is",tokenizer)
