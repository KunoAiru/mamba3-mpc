"""
mamba2-minimal
==============

A minimal, single-file implementation of the Mamba-2 model in PyTorch.

> **Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality**
> Authors: Tri Dao, Albert Gu
> Paper: https://arxiv.org/abs/2405.21060
"""

import json
from dataclasses import dataclass
from typing import Iterable, NamedTuple, TypeAlias, cast

import torch
from einops import rearrange, repeat
import crypten
from torch import LongTensor, Tensor, nn


Device: TypeAlias = str | torch.device | None


@dataclass
class Mamba2Config:
    d_model: int  # model dimension (D)
    n_layer: int = 24  # number of Mamba-2 layers in the language model
    d_state: int = 128  # state dimension (N)
    d_conv: int = 4  # convolution kernel size 直近でみるトークン数
    expand: int = 2  # expansion factor (E)
    headdim: int = 64  # head dimension (P)
    chunk_size: int = 64  # matrix partition size (Q)
    vocab_size: int = 50277
    pad_vocab_size_multiple: int = 16

    # コンストラクタ後の初期化処理
    def __post_init__(self):
        self.d_inner = self.expand * self.d_model
        assert self.d_inner % self.headdim == 0
        self.nheads = self.d_inner // self.headdim

        #GPUの並列計算用に、単語総数をself.pad_vocab_size_multiple(=16)の倍数になるように調整
        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            self.vocab_size += (
                self.pad_vocab_size_multiple
                - self.vocab_size % self.pad_vocab_size_multiple
            )

#1トークンずつ推論する際のキャッシュ
class InferenceCache(NamedTuple):
    conv_state: Tensor  # (batch, d_inner + 2 * d_state, d_conv)
    ssm_state: Tensor  # (batch, nheads, headdim, d_state)

    @staticmethod
    def alloc(batch_size: int, args: Mamba2Config, device: Device = None):
        return InferenceCache(
            crypten.cryptensor(torch.zeros(
                batch_size, args.d_inner + 2 * args.d_state, args.d_conv, device=device
            )),
            crypten.cryptensor(torch.zeros(
                batch_size, args.nheads, args.headdim, args.d_state, device=device
            )),
        )


class Mamba2LMHeadModel(nn.Module):
    def __init__(self, args: Mamba2Config, device: Device = None):
        # nn.Moduleクラスの基底コンストラクタ呼び出し
        super().__init__()
        # Mamba2Configをarg、CPU or GPUをdeviceで定義する
        self.args = args
        self.device = device

        # backbone：Mamba2の主要なコンポーネントを辞書型でまとめる
        self.backbone = nn.ModuleDict(
            dict(
                # 単語埋め込み層 vocab_size(トークンのIDの次元・語彙集合の数) → d_model(単語埋め込みベクトルの次元)
                embedding=nn.Embedding(args.vocab_size, args.d_model, device=device),

                # Mamba2のレイヤーをn_layer個まで積み重ねる
                layers=nn.ModuleList(
                    [
                        nn.ModuleDict(
                            dict(
                                # 状態空間モデル(SSM)
                                mixer=Mamba2(args, device=device),
                                # RMSNormによる層正規化
                                norm=RMSNorm(args.d_model, device=device),
                            )
                        )
                        for _ in range(args.n_layer)
                    ]
                ),

                # 最後に最終的な層正規化を行う
                norm_f=RMSNorm(args.d_model, device=device),
            )
        )

        # d_model(単語埋め込みベクトルの次元) → vocab_size(トークンIDのスコアの次元)へ変換する線形層
        self.lm_head = nn.Linear(
            args.d_model, args.vocab_size, bias=False, device=device
        )

        # 入口のembeddingの重みと、出口のlm_headで最終出力された重みを共有する
        self.lm_head.weight = self.backbone.embedding.weight

    # Hugging Faceから事前学習済みのMamba2の内部パラメータをおろす
    @staticmethod
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
        # 評価（推論）モード
        model.eval()
        return model

    def forward(
        self, input_ids: LongTensor, h: list[InferenceCache] | list[None] | None = None
    ) -> tuple[LongTensor, list[InferenceCache]]:
        """
        Arguments
            input_ids: (batch, seqlen) tokens from `EleutherAI/gpt-neox-20b` tokenizer
            h: hidden states for inference step. If present the constant-time
               (wrt sequence length) inference path will be taken, input_ids
               should have shape (batch, 1) containing the next batch of prompt
               token.

        Return (logits, h)
            logits: (batch, seqlen, vocab_size)
            h: updated inference cache after processing `input_ids`
        """

        # 入力テキストが何トークンかを測る
        seqlen = input_ids.shape[1]

        # 1トークン目の場合（h_0の場合）、各レイヤーの隠れ状態をリセット
        if h is None:
            h = [None for _ in range(self.args.n_layer)]

        # 単語埋め込み処理
        x = self.backbone.embedding(input_ids)
        x = crypten.cryptensor(x)

        # 全レイヤーにおいてこの処理を繰り返す
        for i, layer in enumerate(self.backbone.layers):

            # 正規化を行った単語埋め込みを過去の隠れ状態h_tとともにSSM処理をおこなう
            y, h[i] = layer.mixer(layer.norm(x), h[i])

            # 上の結果と、もとの単語埋め込みを足し、次レイヤーへ送る
            x = y + x

        # 最後に正規化する
        x = self.backbone.norm_f(x)

        # トークンIDと同じ次元に変換し、単語の予測スコアとして保持
        logits = self.lm_head(x)

        # 入力トークン文の予測確率と記憶(キャッシュ)を保存
        return logits[:, :seqlen], cast(list[InferenceCache], h)

    def generate(
        self,
        input_ids: LongTensor,
        max_new_length: int = 20,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        eos_token_id: int = 0,
    ) -> Iterable[tuple[int, list[InferenceCache]]]:
        
        # prefix: 最後の1トークンを除く過去のトークンすべて
        # tokens: 最後の1トークンだけ。これを「現在の入力」として、次の文字の予測を開始
        prefix, tokens = input_ids[:-1], input_ids[-1:].unsqueeze(0)

        # Process prompt
        # The input sequence to forward (non-inference path) must have length multiple that of chunk_size.
        # We split out excess tokens so that n_chunked tokens can be processed by one forward call and
        # process the rest in multiple inference steps.

        # チャンクサイズに分けて処理を行う
        n_chunked = (prefix.shape[0] // self.args.chunk_size) * self.args.chunk_size

        # チャンクで高速計算し、隠れ状態hを初期化
        if n_chunked > 0:
            _, h = self(prefix[:n_chunked].unsqueeze(0), None)

        # チャンクから溢れたものは、hをゼロから立ち上げる
        else:
            h = [
                InferenceCache.alloc(1, self.args, device=self.device)
                for _ in range(self.args.n_layer)
            ]
        
        # 一トークンずつ順番に処理
        for i in range(n_chunked, prefix.shape[0]):
            _, h = self(prefix[i : i + 1].unsqueeze(0), h)

        # Generate
        for _ in range(max_new_length):
            with torch.no_grad():
                out, h = self(tokens, h)
            logits = out[0, -1]
            #スコアを温度で割る
            if temperature != 1.0:
                logits = logits / temperature
            # Top-Kフィルター
            if top_k > 0:
                #確率が高い順にK個だけ残す
                indices_to_remove = logits < torch.topk(logits, k=top_k)[0][-1]
                logits[indices_to_remove] = -torch.inf
            # Top-Pフィルター
            if top_p < 1.0:
                #累積確率で上位top_pのしきい値に入らないトークンは切る
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cum_probs > top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = False
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[indices_to_remove] = -torch.inf
            # 確率分布計算
            probs = F.softmax(logits, dim=-1)
            # 確率分布を基に、次のトークンを予測
            next_token = torch.multinomial(probs, num_samples=1)

            # 最後のトークンなら終了
            if next_token.item() == eos_token_id:
                return
            
            # 作ったトークンを次の入力にセットし、決定したトークンIDと最新の隠れ状態hを呼び出し元へ送る
            tokens = next_token.unsqueeze(0)

            yield cast(int, next_token.item()), h


class Mamba2(nn.Module):
    def __init__(self, args: Mamba2Config, device: Device = None):
        super().__init__()
        self.args = args
        self.device = device

        # z, x, B, C, dtが格納されたベクトル
        # z: ゲート（フィルター）用のデータ
        # x: メインの入力データ
        # B, C: 状態空間モデル（SSM）の行列データ
        # dt: 時間のステップ幅（タイムステップ）
        #d_model次元をd_in_proj次元へ拡張
        d_in_proj = 2 * args.d_inner + 2 * args.d_state + args.nheads
        self.in_proj = nn.Linear(args.d_model, d_in_proj, bias=False, device=device)

        # 直近4トークン分のローカルな並びを混ぜ合わせるための1次元畳み込み
        conv_dim = args.d_inner + 2 * args.d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            kernel_size=args.d_conv,
            groups=conv_dim,
            padding=args.d_conv - 1,
            device=device,
        )

        # 変数定義
        # 時間ステップバイアス
        self.dt_bias = crypten.cryptensor(nn.Parameter(torch.empty(args.nheads, device=device)))
        # システム行列A（対数）
        self.A_log = crypten.cryptensor(nn.Parameter(torch.empty(args.nheads, device=device)))
        # 残差結合にかかる係数D
        self.D = crypten.cryptensor(nn.Parameter(torch.empty(args.nheads, device=device)))

        # 正規化
        self.norm = RMSNorm(args.d_inner, device=device)
        # d_inner次元からd_model次元へ変換
        self.out_proj = nn.Linear(args.d_inner, args.d_model, bias=False, device=device)

    def forward(self, u: Tensor, h: InferenceCache | None = None):
        """
        Arguments
            u: (batch, seqlen, d_model) input. seqlen should be a multiple of chunk_size.
            h: hidden states for inference step. Initialized to 0s if not present.

        Return (y, h)
            y: (batch, seqlen, d_model) output
            h: updated inference cache after processing `u`
        """

        #すでに隠れ状態がある場合はstep関数を用いる
        if h:
            return self.step(u, h)

        # 対数から負の実数に戻す
        A = -torch.exp(self.A_log)  # (nheads,)
        # 次元拡張を行う
        zxbcdt = self.in_proj(u)  # (batch, seqlen, d_in_proj)
        #巨大化したデータを、ゲート用の z、メインデータの xBC、タイムステップ（時間の進み幅）の dt の3つに切り分け
        z, xBC, dt = torch.split(
            zxbcdt,
            [
                self.args.d_inner,
                self.args.d_inner + 2 * self.args.d_state,
                self.args.nheads,
            ],
            dim=-1,
        )
        # 活性化関数に通す
        dt = MPCMamba_Function.softplus(dt + self.dt_bias)  # (batch, seqlen, nheads)

        # 将来の1文字生成（step）のために、現在の状態を記録（パディング処理）
        conv_state = F.pad(
            rearrange(xBC, "b l d -> b d l"), (self.args.d_conv - u.shape[1], 0)
        )
        # 1次元畳み込みを実行し、SiLUで活性化
        xBC = MPCMamba_Function.silu(
            self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)[:, : u.shape[1], :]
        )  # (batch, seqlen, d_inner + 2 * d_state))
        # さらに、x, B, C の3つに切り分ける
        x, B, C = torch.split(
            xBC, [self.args.d_inner, self.args.d_state, self.args.d_state], dim=-1
        )

        # SSMの計算処理
        # h ： n_head（ヘッドの個数）
        # b ： batch_size（バッチサイズ、データの個数）
        # l ： seqlen（シーケンス長、テキストの長さ）
        # p ： headdim （1ヘッドあたりの次元数）
        # n ： d_state（Mambaの状態空間の次元数）
        x = rearrange(x, "b l (h p) -> b l h p", p=self.args.headdim)
        y, ssm_state = ssd(
            x * dt.unsqueeze(-1),
            A * dt,
            rearrange(B, "b l n -> b l 1 n"),
            rearrange(C, "b l n -> b l 1 n"),
            self.args.chunk_size,
            device=self.device,
        )

        # スキップ結合
        y = y + x * self.D.unsqueeze(-1)
        #次元調整
        y = rearrange(y, "b l h p -> b l (h p)")
        #正規化
        y = self.norm(y, z)
        # d_modelに次元変換
        y = self.out_proj(y)

        # 畳み込みの直近状態（conv_state）とSSMの長期記憶（ssm_state）をキャッシュに追加
        h = InferenceCache(conv_state, ssm_state)
        return y, h

    # 直前の隠れ状態hだけを更新して、1トークンを素早く生成する関数
    def step(self, u: Tensor, h: InferenceCache) -> tuple[Tensor, InferenceCache]:
        """Take a single inference step for the current input and hidden state

        Unlike attention-based models, RNN-based models (eg Mamba) does not need
        to look back at all the past tokens to generate a new token. Instead a
        hidden state (initialized to 0s initially) is updated for each input and
        passed to the next inference step. This means that the total inference
        time is linear with respect to the sequence length instead of quadratic
        in attention's case.

        Arguments
            u: (batch, 1, d_model)
            h: initial/running hidden state

        Return (y, h)
            y: (batch, 1, d_model)
            h: updated hidden state
        """
        # 1トークンだけ入ってきているかの確認
        assert u.shape[1] == 1, "Only one token can be decoded per inference step"

        # 次元をd_in_projとして巨大化する
        zxbcdt = self.in_proj(u.squeeze(1))  # (batch, d_in_proj)
        z, xBC, dt = torch.split(
            zxbcdt,
            [
                self.args.d_inner,
                self.args.d_inner + 2 * self.args.d_state,
                self.args.nheads,
            ],
            dim=-1,
        )

        # 隠れ状態4トークン分に対して、左に1つずらす（一番古いトークンを捨てる）
        h.conv_state.copy_(torch.roll(h.conv_state, shifts=-1, dims=-1))
        # 空いた一番右端（最新の位置）に、今入ってきた新入りトークン（xBC）を滑り込ませる
        h.conv_state[:, :, -1] = xBC
        # 直近4トークン分に対して、畳み込み計算（掛け算と足し算）を行う
        xBC = torch.sum(
            h.conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1
        )
        xBC += self.conv1d.bias
        xBC = silu(xBC)

        x, B, C = torch.split(
            xBC, [self.args.d_inner, self.args.d_state, self.args.d_state], dim=-1
        )
        A = -torch.exp(self.A_log)  # (nheads,)

        # SSMの更新式
        dt = F.softplus(dt + self.dt_bias)  # (batch, nheads)
        dA = torch.exp(dt * A)  # (batch, nheads)
        x = rearrange(x, "b (h p) -> b h p", p=self.args.headdim)
        dBx = torch.einsum("bh, bn, bhp -> bhpn", dt, B, x)
        h.ssm_state.copy_(h.ssm_state * rearrange(dA, "b h -> b h 1 1") + dBx)
        y = torch.einsum("bhpn, bn -> bhp", h.ssm_state, C)
        y = y + rearrange(self.D, "h -> h 1") * x

        # 後処理（正規化と次元整理）
        y = rearrange(y, "b h p -> b (h p)")
        y = self.norm(y, z)
        y = self.out_proj(y)

        return y.unsqueeze(1), h

# 「過去から現在にいたるまで、記憶がどれくらい連続して減衰（忘却）してきたか」の累積合計を、巨大な行列として一発で計算するための下請け関数
def segsum(x: Tensor, device: Device = None) -> Tensor:
    """Stable segment sum calculation.

    `exp(segsum(A))` produces a 1-semiseparable matrix, which is equivalent to a scalar SSM.

    Source: https://github.com/state-spaces/mamba/blob/219f03c840d5a44e7d42e4e728134834fddccf45/mamba_ssm/modules/ssd_minimal.py#L23-L32
    """

    # x = ΔtA
    T = x.size(-1)
    # T x T次元の行列変換
    x = repeat(x, "... d -> ... d e", e=T)

    # 対角線下がTrue(対角線を含まない)、上がFalseとなるMaskを生成し、未来からの影響をシャットアウト
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=-1)
    x = x.masked_fill(~mask, 0)

    # 行列の縦方向（dim=-2）に向かって、数値を上から下へと累積足し算（累積和）
    x_segsum = torch.cumsum(x, dim=-2)

    # 対角線下がTrue(対角線を含む)、上がFalseとなるMaskを生成し、未来の状態を0にする（exp(-∞)=0）
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)

    # 累積減衰行列を返す
    return x_segsum

# Mamba2(SSD)の処理
def ssd(x, A, B, C, chunk_size, initial_states=None, device: Device = None):
    """Structed State Space Duality (SSD) - the core of Mamba-2

    This is almost the exact same minimal SSD code from the blog post.

    Arguments
        x: (batch, seqlen, n_heads, d_head)
        A: (batch, seqlen, n_heads)
        B: (batch, seqlen, n_heads, d_state)
        C: (batch, seqlen, n_heads, d_state)

    Return
        y: (batch, seqlen, n_heads, d_head)

    Source
     1. https://tridao.me/blog/2024/mamba2-part3-algorithm/
     2. https://github.com/state-spaces/mamba/blob/219f03c840d5a44e7d42e4e728134834fddccf45/mamba_ssm/modules/ssd_minimal.py#L34-L78
    """
    assert x.shape[1] % chunk_size == 0

    # Rearrange into chunks 
    # Step 1, 2 and 4 of SSD can be computed in parallel for each chunk across devices (sequence parallel)
    # This is not implemented and left as an exercise for the reader 😜
    # データをチャンク（塊）にぶつ切りにしている場所（GPUによる高速化計算は未実装）
    x, A, B, C = [
        rearrange(m, "b (c l) ... -> b c l ...", l=chunk_size) for m in (x, A, B, C)
    ]

    # チャンクの内部（長さ l の方向）に向かって、減衰率 A の累積足し算を行い、キープしておく
    A = rearrange(A, "b c l h -> b h c l")
    A_cumsum = torch.cumsum(A, dim=-1)

    # 1. Compute the output for each intra-chunk (diagonal blocks)
    # Y =(L⊙CB)X
    L = torch.exp(segsum(A, device=device))
    Y_diag = torch.einsum("bclhn, bcshn, bhcls, bcshp -> bclhp", C, B, L, x)

    # 2. Compute the state for each intra-chunk
    # (right term of low-rank factorization of off-diagonal blocks; B terms)
    # チャンク内の各単語の記憶が、チャンクの境界線に到達した時にどれくらい弱まっているか（引き継ぎ用の減衰率）を計算
    decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
    states = torch.einsum("bclhn, bhcl, bclhp -> bchpn", B, decay_states, x)

    # 3. Compute the inter-chunk SSM recurrence; produces correct SSM states at chunk boundaries
    # (middle term of factorization of off-diag blocks; A terms)
    # 過去の履歴がない場合は初期化
    if initial_states is None:
        initial_states = torch.zeros_like(states[:, :1])
    states = torch.cat([initial_states, states], dim=1)
    # チャンクをまたぐ時の減衰
    decay_chunk = torch.exp(segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0)), device=device))
    new_states = torch.einsum("bhzc, bchpn -> bzhpn", decay_chunk, states)
    states, final_state = new_states[:, :-1], new_states[:, -1]

    # 4. Compute state -> output conversion per chunk
    # (left term of low-rank factorization of off-diagonal blocks; C terms)
    # 前のチャンクから引き継いだ記憶を、各チャンク内の64文字の単語たちに分配して、フェーズ1の結果と足し算
    state_decay_out = torch.exp(A_cumsum)
    Y_off = torch.einsum("bclhn, bchpn, bhcl -> bclhp", C, states, state_decay_out)

    # Add output of intra-chunk and inter-chunk terms (diagonal and off-diagonal blocks)
    Y = rearrange(Y_diag + Y_off, "b c l h p -> b (c l) h p")

    return Y, final_state


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5, device: Device = None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d, device=device))

    def forward(self, x, z=None):
        if z is not None:
            x = x * MPCMamba_Function.silu(z)
        
        return x * MPCMamba_Function.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

class MPCMamba_Function():

    def exp(self,x,n=8):
        power_of_two = 2 ** n
        y = 1.0 + x / power_of_two
        for _ in range(n):
            y = y * y
        return y

    def reciprocal(self,x,n=8):
        y = 3.0*self.exp(0.5-x) + 0.0003
        for _ in range(n):
            y = y * (2.0-x*y)
        return y

    def log(self,x,n=3,k=5):
        y = x/120.0 - 20.0 * self.exp(-2.0*x-1.0) + 3.0
        for _ in range(n):
            t = 1.0 - x * self.exp(-y)
            
            # ln(1 - t_n)のテイラー項
            taylor_term = 0.0
            t_pow = 1.0
            for i in range(1,k+1):
                t_pow = t_pow * t
                taylor_term += (1.0/i) * t_pow
            y = y - taylor_term
        return y

    def rsqrt(self,x,y_initial=None,n=5):
        if y_initial is None:
            y = (2.2*self.exp(-(x*0.5 + 0.2)) + 0.2 -x) * 2 ** (-10)
        else:
            y = y_initial if crypten.is_encrypted_tensor(y_initial) else crypten.cryptensor(y_initial)

        for _ in range(n):
            y = 0.5 * y * (3.0 - x * y * y)
        return y
    
    def silu(self,x):
        base = self.exp(-x) + 1.0
        r = self.rsqrt(base,y_initial=0.75)
        sign_flag = (1.0 - x.sign()) / 2.0
        r = sign_flag * (1.0-r) + (1.0-sign_flag)*r
        return base * r
    
    def softplus(self,x):
        return self.log(self.exp(x) + 1.0)
    



    



