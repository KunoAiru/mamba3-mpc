"""
mpcmamba2-minimal
==============

A minimal, single-file implementation of the Mamba-2 model in PyTorch.

> **Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality**
> Authors: Tri Dao, Albert Gu
> Paper: https://arxiv.org/abs/2405.21060
"""

from dataclasses import dataclass
from typing import Iterable, NamedTuple, TypeAlias, cast

import torch
import crypten
from crypten.mpc import MPCTensor
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
    conv_state: MPCTensor  # (batch, d_inner + 2 * d_state, d_conv)
    ssm_state: MPCTensor  # (batch, nheads, headdim, d_state)

    @staticmethod
    def alloc(batch_size: int, args: Mamba2Config, device: Device = None):
        cache1 = torch.zeros(
                batch_size, args.d_inner + 2 * args.d_state, args.d_conv, device=device
            )
        cache2 = torch.zeros(
                batch_size, args.nheads, args.headdim, args.d_state, device=device
            )
        crypt_cache1 = crypten.cryptensor(cache1)
        crypt_cache2 = crypten.cryptensor(cache2)
        return InferenceCache(
            crypt_cache1,
            crypt_cache2
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
                                norm=RMSNorm(args.d_model,debug=1,device=device),
                            )
                        )
                        for _ in range(args.n_layer)
                    ]
                ),

                # 最後に最終的な層正規化を行う
                norm_f=RMSNorm(args.d_model,debug=3, device=device),
            )
        )

        # d_model(単語埋め込みベクトルの次元) → vocab_size(トークンIDのスコアの次元)へ変換する線形層
        self.lm_head = nn.Linear(
            args.d_model, args.vocab_size, bias=False, device=device
        )

        # 入口のembeddingの重みと、出口のlm_headで最終出力された重みを共有する
        self.lm_head.weight = self.backbone.embedding.weight

        self.lm_head_weight_crypt = None
    
    def encrypt_lm_head(self):
        if self.lm_head_weight_crypt is None:
            if isinstance(self.lm_head.weight, crypten.mpc.MPCTensor):
                self.lm_head_weight_crypt = self.lm_head.weight
            else:
                self.lm_head_weight_crypt = crypten.cryptensor(self.lm_head.weight).to(self.device)
    
    def load_encrypted_state_dict(self, encrypted_state_dict):
        """
        ユーザー側で暗号化された重み（CrypTensorの辞書）を受け取り、
        モデルのパラメータを完全に暗号化状態に置き換える
        """
        for name, r_tensor in encrypted_state_dict.items():
            attrs = name.split('.')
            submodule = self
            for attr in attrs[:-1]:
                submodule = getattr(submodule, attr)
            
            if attrs[-1] in submodule._parameters:
                del submodule._parameters[attrs[-1]]
            if attrs[-1] in submodule._buffers:
                del submodule._buffers[attrs[-1]]
            
            setattr(submodule, attrs[-1], r_tensor)

    def forward(
        self, input_onehot_mpc: MPCTensor, h: list[InferenceCache] | list[None] | None = None
    ) -> tuple[MPCTensor, list[InferenceCache]]:

        # 入力テキストが何トークンかを測る
        seqlen = input_onehot_mpc.shape[1]

        # 1トークン目の場合（h_0の場合）、各レイヤーの隠れ状態をリセット
        if h is None:
            h = [None for _ in range(self.args.n_layer)]

        # 単語埋め込み処理
        x = input_onehot_mpc.matmul(self.backbone.embedding.weight) # (batch, seqlen, d_model)

        # 全レイヤーにおいてこの処理を繰り返す
        for i, layer in enumerate(self.backbone.layers):

            # 正規化を行った単語埋め込みを過去の隠れ状態h_tとともにSSM処理をおこなう
            y, h[i] = layer.mixer(layer.norm(x), h[i])

            # 上の結果と、もとの単語埋め込みを足し、次レイヤーへ送る
            x = y + x

        # 最後に正規化する
        x = self.backbone.norm_f(x)

        # トークンIDと同じ次元に変換し、単語の予測スコアとして保持
        self.encrypt_lm_head()
        logits = x.matmul(self.lm_head_weight_crypt.t())

        # 入力トークン文の予測確率と記憶(キャッシュ)を保存
        return logits[:, :seqlen], cast(list[InferenceCache], h)

    def predict_next_logit_mpc(
        self, 
        current_token_onehot: MPCTensor, 
        h: list[InferenceCache] | list[None] | None = None
    ) -> tuple[MPCTensor, list[InferenceCache]]:
        """
        サーバー側で1トークン分の秘密計算を行い、次のロジットのシェアを返す。
        
        Arguments:
            current_token_onehot: (batch=1, seqlen=1, vocab_size) の暗号化One-hotテンソル
            h: 前ステップまでの暗号化された隠れ状態（キャッシュ）
        """
        if h is None:
            h = [None for _ in range(self.args.n_layer)]

        # nn.Embeddingの代わりにOne-hot行列積で暗号化埋め込みベクトルを作る
        x = current_token_onehot.matmul(self.backbone.embedding.weight)  # (1, 1, d_model)

        # 全レイヤーでSSM処理を繰り返す (すべて暗号化空間)
        for i, layer in enumerate(self.backbone.layers):
            y, h[i] = layer.mixer(layer.norm(x), h[i])
            x = y + x

        # 最終正規化
        x = self.backbone.norm_f(x)

        # 出力ロジットの計算
        self.encrypt_lm_head()
        logits = x.matmul(self.lm_head_weight_crypt.t())  # (1, 1, vocab_size)

        # 暗号化されたロジットと、更新されたキャッシュを返す
        return logits, h


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
        self.dt_bias = nn.Parameter(torch.empty(args.nheads, device=device))
        # システム行列A（対数）
        self.A_log = nn.Parameter(torch.empty(args.nheads, device=device))
        # 残差結合にかかる係数D
        self.D = nn.Parameter(torch.empty(args.nheads, device=device))

        # 正規化
        self.norm = RMSNorm(args.d_inner,debug=2, device=device)
        # d_inner次元からd_model次元へ変換
        self.out_proj = nn.Linear(args.d_inner, args.d_model, bias=False, device=device)

        self.in_proj_weight_crypt = None
        self.out_proj_weight_crypt = None
        self.dt_bias_crypt = None
        self.A_log_crypt = None
        self.D_crypt = None
        self.conv1d_weight_crypt = None
        self.conv1d_bias_crypt = None
    
    def encrypt_weights(self):
        if self.in_proj_weight_crypt is None:
            if isinstance(self.in_proj.weight, crypten.mpc.MPCTensor):
                self.in_proj_weight_crypt = self.in_proj.weight
                self.out_proj_weight_crypt = self.out_proj.weight
                self.dt_bias_crypt = self.dt_bias
                self.A_log_crypt = self.A_log
                self.D_crypt = self.D
            else:
                self.in_proj_weight_crypt = crypten.cryptensor(self.in_proj.weight).to(self.device)
                self.out_proj_weight_crypt = crypten.cryptensor(self.out_proj.weight).to(self.device)
                self.dt_bias_crypt = crypten.cryptensor(self.dt_bias).to(self.device)
                self.A_log_crypt = crypten.cryptensor(self.A_log).to(self.device)
                self.D_crypt = self.D
        
        if self.conv1d_weight_crypt is None:
            if isinstance(self.conv1d.weight, crypten.mpc.MPCTensor):
                self.conv1d_weight_crypt = self.conv1d.weight
                self.conv1d_bias_crypt = self.conv1d.bias
            else:
                self.conv1d_weight_crypt = crypten.cryptensor(self.conv1d.weight).to(self.device)
                self.conv1d_bias_crypt = crypten.cryptensor(self.conv1d.bias).to(self.device)
    
    def causal_conv1d(self, x: MPCTensor) -> MPCTensor:
        """CrypTen（MPC）環境で動作する1D畳み込み
        
        Arguments:
            x: (batch, seqlen, conv_dim) の暗号化テンソル
            
        Returns:
            (batch, seqlen, conv_dim) の畳み込み後の暗号化テンソル
        """
        # 形状の確認と変換 (batch, seqlen, conv_dim) -> (batch, conv_dim, seqlen)
        x_t = x.transpose(1, 2)
        batch, conv_dim, seqlen = x_t.shape[0], x_t.shape[1], x_t.shape[2]
        
        # 重み (conv_dim, 1, kernel_size) -> (conv_dim, kernel_size) に平坦化
        _, _, k_size = self.conv1d_weight_crypt.shape
        w_flat = self.conv1d_weight_crypt.reshape(conv_dim, k_size)
        
        # 1. 因果性（Causal）を担保するため、時系列の「前側（左側）」にだけパディングを行う
        pad_left = k_size - 1
        zeros_pad = crypten.cryptensor(torch.zeros(batch, conv_dim, pad_left, device=self.device)).to(self.device)
        x_padded = crypten.cat([zeros_pad, x_t], dim=-1) # (batch, conv_dim, pad_left + seqlen)
        
        # 出力用のバッファを0で初期化
        conv_out = crypten.cryptensor(torch.zeros(batch, conv_dim, seqlen, device=self.device)).to(self.device)
        
        # 2. カーネルサイズ分（Mamba2の標準設定では4回）のループで畳み込みを計算
        # 各カーネル位置の重みを、時間軸をずらした入力スライスに対して要素ごとに掛け算して足し合わせる
        for k in range(k_size):
            # 重みのk番目の要素: (conv_dim, 1)
            w_k = w_flat[:, k].unsqueeze(-1)
            
            # 入力から対応する時間軸の窓（ウィンドウ）を抽出
            slice_k = x_padded[:, :, k : k + seqlen]
            
            # 積和算
            conv_out = conv_out + (slice_k * w_k)
            
        # 3. バイアスの加算: (conv_dim,) -> (1, conv_dim, 1) に拡張して足す
        conv_out = conv_out + self.conv1d_bias_crypt.unsqueeze(0).unsqueeze(-1)
        
        # 形状を元に戻す (batch, conv_dim, seqlen) -> (batch, seqlen, conv_dim)
        return conv_out.transpose(1, 2)

    def forward(self, u: MPCTensor, h: InferenceCache | None = None):
        """
        Arguments
            u: (batch, seqlen, d_model) input. seqlen should be a multiple of chunk_size.
            h: hidden states for inference step. Initialized to 0s if not present.

        Return (y, h)
            y: (batch, seqlen, d_model) output
            h: updated inference cache after processing `u`
        """
        # SSMの計算処理
        # h ： n_head（ヘッドの個数）
        # b ： batch_size（バッチサイズ、データの個数）
        # l ： seqlen（シーケンス長、テキストの長さ）
        # p ： headdim （1ヘッドあたりの次元数）
        # n ： d_state（Mambaの状態空間の次元数）

        self.encrypt_weights()

        #すでに隠れ状態がある場合はstep関数を用いる
        if h:
            return self.step(u, h)

        # 対数から負の実数に戻す
        A = -MPCMamba_Function.exp(self.A_log_crypt,debug=1)  # (nheads,)

        # 次元拡張を行う
        zxbcdt = u.matmul(self.in_proj_weight_crypt.t()) # (batch, seqlen, d_in_proj)
        print(f"zxbcdt max_val : {zxbcdt.get_plain_text().max().item()}")
        
        #巨大化したデータを、ゲート用の z、メインデータの xBC、タイムステップ（時間の進み幅）の dt の3つに切り分け
        #z, xBC, dt = torch.split(zxbcdt,[self.args.d_inner,self.args.d_inner + 2 * self.args.d_state,self.args.nheads,],dim=-1,)
        idx1 = self.args.d_inner
        idx2 = idx1 + (self.args.d_inner + 2 * self.args.d_state)
        z = zxbcdt[..., :idx1]
        xBC = zxbcdt[..., idx1:idx2]
        dt = zxbcdt[..., idx2:]

        #print(f"dt max_val : {dt.get_plain_text().max().item()}")
        #print(f"xBC max_val : {xBC.get_plain_text().max().item()}")
        #print(f"z max_val : {z.get_plain_text().max().item()}")

        # 活性化関数に通す
        dt = MPCMamba_Function.softplus(dt + self.dt_bias_crypt)  # (batch, seqlen, nheads)
        #print(f"dt max_val after softplus: {dt.get_plain_text().max().item()}")

        # 将来の1文字生成（step）のために、現在の状態を記録（パディング処理）
        # xBC = (b l d -> b d l)
        xBC_t = xBC.transpose(1, 2)
        if u.shape[1] >= self.args.d_conv:
            conv_state = xBC_t[..., -self.args.d_conv:]
        else:
            conv_state = MPCMamba_Function.pad_left(xBC_t, self.args.d_conv - u.shape[1])
        # 1次元畳み込みを実行し、SiLUで活性化
        xBC = self.causal_conv1d(xBC)
        xBC = MPCMamba_Function.silu(xBC) # (batch, seqlen, d_inner + 2 * d_state)

        # さらに、x, B, C の3つに切り分ける
        #x, B, C = torch.split(xBC, [self.args.d_inner, self.args.d_state, self.args.d_state], dim=-1)
        idx1 = self.args.d_inner
        idx2 = idx1 + self.args.d_state
        
        x = xBC[..., :idx1]
        B = xBC[..., idx1:idx2]
        C = xBC[..., idx2:]

        # x = rearrange(x, "b l (h p) -> b l h p", p=self.args.headdim)
        batch, seqlen, d_inner = x.shape[0], x.shape[1], x.shape[2]
        nheads = d_inner // self.args.headdim
        x = x.reshape(batch, seqlen, nheads, self.args.headdim)

        y, ssm_state = self.ssd(
            x * dt.unsqueeze(-1),
            A * dt,
            B.unsqueeze(-2),  # rearrange(B, "b l n -> b l 1 n") 
            C.unsqueeze(-2),  # rearrange(C, "b l n -> b l 1 n") 
        )

        # スキップ結合
        y = y + x * self.D_crypt.unsqueeze(-1)

        #次元調整
        # y = rearrange(y, "b l h p -> b l (h p)")
        batch, seqlen, nheads, headdim = y.shape[0], y.shape[1], y.shape[2], y.shape[3]
        y = y.reshape(batch, seqlen, nheads * headdim)

        #正規化
        y = self.norm(y, z)
        # d_modelに次元変換
        y = y.matmul(self.out_proj_weight_crypt.t())

        # 畳み込みの直近状態（conv_state）とSSMの長期記憶（ssm_state）をキャッシュに追加
        h = InferenceCache(conv_state, ssm_state)
        return y, h

    # 直前の隠れ状態hだけを更新して、1トークンを素早く生成する関数
    def step(self, u: MPCTensor, h: InferenceCache) -> tuple[MPCTensor, InferenceCache]:
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
        zxbcdt = u.squeeze(1).matmul(self.in_proj_weight_crypt.t())  # (batch, d_in_proj)
        #z, xBC, dt = torch.split(zxbcdt,[self.args.d_inner,self.args.d_inner + 2 * self.args.d_state,self.args.nheads,],dim=-1,)
        idx1 = self.args.d_inner
        idx2 = idx1 + (self.args.d_inner + 2 * self.args.d_state)

        z = zxbcdt[..., :idx1]
        xBC = zxbcdt[..., idx1:idx2]
        dt = zxbcdt[..., idx2:]

        # 隠れ状態4トークン分に対して、左に1つずらす（一番古いトークンを捨てる）
        # h.conv_state.copy_(torch.roll(h.conv_state, shifts=-1, dims=-1))
        shifted = h.conv_state[:, :, 1:]
        new_conv_state = crypten.cat([shifted, xBC.unsqueeze(-1)], dim=-1)
        #h.conv_state[:, :, -1] = xBC
        h = InferenceCache(new_conv_state, h.ssm_state)

        # 直近4トークン分に対して、畳み込み計算（掛け算と足し算）を行う
        #xBC = torch.sum(h.conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)
        #xBC += self.conv1d.bias
        c_dim, _, k_size = self.conv1d_weight_crypt.shape
        w_flat = self.conv1d_weight_crypt.reshape(c_dim, k_size)    
        xBC = (h.conv_state * w_flat.unsqueeze(0)).sum(dim=-1)
        xBC = xBC + self.conv1d_bias_crypt.unsqueeze(0)

        xBC = MPCMamba_Function.silu(xBC)

        #x, B, C = torch.split(xBC, [self.args.d_inner, self.args.d_state, self.args.d_state], dim=-1)
        idx1 = self.args.d_inner
        idx2 = idx1 + self.args.d_state
        
        x = xBC[..., :idx1]
        B = xBC[..., idx1:idx2]
        C = xBC[..., idx2:]
        A = -MPCMamba_Function.exp(self.A_log_crypt,debug=2)  # (nheads,)

        # SSMの更新式
        dt = MPCMamba_Function.softplus(dt + self.dt_bias_crypt)  # (batch, nheads)
        dA = MPCMamba_Function.exp(dt * A,debug=3)  # (batch, nheads)
        # x = rearrange(x, "b (h p) -> b h p", p=self.args.headdim)
        batch,d_inner = x.shape[0],x.shape[1]
        nheads = d_inner // self.args.headdim
        x = x.reshape(batch,nheads,self.args.headdim)

        # dBx = torch.einsum("bh, bn, bhp -> bhpn", dt, B, x)
        # 各テンソルに unsqueeze を適用して4次元 (batch, nheads, headdim, d_state) に拡張し、要素積
        # dt: (b, h) -> (b, h, 1, 1)
        # B:  (b, n) -> (b, 1, 1, n)
        # x:  (b, h, p) -> (b, h, p, 1)
        dBx = dt.unsqueeze(-1).unsqueeze(-1) * B.unsqueeze(1).unsqueeze(1) * x.unsqueeze(-1)

        # h.ssm_state.copy_(h.ssm_state * rearrange(dA, "b h -> b h 1 1") + dBx)
        dA_4d = dA.unsqueeze(-1).unsqueeze(-1)
        new_ssm_state = h.ssm_state * dA_4d + dBx
        h = InferenceCache(h.conv_state, new_ssm_state)

        # y = torch.einsum("bhpn, bn -> bhp", h.ssm_state, C)
        # C: (batch, d_state) -> (batch, 1, 1, d_state)
        C_4d = C.unsqueeze(1).unsqueeze(1)
        # y：(batch, nheads, headdim, d_state) 
        y_4d = h.ssm_state * C_4d
        # 縮小したい次元（最後の次元: dim=-1）方向に足し合わせ(batch, nheads, headdim) に次元縮小
        y = y_4d.sum(dim=-1)

        # y = y + rearrange(self.D, "h -> h 1") * x
        y = y + self.D_crypt.unsqueeze(-1) * x

        # 後処理（正規化と次元整理）
        # y = rearrange(y, "b h p -> b (h p)")
        batch,nheads,headdim = y.shape[0],y.shape[1],y.shape[2]
        y = y.reshape(batch,nheads*headdim)

        y = self.norm(y, z)
        y = y.matmul(self.out_proj_weight_crypt.t())

        return y.unsqueeze(1), h
    
    # Mamba2(SSD)の処理
    def ssd(self, x: MPCTensor, A: MPCTensor, B: MPCTensor, C: MPCTensor, initial_states: MPCTensor | None = None) -> tuple[MPCTensor, MPCTensor]:
        """Structed State Space Duality (SSD) - the core of Mamba-2

        This is almost the exact same minimal SSD code from the blog post.

        Arguments
            x: (batch, seqlen, n_heads, d_head)
            A: (batch, seqlen, n_heads)
            B: (batch, seqlen, n_heads, d_state)
            C: (batch, seqlen, n_heads, d_state)

        Return
            y: (batch, seqlen, n_heads, d_head)

            self.args.chunk_size,
            device=self.device,

        Source
         1. https://tridao.me/blog/2024/mamba2-part3-algorithm/
         2. https://github.com/state-spaces/mamba/blob/219f03c840d5a44e7d42e4e728134834fddccf45/mamba_ssm/modules/ssd_minimal.py#L34-L78
        """
        assert x.shape[1] % self.args.chunk_size == 0

        # Rearrange into chunks 
        # Step 1, 2 and 4 of SSD can be computed in parallel for each chunk across devices (sequence parallel)
        # This is not implemented and left as an exercise for the reader 😜
        # データをチャンク（塊）にぶつ切りにしている場所（GPUによる高速化計算は未実装）
        #x, A, B, C = [
        #    rearrange(m, "b (c l) ... -> b c l ...", l=chunk_size) for m in (x, A, B, C)
        #]
        #A = rearrange(A, "b c l h -> b h c l")
        batch_size = x.shape[0]
        num_chunks = x.shape[1] // self.args.chunk_size
        x = x.reshape(batch_size, num_chunks, self.args.chunk_size, self.args.nheads, self.args.headdim)
        A = A.reshape(batch_size, num_chunks, self.args.chunk_size, self.args.nheads)
        A = A.permute(0, 3, 1, 2)
        B = B.reshape(batch_size, num_chunks, self.args.chunk_size, self.args.d_state)
        C = C.reshape(batch_size, num_chunks, self.args.chunk_size, self.args.d_state)

        # チャンクの内部（長さ l の方向）に向かって、減衰率 A の累積足し算を行い、キープしておく
        A_cumsum = MPCMamba_Function.cumsum(A, dim=-1)

        # 1. Compute the output for each intra-chunk (diagonal blocks)
        # Y =(L⊙CB)X
        L = MPCMamba_Function.exp(segsum(A, device=self.device),debug=4)
        triu_mask_0 = torch.triu(torch.ones(self.args.chunk_size, self.args.chunk_size, device=self.device), diagonal=1)
        keep_mask = 1 - triu_mask_0
        L = L * crypten.cryptensor(keep_mask).to(self.device)

        #Y_diag = torch.einsum("bclhn, bcshn, bhcls, bcshp -> bclhp", C, B, L, x)
        # ターゲット形状: (b, c, l, s, h, p, n) になるように拡張
        # C: (b, c, l, n) -> (b, c, l, 1, 1, 1, n)
        # B: (b, c, s, n) -> (b, c, 1, s, 1, 1, n)
        # L: (b, h, c, l, s) -> (b, c, l, s, h, 1, 1)
        # x: (b, c, s, h, p) -> (b, c, 1, s, h, p, 1)
        L_perm = L.permute(0, 2, 3, 4, 1)

        # 各テンソルの元の形状
        # C: (b, c, l, n)
        # B: (b, c, s, n)
        # L_perm: (b, c, l, s, h)
        # x: (b, c, s, h, p)

        # (b, c, l, 1, n) * (b, c, 1, s, n) -> (b, c, l, s, n) -> sum(-1) -> (b, c, l, s)
        C_expanded = C.unsqueeze(3) # (b, c, l, 1, n)
        B_expanded = B.unsqueeze(2) # (b, c, 1, s, n)
        CB = (C_expanded * B_expanded).sum(dim=-1) # (b, c, l, s)

        # (b, c, l, s, 1) * (b, c, l, s, h) -> (b, c, l, s, h)
        CBL = CB.unsqueeze(-1) * L_perm # (b, c, l, s, h)
        # CBL (b, c, l, s, h, 1) * x (b, c, 1, s, h, p) -> sum(dim=3) -> (b, c, l, h, p)
        CBL_expanded = CBL.unsqueeze(-1) # (b, c, l, s, h, 1)
        x_expanded = x.unsqueeze(2)      # (b, c, 1, s, h, p)
        
        Y_diag = (CBL_expanded * x_expanded).sum(dim=3) # (b, c, l, h, p)

        # 2. Compute the state for each intra-chunk
        # (right term of low-rank factorization of off-diagonal blocks; B terms)
        # チャンク内の各単語の記憶が、チャンクの境界線に到達した時にどれくらい弱まっているか（引き継ぎ用の減衰率）を計算
        decay_states = MPCMamba_Function.exp(A_cumsum[:, :, :, -1:] - A_cumsum,debug=5)

        #states = torch.einsum("bclhn, bhcl, bclhp -> bchpn", B, decay_states, x)
        # ターゲット形状: (b, c, l, h, p, n)
        # B:            (b, c, l, n)    -> (b, c, l, 1, 1, n)
        # decay_states: (b, h, c, l)    -> (b, c, l, h, 1, 1)
        # x:            (b, c, l, h, p) -> (b, c, l, h, p, 1)
        decay_states_perm = decay_states.permute(0, 2, 3, 1)
        states_6d = (
            B.unsqueeze(3).unsqueeze(4) * decay_states_perm.unsqueeze(4).unsqueeze(-1) * x.unsqueeze(-1)
        )
        states = states_6d.sum(dim=2) # 結果: (b, c, h, p, n)

        # 3. Compute the inter-chunk SSM recurrence; produces correct SSM states at chunk boundaries
        # (middle term of factorization of off-diag blocks; A terms)
        # 過去の履歴がない場合は初期化
        if initial_states is None:
            crypt_initial_states = crypten.cryptensor(torch.zeros(batch_size, 1, self.args.nheads, self.args.headdim, self.args.d_state, device=self.device)).to(self.device)
        else:
            crypt_initial_states = initial_states
        states = crypten.cat([crypt_initial_states, states], dim=1)

        # チャンクをまたぐ時の減衰
        decay_chunk = MPCMamba_Function.exp(segsum(MPCMamba_Function.pad_left(A_cumsum[:, :, :, -1], 1), device=self.device),debug=6)
        num_chunks_plus1 = decay_chunk.size(-1)
        triu_mask_c = torch.triu(torch.ones(num_chunks_plus1, num_chunks_plus1, device=self.device), diagonal=1)
        keep_mask_c = 1 - triu_mask_c
        decay_chunk = decay_chunk * crypten.cryptensor(keep_mask_c).to(self.device)

        #new_states = torch.einsum("bhzc, bchpn -> bzhpn", decay_chunk, states)
        decay_chunk_exp = decay_chunk.unsqueeze(-1).unsqueeze(-1) # (b, h, z, c, 1, 1)
        # states を (b, c, h, p, n) -> (b, h, 1, c, p, n) に変形
        states_perm = states.permute(0, 2, 1, 3, 4).unsqueeze(2)
        
        new_states_6d = decay_chunk_exp * states_perm
        new_states = new_states_6d.sum(dim=3) # -> (b, h, z, p, n)
        new_states = new_states.permute(0, 2, 1, 3, 4) #-> (b, z, h, p, n)

        states, final_state = new_states[:, :-1], new_states[:, -1]

        # 4. Compute state -> output conversion per chunk
        # (left term of low-rank factorization of off-diagonal blocks; C terms)
        # 前のチャンクから引き継いだ記憶を、各チャンク内の64文字の単語たちに分配して、フェーズ1の結果と足し算
        state_decay_out = MPCMamba_Function.exp(A_cumsum,debug=7)

        #Y_off = torch.einsum("bclhn, bchpn, bhcl -> bclhp", C, states, state_decay_out)
        # ターゲット形状: (b, c, l, h, p, n)
        # C:               (b, c, l, n) -> (b, c, l, 1, 1, n)
        # states:          (b, c, h, p, n) -> (b, c, 1, h, p, n)
        # state_decay_out: (b, h, c, l) -> (b, c, l, h, 1, 1)
        state_decay_out_perm = state_decay_out.permute(0, 2, 3, 1)
        C_exp = C.unsqueeze(3).unsqueeze(4)       # (b, c, l, 1, 1, n)
        states_exp = states.unsqueeze(2)           # (b, c, 1, h, p, n)
        C_states = (C_exp * states_exp).sum(dim=-1) # (b, c, l, h, p)
        Y_off = C_states * state_decay_out_perm.unsqueeze(-1)

        # Add output of intra-chunk and inter-chunk terms (diagonal and off-diagonal blocks)
        #Y = rearrange(Y_diag + Y_off, "b c l h p -> b (c l) h p")
        Y_combined = Y_diag + Y_off
        Y = Y_combined.reshape(batch_size, num_chunks * self.args.chunk_size, self.args.nheads, self.args.headdim)

        return Y, final_state

# 「過去から現在にいたるまで、記憶がどれくらい連続して減衰（忘却）してきたか」の累積合計を、巨大な行列として一発で計算するための下請け関数
def segsum(x: MPCTensor, device: Device = None) -> MPCTensor:
    """Stable segment sum calculation.

    `exp(segsum(A))` produces a 1-semiseparable matrix, which is equivalent to a scalar SSM.

    Source: https://github.com/state-spaces/mamba/blob/219f03c840d5a44e7d42e4e728134834fddccf45/mamba_ssm/modules/ssd_minimal.py#L23-L32
    """

    # x = ΔtA
    T = x.size(-1)
    # T x T次元の行列変換
    # x = repeat(x, "... d -> ... d e", e=T)
    x = MPCMamba_Function.mpc_repeat_inter_chunk(x,T)

    # 対角線下がTrue(対角線を含まない)、上がFalseとなるMaskを生成し、未来からの影響をシャットアウト
    # mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=-1)
    # x = x.masked_fill(~mask, 0)
    tril_mask_minus1 = torch.tril(torch.ones(T, T, device=device), diagonal=-1)
    x = x * crypten.cryptensor(tril_mask_minus1).to(device)

    # 行列の縦方向（dim=-2）に向かって、数値を上から下へと累積足し算（累積和）
    x_segsum = MPCMamba_Function.cumsum(x, dim=-2)

    # 対角線下がTrue(対角線を含む)、上がFalseとなるMaskを生成し、未来の状態を0にする（exp(-∞)=0）
    # mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=0)
    # x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    triu_mask_0 = torch.triu(torch.ones(T, T, device=device), diagonal=1)
    keep_mask = 1 - triu_mask_0
    x_segsum = x_segsum * crypten.cryptensor(keep_mask).to(device)

    # 累積減衰行列を返す
    return x_segsum


class RMSNorm(nn.Module):
    def __init__(self, d: int, debug: int, eps: float = 1e-5, device: Device = None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(d, device=device))
        self.debug = debug

    def forward(self, x, z=None):
        if z is not None:
            x = x * MPCMamba_Function.silu(z)
        
        var = (x * x).sum(dim=-1, keepdim=True) / x.shape[-1]
        #print(f"RMSNorm dubug ID:{self.debug}")
        #print("var:", var.get_plain_text().abs().max().item())
        #print("x:", x.get_plain_text().abs().max().item())
        #print("result:", (x * MPCMamba_Function.rsqrt(var + self.eps) * self.weight).get_plain_text().abs().max().item())
        var = MPCMamba_Function.clamp(var, min_val=1.0,max_val=4096.0)
        u = var / 1024.0

        return x * MPCMamba_Function.rsqrt(var + self.eps) * self.weight

class MPCMamba_Function():
    @staticmethod
    def exp(x : MPCTensor,debug: int) -> MPCTensor:
        #max_val = x.get_plain_text().max().item()
        #min_val = x.get_plain_text().min().item()
        #print(f"exp dubug ID:{debug}")
        #print(f"exp input max: {max_val}")   
        #print(f"exp input min: {min_val}")
        x = MPCMamba_Function.clamp(x,min_val=-500.0,max_val = 10.0)   
        return x.exp()

    @staticmethod
    def reciprocal(x : MPCTensor) -> MPCTensor:
        return x.reciprocal()

    @staticmethod
    def log(x : MPCTensor) -> MPCTensor:
        return x.log()

    @staticmethod
    def rsqrt(x : MPCTensor) -> MPCTensor:
        max_val = x.get_plain_text().max().item()
        print(f"rsqrt input max: {max_val}")   
        x = MPCMamba_Function.clamp(x,max_val = 165.0)
        return x.inv_sqrt()
    
    @staticmethod
    def silu(x : MPCTensor) -> MPCTensor:
        #x = MPCMamba_Function.clamp(x,max_val = 500.0)
        return x * x.sigmoid()
    
    @staticmethod
    def softplus(x : MPCTensor) -> MPCTensor:
        x = MPCMamba_Function.clamp(x, max_val=5.5)
        result =  (x.exp() + 1.0).log()
        return result
    
    @staticmethod
    def clamp(x: crypten.mpc.MPCTensor, max_val=None, min_val=None):
        res = x
        if max_val is not None:
            res = max_val - (max_val - res).relu()
        if min_val is not None:
            res = min_val + (res - min_val).relu()          
        return res
    
    @staticmethod
    def pad_left(x, pad_size):
        """
        F.pad(x, (pad_size, 0)) の代替 (左側/過去方向へのゼロパディング)

        Arguments:
            x: (batch, dim, seqlen) などの暗号化テンソル
        """
        if pad_size <= 0:
            return x
        shape = list(x.shape)
        shape[-1] = pad_size
        zeros = crypten.cryptensor(torch.zeros(*shape, device=x.device)).to(x.device)
        return crypten.cat([zeros, x], dim=-1)
    
    @staticmethod
    def mpc_repeat_inter_chunk(x, T):
        """
        einops.repeat(x, "... d -> ... d e", e=T) の代替 (segsum用)

        Arguments:
            x: (..., T) 形状のテンソルを (..., T, T) に拡張
        """
        # 末尾に次元を追加して (..., T, 1) にする
        x_expanded = x.unsqueeze(-1)
        
        # Tのサイズを持つ「1」で満たされた平文テンソルを掛けてブロードキャストさせる
        ones = torch.ones([1] * (len(x.shape)) + [T], device=x.device)
        ones = crypten.cryptensor(ones).to(x.device)
        return x_expanded * ones
    
    @staticmethod
    def cumsum(x, dim):
        def mpc_select(tensor, d, idx):
            slices = [slice(None)] * len(tensor.shape)
            slices[d] = idx
            return tensor[tuple(slices)]

        res = [mpc_select(x, dim, 0)]
        for i in range(1, x.shape[dim]):
            res.append(res[-1] + mpc_select(x, dim, i))
            
        return crypten.stack(res, dim=dim)
