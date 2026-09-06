# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt)
# Source for "Build a Reasoning Model (From Scratch)": https://mng.bz/lZ5B
# Code repository: https://github.com/rasbt/reasoning-from-scratch

from .utils import download_file

from pathlib import Path
import re

import torch
import torch.nn as nn


# 0,6 milyar parametre
QWEN_CONFIG_06_B = {
    "vocab_size": 151_936,     # Sözcük dağarcığı boyutu
    "context_length": 40_960,  # Eğitim sırasında özgün olarak kullanılan uzunluk
    "emb_dim": 1024,           # Gömme (embedding) boyutu
    "n_heads": 16,             # Dikkat başlığı sayısı
    "n_layers": 28,            # Katman sayısı
    "hidden_dim": 3072,        # FeedForward içindeki ara boyutun büyüklüğü
    "head_dim": 128,           # GQA içindeki başlıkların boyutu
    "qk_norm": True,           # GQA'da sorgu ve anahtarların normalleştirilip normalleştirilmeyeceği
    "n_kv_groups": 8,          # GQA için anahtar-değer grupları
    "rope_base": 1_000_000.0,  # RoPE'nin "theta" değerindeki taban
    "dtype": torch.bfloat16,   # Belleği azaltmak için daha düşük duyarlıklı dtype
}


class Qwen3Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # Ana model parametreleri
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])

        self.trf_blocks = nn.ModuleList(  # Sequential yalnızca tek girdi alabildiği ve bize `x, mask, cos, sin` gerektiği için ModuleList
            [TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = RMSNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])

        # Yeniden kullanılabilir yardımcılar
        if cfg["head_dim"] is None:
            head_dim = cfg["emb_dim"] // cfg["n_heads"]
        else:
            head_dim = cfg["head_dim"]
        cos, sin = compute_rope_params(
            head_dim=head_dim,
            theta_base=cfg["rope_base"],
            context_length=cfg["context_length"]
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.cfg = cfg
        self.current_pos = 0  # KV önbelleğindeki geçerli konumu izle

    def forward(self, in_idx, cache=None):
        # İleri geçiş
        tok_embeds = self.tok_emb(in_idx)
        x = tok_embeds

        num_tokens = x.shape[1]
        if cache is not None:
            pos_start = self.current_pos
            pos_end = pos_start + num_tokens
            self.current_pos = pos_end
            mask = torch.triu(
                torch.ones(pos_end, pos_end, device=x.device, dtype=torch.bool), diagonal=1
            )[pos_start:pos_end, :pos_end]
        else:
            pos_start = 0  # Kesinlikle gerekli değil ama torch.compile'a yardımcı olur
            mask = torch.triu(
                torch.ones(num_tokens, num_tokens, device=x.device, dtype=torch.bool), diagonal=1
            )
        # Ön doldurma (önbelleksiz): maske (num_tokens, num_tokens) olarak başlar
        # Önbellekli kod çözme: maske (num_tokens, prev_k_number_tokens + num_tokens) olarak başlar
        #
        # Başa iki boyut ekliyoruz; böylece maske ön doldurma sırasında
        # (1, 1, num_tokens, num_tokens), önbellekli kod çözme sırasında ise
        # (1, 1, num_tokens, total_key_tokens) hâline gelir.
        # Bu ek boyutlar, maske (batch, num_heads, num_tokens, total_key_tokens)
        # şeklindeki attn_scores üzerine uygulanırken PyTorch'un aynı maskeyi tüm
        # yığınlar ve dikkat başlıkları boyunca yayınlamasını sağlar.
        mask = mask[None, None, :, :]  # maskeyi yayınla

        for i, block in enumerate(self.trf_blocks):
            blk_cache = cache.get(i) if cache else None
            x, new_blk_cache = block(x, mask, self.cos, self.sin,
                                     start_pos=pos_start,
                                     cache=blk_cache)
            if cache is not None:
                cache.update(i, new_blk_cache)

        x = self.final_norm(x)
        logits = self.out_head(x.to(self.cfg["dtype"]))
        return logits

    def reset_kv_cache(self):
        self.current_pos = 0


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = GroupedQueryAttention(
            d_in=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            head_dim=cfg["head_dim"],
            num_kv_groups=cfg["n_kv_groups"],
            qk_norm=cfg["qk_norm"],
            dtype=cfg["dtype"]
        )
        self.ff = FeedForward(cfg)
        self.norm1 = RMSNorm(cfg["emb_dim"], eps=1e-6)
        self.norm2 = RMSNorm(cfg["emb_dim"], eps=1e-6)

    def forward(self, x, mask, cos, sin, start_pos=0, cache=None):
        # Dikkat bloğu için kestirme (shortcut) bağlantı
        shortcut = x
        x = self.norm1(x)
        x, next_cache = self.att(x, mask, cos, sin, start_pos=start_pos, cache=cache)  # Şekil [batch_size, num_tokens, emb_size]
        x = x + shortcut  # Özgün girdiyi geri ekle

        # İleri beslemeli blok için kestirme (shortcut) bağlantı
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = x + shortcut  # Özgün girdiyi geri ekle

        return x, next_cache


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc2 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc3 = nn.Linear(cfg["hidden_dim"], cfg["emb_dim"], dtype=cfg["dtype"], bias=False)

    def forward(self, x):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = nn.functional.silu(x_fc1) * x_fc2
        return self.fc3(x)


class GroupedQueryAttention(nn.Module):
    def __init__(
        self, d_in, num_heads, num_kv_groups, head_dim=None, qk_norm=False, dtype=None
    ):
        super().__init__()
        assert num_heads % num_kv_groups == 0, "num_heads must be divisible by num_kv_groups"

        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups

        if head_dim is None:
            assert d_in % num_heads == 0, "`d_in` must be divisible by `num_heads` if `head_dim` is not set"
            head_dim = d_in // num_heads

        self.head_dim = head_dim
        self.d_out = num_heads * head_dim

        self.W_query = nn.Linear(d_in, self.d_out, bias=False, dtype=dtype)
        self.W_key = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)
        self.W_value = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)

        self.out_proj = nn.Linear(self.d_out, d_in, bias=False, dtype=dtype)

        if qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=1e-6)
            self.k_norm = RMSNorm(head_dim, eps=1e-6)
        else:
            self.q_norm = self.k_norm = None

    def forward(self, x, mask, cos, sin, start_pos=0, cache=None):
        b, num_tokens, _ = x.shape

        # İzdüşümleri uygula
        queries = self.W_query(x)  # (b, num_tokens, num_heads * head_dim)
        keys = self.W_key(x)       # (b, num_tokens, num_kv_groups * head_dim)
        values = self.W_value(x)   # (b, num_tokens, num_kv_groups * head_dim)

        # Başlıklara / kv gruplarına yeniden şekillendir
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        keys_new = keys.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
        values_new = values.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)

        # İsteğe bağlı normalleştirme
        if self.q_norm:
            queries = self.q_norm(queries)
        if self.k_norm:
            keys_new = self.k_norm(keys_new)

        # RoPE uygula
        queries = apply_rope(queries, cos, sin, offset=start_pos)
        keys_new = apply_rope(keys_new, cos, sin, offset=start_pos)

        if cache is not None:
            prev_k, prev_v = cache
            keys = torch.cat([prev_k, keys_new], dim=2)
            values = torch.cat([prev_v, values_new], dim=2)
        else:
            start_pos = 0  # RoPE'yi sıfırla
            keys, values = keys_new, values_new
        next_cache = (keys, values)

        # K ve V tensörlerini baş sayısıyla eşleşecek şekilde genişlet
        keys = keys.repeat_interleave(self.group_size, dim=1)
        values = values.repeat_interleave(self.group_size, dim=1)

        # Dikkat
        attn_scores = queries @ keys.transpose(2, 3)
        attn_scores = attn_scores.masked_fill(mask, -torch.inf)
        attn_weights = torch.softmax(attn_scores / self.head_dim**0.5, dim=-1)

        context = (attn_weights @ values).transpose(1, 2).reshape(b, num_tokens, self.d_out)
        return self.out_proj(context), next_cache


# ==============================================================================
# RoPE uygulamasının özeti
#
#
# RoPE'yi uygulamanın, matematiksel olarak eşdeğer olan
# iki yaygın biçimi vardır;
# temel fark, döndürme matrisinin boyutları nasıl eşleştirdiğidir.
#
# 1) Yarıya bölme biçimi (bu depo, Hugging Face Transformers):
#
#   Gizli boyut d = 4 için (örnek):
#
#       [ x0   x1 | x2   x3 ]
#         │    │    │    │
#         ▼    ▼    ▼    ▼
#        cos  cos  sin  sin
#
#   Döndürme matrisi:
#
#       [ cosθ0   0    -sinθ0   0   ]
#       [  0    cosθ1    0    -sinθ1]
#       [ sinθ0   0     cosθ0   0   ]
#       [  0    sinθ1    0     cosθ1]
#
#   Burada gömme boyutları iki yarıya ayrılır ve ardından
#   her biri bloklar hâlinde döndürülür.
#
#
# 2) Çapraz geçmeli (tek/çift) biçim (özgün makale, Llama deposu):
#
#   Gizli boyut d = 4 için (örnek):
#
#       [ x0   x1   x2   x3 ]
#         │    │    │    │
#         ▼    ▼    ▼    ▼
#        cos  sin  cos  sin
#
#   Döndürme matrisi:
#
#       [ cosθ0  -sinθ0   0       0    ]
#       [ sinθ0   cosθ0   0       0    ]
#       [  0        0    cosθ1  -sinθ1 ]
#       [  0        0    sinθ1   cosθ1 ]
#
#
#   Burada gömme boyutları tek/çift kosinüs/sinüs çiftleri olarak çapraz geçirilir.
#
# Her iki yerleşim de aynı göreli konumları kodlar; tek fark boyutların
# nasıl eşleştirildiğidir.
# ==============================================================================


def compute_rope_params(head_dim, theta_base=10_000, context_length=4096, dtype=torch.float32):
    assert head_dim % 2 == 0, "Embedding dimension must be even"

    # Ters frekansları hesapla
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2, dtype=dtype)[: (head_dim // 2)].float() / head_dim))

    # Konum indekslerini üret
    positions = torch.arange(context_length, dtype=dtype)

    # Açıları hesapla
    angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0)  # Shape: (context_length, head_dim // 2)

    # Açıları head_dim ile eşleşecek şekilde genişlet
    angles = torch.cat([angles, angles], dim=1)  # Shape: (context_length, head_dim)

    # Sinüs ve kosinüsü önceden hesapla
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin


def apply_rope(x, cos, sin, offset=0):
    # x: (batch_size, num_heads, seq_len, head_dim)
    batch_size, num_heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "Head dimension must be even"

    # x tensörünü birinci ve ikinci yarıya böl
    x1 = x[..., : head_dim // 2]  # İlk yarı
    x2 = x[..., head_dim // 2:]  # İkinci yarı

    # sin ve cos şekillerini ayarla
    cos = cos[offset:offset + seq_len, :].unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, seq_len, head_dim)
    sin = sin[offset:offset + seq_len, :].unsqueeze(0).unsqueeze(0)

    # Döner (rotary) dönüşümü uygula
    rotated = torch.cat((-x2, x1), dim=-1)
    x_rotated = (x * cos) + (rotated * sin)

    # cos ve sin döndürmesi uygulandıktan sonra daha düşük hassasiyet kullanmak sorun değil
    return x_rotated.to(dtype=x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-6, bias=False, qwen3_compatible=True):
        super().__init__()
        self.eps = eps
        self.qwen3_compatible = qwen3_compatible
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim)) if bias else None

    def forward(self, x):
        input_dtype = x.dtype

        if self.qwen3_compatible:
            x = x.to(torch.float32)

        variance = x.pow(2).mean(dim=-1, keepdim=True)
        norm_x = x * torch.rsqrt(variance + self.eps)
        norm_x = norm_x * self.scale

        if self.shift is not None:
            norm_x = norm_x + self.shift

        return norm_x.to(input_dtype)


class Qwen3Tokenizer:
    _SPECIALS = [
        "<|endoftext|>",
        "<|im_start|>", "<|im_end|>",
        "<|object_ref_start|>", "<|object_ref_end|>",
        "<|box_start|>", "<|box_end|>",
        "<|quad_start|>", "<|quad_end|>",
        "<|vision_start|>", "<|vision_end|>",
        "<|vision_pad|>", "<|image_pad|>", "<|video_pad|>",
    ]
    _SPLIT_RE = re.compile(r"(<\|[^>]+?\|>)")

    def __init__(self, tokenizer_file_path="tokenizer-base.json",
                 apply_chat_template=False,
                 add_generation_prompt=False,
                 add_thinking=False):
        from tokenizers import Tokenizer

        self.apply_chat_template = apply_chat_template
        self.add_generation_prompt = add_generation_prompt
        self.add_thinking = add_thinking

        tok_path = Path(tokenizer_file_path)
        if not tok_path.is_file():
            raise FileNotFoundError(
                f"Tokenizer file '{tok_path}' not found. Please ensure it's available."
            )

        self._tok = Tokenizer.from_file(str(tok_path))
        self._special_to_id = {t: self._tok.token_to_id(t) for t in self._SPECIALS}

        self.pad_token = "<|endoftext|>"
        self.pad_token_id = self._special_to_id.get(self.pad_token)

        # HF davranışına uy: sohbet modeli → <|im_end|>, temel model → <|endoftext|>
        fname = tok_path.name.lower()
        if "base" in fname and "reasoning" not in fname:
            self.eos_token = "<|endoftext|>"
        else:
            self.eos_token = "<|im_end|>"
        self.eos_token_id = self._special_to_id.get(self.eos_token)

    def encode(self, prompt, chat_wrapped=None):
        if chat_wrapped is None:
            chat_wrapped = self.apply_chat_template

        stripped = prompt.strip()
        if stripped in self._special_to_id and "\n" not in stripped:
            return [self._special_to_id[stripped]]

        if chat_wrapped:
            prompt = self._wrap_chat(prompt)

        ids = []
        for part in filter(None, self._SPLIT_RE.split(prompt)):
            if part in self._special_to_id:
                ids.append(self._special_to_id[part])
            else:
                ids.extend(self._tok.encode(part).ids)
        return ids

    def decode(self, token_ids):
        return self._tok.decode(token_ids, skip_special_tokens=False)

    def _wrap_chat(self, user_msg):
        s = f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        if self.add_generation_prompt:
            s += "<|im_start|>assistant"
            if self.add_thinking:
                s += "\n"  # <think> etiketi ekleme, yalnızca yeni satır
            else:
                s += "\n<think>\n\n</think>\n\n"
        return s


class KVCache:
    def __init__(self, n_layers):
        self.cache = [None] * n_layers

    def get(self, layer_idx):
        return self.cache[layer_idx]

    def update(self, layer_idx, value):
        self.cache[layer_idx] = value

    def get_all(self):
        return self.cache

    def reset(self):
        for i in range(len(self.cache)):
            self.cache[i] = None


def download_qwen3_small(kind="base", tokenizer_only=False, out_dir="."):
    files = {
        "base": {"model": "qwen3-0.6B-base.pth", "tokenizer": "tokenizer-base.json"},
        "reasoning": {"model": "qwen3-0.6B-reasoning.pth", "tokenizer": "tokenizer-reasoning.json"},
    }
    if kind not in files:
        raise ValueError("kind must be 'base' or 'reasoning'")

    repo = "rasbt/qwen3-from-scratch"
    hf_fmt = "https://huggingface.co/{repo}/resolve/main/{file}"
    backup_root = "https://f001.backblazeb2.com/file/reasoning-from-scratch/qwen3-0.6B"
    targets = ["tokenizer"] if tokenizer_only else ["model", "tokenizer"]

    for key in targets:
        fname = files[kind][key]
        primary = hf_fmt.format(repo=repo, file=fname)
        backup = f"{backup_root}/{fname}"
        download_file(primary, out_dir=out_dir, backup_url=backup)


def download_qwen3_grpo_checkpoints(
    grpo_type="no_kl",
    step="00050",
    out_dir=".",
):
    mapper = {
        "no_kl": "grpo_original_no_kl",
        "tracking": "7_3_plus_tracking/checkpoints",
        "clip_ratio": "7_4_plus_clip_ratio/checkpoints",
        "kl": "7_5_plus_kl/checkpoints",
        "format_reward": "7_6_plus_format_reward/checkpoints",
    }
    if grpo_type not in mapper:
        raise ValueError(f"only grpo_type in {mapper.keys()} are supported for now")

    repo = "rasbt/qwen3-from-scratch-grpo-checkpoints"
    step = str(step)
    if step.isdigit():
        step = step.zfill(5)
    fname = f"qwen3-0.6B-rlvr-grpo-step{step}.pth"
    primary = f"https://huggingface.co/{repo}/resolve/main/{mapper[grpo_type]}/{fname}"

    backup = None
    if grpo_type == "no_kl" and step == "00050":
        backup_root = (
            "https://f001.backblazeb2.com/file/"
            "reasoning-from-scratch/qwen3-0.6B-checkpoints"
        )
        fname = (
            "grpo_original_no_kl/qwen3-0.6B-rlvr-grpo-step00050.pth"
        )
        backup = f"{backup_root}/{fname}"

    return download_file(primary, out_dir=out_dir, backup_url=backup)


def download_qwen3_distill_checkpoints(
    distill_type="deepseek_r1",
    step="06682",
    out_dir=".",
):
    mapper = {
        "deepseek_r1": {
            "06682": "qwen3-0.6B-distill-step06682-epoch1.pth",
            "13364": "qwen3-0.6B-distill-step13364-epoch2.pth",
            "20046": "qwen3-0.6B-distill-step20046-epoch3.pth",
        },
        "qwen3_235b_a22b": {
            "05746": "qwen3-0.6B-distill-step05746-epoch1.pth",
            "11492": "qwen3-0.6B-distill-step11492-epoch2.pth",
            "17238": "qwen3-0.6B-distill-step17238-epoch3.pth",
        },
    }
    folder_map = {
        "deepseek_r1": "ch08_distill_deepseek_r1/checkpoints",
        "qwen3_235b_a22b": "ch08_distill_qwen3_235b_a22b/checkpoints",
    }
    if distill_type not in mapper:
        raise ValueError(f"only distill_type in {mapper.keys()} are supported for now")

    step = str(step)
    if step.isdigit():
        step = step.zfill(5)
    if step not in mapper[distill_type]:
        raise ValueError(
            f"only step in {mapper[distill_type].keys()} are supported for {distill_type}"
        )

    repo = "rasbt/qwen3-from-scratch-distill-checkpoints"
    fname = mapper[distill_type][step]
    primary = f"https://huggingface.co/{repo}/resolve/main/{folder_map[distill_type]}/{fname}"
    return download_file(primary, out_dir=out_dir)


def load_hf_weights_into_qwen(model, param_config, params):
    """
    Only used in Appendix D for loading the other Qwen3 variants.
    """
    def assign(left, right, tensor_name="unknown"):
        if left.shape != right.shape:
            raise ValueError(f"Shape mismatch in tensor '{tensor_name}'. Left: {left.shape}, Right: {right.shape}")

        with torch.no_grad():
            if isinstance(right, torch.Tensor):
                left.copy_(right)
            else:
                left.copy_(torch.as_tensor(right, dtype=left.dtype, device=left.device))

        return left

    model.tok_emb.weight = assign(model.tok_emb.weight, params["model.embed_tokens.weight"], "model.embed_tokens.weight")

    for l in range(param_config["n_layers"]):  # noqa: E741
        block = model.trf_blocks[l]
        att = block.att

        # Q, K, V izdüşümleri
        att.W_query.weight = assign(
            att.W_query.weight,
            params[f"model.layers.{l}.self_attn.q_proj.weight"],
            f"model.layers.{l}.self_attn.q_proj.weight"
        )
        att.W_key.weight = assign(
            att.W_key.weight,
            params[f"model.layers.{l}.self_attn.k_proj.weight"],
            f"model.layers.{l}.self_attn.k_proj.weight"
        )
        att.W_value.weight = assign(
            att.W_value.weight,
            params[f"model.layers.{l}.self_attn.v_proj.weight"],
            f"model.layers.{l}.self_attn.v_proj.weight"
        )

        # Çıkış izdüşümü
        att.out_proj.weight = assign(
            att.out_proj.weight,
            params[f"model.layers.{l}.self_attn.o_proj.weight"],
            f"model.layers.{l}.self_attn.o_proj.weight"
        )

        # QK normları
        if hasattr(att, "q_norm") and att.q_norm is not None:
            att.q_norm.scale = assign(
                att.q_norm.scale,
                params[f"model.layers.{l}.self_attn.q_norm.weight"],
                f"model.layers.{l}.self_attn.q_norm.weight"
            )
        if hasattr(att, "k_norm") and att.k_norm is not None:
            att.k_norm.scale = assign(
                att.k_norm.scale,
                params[f"model.layers.{l}.self_attn.k_norm.weight"],
                f"model.layers.{l}.self_attn.k_norm.weight"
            )

        # Dikkat katman normalleştirmesi
        block.norm1.scale = assign(
            block.norm1.scale,
            params[f"model.layers.{l}.input_layernorm.weight"],
            f"model.layers.{l}.input_layernorm.weight"
        )

        # İleri beslemeli ağırlıklar
        if "num_experts" in param_config:
            # Yönlendirici (kapılama) ağırlıklarını yükle
            block.ff.gate.weight = assign(
                block.ff.gate.weight,
                params[f"model.layers.{l}.mlp.gate.weight"],
                f"model.layers.{l}.mlp.gate.weight"
            )
            # Uzman ağırlıklarını yükle
            for e in range(param_config["num_experts"]):
                prefix = f"model.layers.{l}.mlp.experts.{e}"
                block.ff.fc1[e].weight = assign(
                    block.ff.fc1[e].weight,
                    params[f"{prefix}.gate_proj.weight"],
                    f"{prefix}.gate_proj.weight"
                )
                block.ff.fc2[e].weight = assign(
                    block.ff.fc2[e].weight,
                    params[f"{prefix}.up_proj.weight"],
                    f"{prefix}.up_proj.weight"
                )
                block.ff.fc3[e].weight = assign(
                    block.ff.fc3[e].weight,
                    params[f"{prefix}.down_proj.weight"],
                    f"{prefix}.down_proj.weight"
                )
                # Ağırlıklar atandıktan sonra uzman katmanlarını meta'dan CPU'ya taşı
                block.ff.fc1[e] = block.ff.fc1[e].to("cpu")
                block.ff.fc2[e] = block.ff.fc2[e].to("cpu")
                block.ff.fc3[e] = block.ff.fc3[e].to("cpu")

        else:
            block.ff.fc1.weight = assign(
                block.ff.fc1.weight,
                params[f"model.layers.{l}.mlp.gate_proj.weight"],
                f"model.layers.{l}.mlp.gate_proj.weight"
            )
            block.ff.fc2.weight = assign(
                block.ff.fc2.weight,
                params[f"model.layers.{l}.mlp.up_proj.weight"],
                f"model.layers.{l}.mlp.up_proj.weight"
            )
            block.ff.fc3.weight = assign(
                block.ff.fc3.weight,
                params[f"model.layers.{l}.mlp.down_proj.weight"],
                f"model.layers.{l}.mlp.down_proj.weight"
            )

        block.norm2.scale = assign(
            block.norm2.scale,
            params[f"model.layers.{l}.post_attention_layernorm.weight"],
            f"model.layers.{l}.post_attention_layernorm.weight"
        )

    # Son normalleştirme ve çıkış başı
    model.final_norm.scale = assign(model.final_norm.scale, params["model.norm.weight"], "model.norm.weight")

    if "lm_head.weight" in params:
        model.out_head.weight = assign(model.out_head.weight, params["lm_head.weight"], "lm_head.weight")
    else:
        model.out_head.weight = model.tok_emb.weight
        print("Model uses weight tying.")
