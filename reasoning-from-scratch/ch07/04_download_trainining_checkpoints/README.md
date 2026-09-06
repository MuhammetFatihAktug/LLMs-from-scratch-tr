# Eğitim Kontrol Noktalarını (Checkpoint) İndirmek ve Kullanmak

> 🇹🇷 **Türkçe çeviri.** Orijinal İngilizce sürüm: [README.md](https://github.com/rasbt/reasoning-from-scratch/blob/main/ch07/04_download_trainining_checkpoints/README.md) · Kod blokları birebir korunmuştur.

Bu klasör, 7. bölümün eğitim kontrol noktalarının Hugging Face üzerinden ([https://huggingface.co/rasbt/qwen3-from-scratch-grpo-checkpoints](https://huggingface.co/rasbt/qwen3-from-scratch-grpo-checkpoints)) nasıl indirileceğini ve kullanılacağını açıklar.

Kontrol noktaları, `reasoning_from_scratch` paketi için düz PyTorch `state_dict` dosyalarıdır. Hugging Face Transformers kontrol noktaları değildir.

---

**Not**: `uv` kullanıcısı değilseniz, aşağıdaki örneklerde `uv run ...py` yerine `python ...py` yazın.

---

&nbsp;
## Mevcut Kontrol Noktası Klasörleri

- `7_3_plus_tracking`: ek ölçüt izlemeli GRPO kontrol noktaları
- `7_4_plus_clip_ratio`: kırpılmış politika oranlı GRPO kontrol noktaları
- `7_5_plus_kl`: KL terimli GRPO kontrol noktaları
- `7_6_plus_format_reward`: `<think>` etiketleri için açık biçim ödüllü GRPO kontrol noktaları

Kontrol noktaları şurada barındırılmaktadır:

- [rasbt/qwen3-from-scratch-grpo-checkpoints](https://huggingface.co/rasbt/qwen3-from-scratch-grpo-checkpoints)

&nbsp;
## Bir Kontrol Noktası İndirmek

[`reasoning_from_scratch.qwen3`](https://github.com/rasbt/reasoning-from-scratch/blob/main/reasoning_from_scratch/qwen3.py) içindeki `download_qwen3_grpo_checkpoints(...)` fonksiyonunu kullanın:

```python
from reasoning_from_scratch.qwen3 import download_qwen3_grpo_checkpoints

checkpoint_path = download_qwen3_grpo_checkpoints(
    grpo_type="clip_ratio",
    step="00050",
    out_dir="qwen3",
)
```

&nbsp;
## Hangi Tokenizer Kullanılmalı

Şunlar için temel (base) tokenizer'ı kullanın:

- `7_3_plus_tracking`
- `7_4_plus_clip_ratio`
- `7_5_plus_kl`

Şunun için akıl yürütme (reasoning) tokenizer'ını kullanın:

- `7_6_plus_format_reward`

Bunun nedeni, `7_6_plus_format_reward` kontrol noktasının akıl yürütme modelinden eğitilmiş olması ve akıl yürütme sohbet biçimlendirmesini beklemesidir.

&nbsp;
## Kullanım Örneği

Aşağıdaki örnek bir kontrol noktasını indirir, eşleşen tokenizer'ı indirir, modeli yükler ve 2. bölümdeki `generate_text_basic_stream_cache` ile metin üretir:

```python
from pathlib import Path
import torch

from reasoning_from_scratch.ch02 import (
    get_device,
    generate_text_basic_stream_cache,
)
from reasoning_from_scratch.ch03 import render_prompt
from reasoning_from_scratch.qwen3 import (
    download_qwen3_grpo_checkpoints,
    download_qwen3_small,
    Qwen3Model,
    Qwen3Tokenizer,
    QWEN_CONFIG_06_B,
)

device = get_device()
local_dir = Path("qwen3")

checkpoint_path = download_qwen3_grpo_checkpoints(
    grpo_type="clip_ratio",
    step="00050",
    out_dir=local_dir,
)
download_qwen3_small(kind="base", tokenizer_only=True, out_dir=local_dir)

tokenizer = Qwen3Tokenizer(tokenizer_file_path=local_dir / "tokenizer-base.json")
model = Qwen3Model(QWEN_CONFIG_06_B)
state_dict = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

prompt = render_prompt("Solve: If x + 7 = 19, what is x?")
input_ids = torch.tensor(tokenizer.encode(prompt), device=device).unsqueeze(0)

for token in generate_text_basic_stream_cache(
    model=model,
    token_ids=input_ids,
    max_new_tokens=256,
    eos_token_id=tokenizer.eos_token_id,
):
    token_id = token.squeeze(0).item()
    print(tokenizer.decode([token_id]), end="", flush=True)
```

&nbsp;
## Biçim Ödülü (Format-Reward) Örneği

`7_6_plus_format_reward` için akıl yürütme tokenizer'ına geçin:

```python
from pathlib import Path

from reasoning_from_scratch.qwen3 import (
    download_qwen3_small,
    Qwen3Tokenizer,
)

local_dir = Path("qwen3")
download_qwen3_small(kind="reasoning", tokenizer_only=True, out_dir=local_dir)

tokenizer = Qwen3Tokenizer(
    tokenizer_file_path=local_dir / "tokenizer-reasoning.json",
    apply_chat_template=True,
    add_generation_prompt=True,
    add_thinking=True,
)
```

&nbsp;
## Bölüm 6 Örneği

Aynı yardımcı fonksiyon, orijinal 6. bölümdeki KL'siz (no-KL) kontrol noktasını da destekler:

```python
from reasoning_from_scratch.qwen3 import download_qwen3_grpo_checkpoints

download_qwen3_grpo_checkpoints(grpo_type="no_kl", step="00050", out_dir="qwen3")
```

&nbsp;
## Mevcut Kontrol Noktaları

Bölüm eşlemesi:

- `no_kl`: orijinal KL'siz GRPO kurulumundan gelen 6. bölüm temel çizgisi
- `tracking`: ana bölümdeki 7.3 kısmı
- `clip_ratio`: ana bölümdeki 7.4 kısmı
- `kl`: ana bölümdeki 7.5 kısmı
- `format_reward`: ana bölümdeki 7.6 kısmı

Kaydedilmiş mevcut adımlar:

- `no_kl`: `00050`, `00100`, `00500`, `01000`, `01500`, `03000`, `05000`, `09000`
- `tracking`: `00050`, `00100`, `00150`, `00200`, `00250`, `00300`, `00350`, `00400`, `00450`, `00500`
- `clip_ratio`: `00050`, `00100`, `00150`, `00200`, `00250`, `00300`, `00350`, `00400`, `00450`, `00500`
- `kl`: `00050`, `00100`, `00150`, `00200`, `00250`, `00300`, `00350`, `00400`, `00450`, `00500`
- `format_reward`: `00050`, `00100`, `00150`, `00200`, `00250`, `00300`, `00350`, `00400`, `00450`, `00500`
