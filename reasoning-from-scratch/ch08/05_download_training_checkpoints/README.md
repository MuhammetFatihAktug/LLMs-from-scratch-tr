# Eğitim Kontrol Noktalarını (Checkpoint) İndirmek ve Kullanmak

> 🇹🇷 **Türkçe çeviri.** Orijinal İngilizce sürüm: [README.md](https://github.com/rasbt/reasoning-from-scratch/blob/main/ch08/05_download_training_checkpoints/README.md) · Kod blokları birebir korunmuştur.

Bu klasör, 8. bölümün damıtma kontrol noktalarının Hugging Face model merkezinden ([https://huggingface.co/rasbt/qwen3-from-scratch-distill-checkpoints](https://huggingface.co/rasbt/qwen3-from-scratch-distill-checkpoints)) nasıl indirileceğini ve kullanılacağını açıklar.

Kontrol noktaları, `reasoning_from_scratch` paketi için düz PyTorch `state_dict` dosyalarıdır. Hugging Face Transformers kontrol noktaları değildir.

---

**Not**: `uv` kullanıcısı değilseniz, aşağıdaki örneklerde `uv run ...py` yerine `python ...py` yazın.

---

&nbsp;
## Mevcut Kontrol Noktası Klasörleri

- `ch08_distill_deepseek_r1`: [`ch08_main.ipynb`](https://github.com/rasbt/reasoning-from-scratch/blob/main/ch08/01_main-chapter-code/ch08_main.ipynb) içindeki 3-5. satırlarda kullanılan 3 DeepSeek-R1 damıtma kontrol noktası
- `ch08_distill_qwen3_235b_a22b`: [`ch08_main.ipynb`](https://github.com/rasbt/reasoning-from-scratch/blob/main/ch08/01_main-chapter-code/ch08_main.ipynb) içindeki 6-8. satırlarda kullanılan 3 Qwen3 235B A22B damıtma kontrol noktası

Kontrol noktaları şurada barındırılmaktadır:

- [rasbt/qwen3-from-scratch-distill-checkpoints](https://huggingface.co/rasbt/qwen3-from-scratch-distill-checkpoints)

&nbsp;
## Bir Kontrol Noktası İndirmek

[`reasoning_from_scratch.qwen3`](https://github.com/rasbt/reasoning-from-scratch/blob/main/reasoning_from_scratch/qwen3.py) içindeki `download_qwen3_distill_checkpoints(...)` fonksiyonunu kullanın:

```python
from reasoning_from_scratch.qwen3 import download_qwen3_distill_checkpoints

checkpoint_path = download_qwen3_distill_checkpoints(
    distill_type="deepseek_r1",
    step="06682",
    out_dir="qwen3",
)
```

&nbsp;
## Hangi Tokenizer Kullanılmalı

Şunlar için akıl yürütme (reasoning) tokenizer'ını kullanın:

- `ch08_distill_deepseek_r1`
- `ch08_distill_qwen3_235b_a22b`

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
    download_qwen3_distill_checkpoints,
    download_qwen3_small,
    Qwen3Model,
    Qwen3Tokenizer,
    QWEN_CONFIG_06_B,
)

device = get_device()
local_dir = Path("qwen3")

checkpoint_path = download_qwen3_distill_checkpoints(
    distill_type="deepseek_r1",
    step="06682",
    out_dir=local_dir,
)
download_qwen3_small(kind="reasoning", tokenizer_only=True, out_dir=local_dir)

tokenizer = Qwen3Tokenizer(
    tokenizer_file_path=local_dir / "tokenizer-reasoning.json",
    apply_chat_template=True,
    add_generation_prompt=True,
    add_thinking=True,
)
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
## Qwen3 Örneği

`ch08_distill_qwen3_235b_a22b` için, aynı yardımcı fonksiyonu diğer `distill_type` değeriyle kullanın:

```python
from reasoning_from_scratch.qwen3 import download_qwen3_distill_checkpoints

download_qwen3_distill_checkpoints(
    distill_type="qwen3_235b_a22b",
    step="05746",
    out_dir="qwen3",
)
```

&nbsp;
## Mevcut Adımlar

`deepseek_r1` için kaydedilmiş mevcut adımlar:

- `06682`
- `13364`
- `20046`

`qwen3_235b_a22b` için kaydedilmiş mevcut adımlar:

- `05746`
- `11492`
- `17238`
