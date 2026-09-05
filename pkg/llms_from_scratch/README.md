# `llms-from-scratch` PyPI Paketi

> 🇹🇷 **Türkçe çeviri.** Orijinal İngilizce sürüm: [README.md](https://github.com/rasbt/LLMs-from-scratch/blob/main/pkg/llms_from_scratch/README.md) · Kod ve komut blokları birebir korunmuştur.

Bu isteğe bağlı PyPI paketi, *Build a Large Language Model From Scratch* kitabının çeşitli bölümlerindeki kodu rahatça içe aktarmanızı sağlar.

&nbsp;
## Kurulum

&nbsp;
### PyPI üzerinden

`llms-from-scratch` paketini resmî [Python Package Index](https://pypi.org/project/llms-from-scratch/) (PyPI) üzerinden kurun:

```bash
pip install llms-from-scratch
```

> **Not:** [`uv`](https://github.com/astral-sh/uv) kullanıyorsanız `pip` yerine `uv pip` kullanın veya `uv add` komutunu tercih edin:

```bash
uv add llms-from-scratch
```



&nbsp;
### GitHub'dan düzenlenebilir (editable) kurulum

Kodu değiştirmek ve bu değişikliklerin geliştirme sırasında yansımasını istiyorsanız:

```bash
git clone https://github.com/rasbt/LLMs-from-scratch.git
cd LLMs-from-scratch
pip install -e .
```

> **Not:** `uv` ile şunu kullanın:

```bash
uv add --editable . --dev
```



&nbsp;
## Paketi Kullanmak

Kurulumdan sonra, herhangi bir bölümden kodu şu şekilde içe aktarabilirsiniz:

```python
from llms_from_scratch.ch02 import GPTDatasetV1, create_dataloader_v1

from llms_from_scratch.ch03 import (
    SelfAttention_v1,
    SelfAttention_v2,
    CausalAttention,
    MultiHeadAttentionWrapper,
    MultiHeadAttention,
    PyTorchMultiHeadAttention # Bonus: Faster variant using PyTorch's scaled_dot_product_attention
)

from llms_from_scratch.ch04 import (
    LayerNorm,
    GELU,
    FeedForward,
    TransformerBlock,
    GPTModel,
    GPTModelFast # Bonus: Faster variant using PyTorch's scaled_dot_product_attention
    generate_text_simple
)

from llms_from_scratch.ch05 import (
    generate,
    train_model_simple,
    evaluate_model,
    generate_and_print_sample,
    assign,
    load_weights_into_gpt,
    text_to_token_ids,
    token_ids_to_text,
    calc_loss_batch,
    calc_loss_loader,
    plot_losses,
    download_and_load_gpt2
)

from llms_from_scratch.ch06 import (
    download_and_unzip_spam_data,
    create_balanced_dataset,
    random_split,
    SpamDataset,
    calc_accuracy_loader,
    evaluate_model,
    train_classifier_simple,
    plot_values,
    classify_review
)

from llms_from_scratch.ch07 import (
    download_and_load_file,
    format_input,
    InstructionDataset,
    custom_collate_fn,
    check_if_running,
    query_model,
    generate_model_scores
)

	
from llms_from_scratch.appendix_a import NeuralNetwork, ToyDataset

from llms_from_scratch.appendix_d import find_highest_gradient, train_model
```



&nbsp;

### GPT-2 KV önbelleği varyantı (Bonus materyal)

```python
from llms_from_scratch.kv_cache.gpt2 import GPTModel
from llms_from_scratch.kv_cache.generate import generate_text_simple
```

KV önbellekleme hakkında daha fazla bilgi için lütfen [KV önbelleği README dosyasına](../../ch04/03_kv-cache) bakın.



&nbsp;

### Llama 3 (Bonus materyal)

```python
from llms_from_scratch.llama3 import (
		load_weights_into_llama,
  	Llama3Model,
    Llama3ModelFast,
    Llama3Tokenizer,
    ChatFormat,
    clean_text
)

# KV cache drop-in replacements
from llms_from_scratch.kv_cache.llama3 import Llama3Model
from llms_from_scratch.kv_cache.generate import generate_text_simple
```

`llms_from_scratch.llama3` kullanım bilgileri için lütfen [bu bonus bölüme](../../ch05/07_gpt_to_llama/README.md) bakın.

KV önbellekleme hakkında daha fazla bilgi için lütfen [KV önbelleği README dosyasına](../../ch04/03_kv-cache) bakın.


&nbsp;
### Qwen3 (Bonus materyal)

```python
from llms_from_scratch.qwen3 import (
    load_weights_into_qwen,
    Qwen3Model,
    Qwen3Tokenizer,
)

# KV cache drop-in replacements
from llms_from_scratch.kv_cache.qwen3 import Qwen3Model
from llms_from_scratch.kv_cache.generate import (
    generate_text_simple,
    generate_text_simple_stream
)

# KV cache drop-in replacements with batched inference support
from llms_from_scratch.kv_cache_batched.generate import (
    generate_text_simple,
    generate_text_simple_stream
)
from llms_from_scratch.kv_cache_batched.qwen3 import Qwen3Model
```

`llms_from_scratch.qwen3` kullanım bilgileri için lütfen [bu bonus bölüme](../../ch05/11_qwen3/README.md) bakın.

KV önbellekleme hakkında daha fazla bilgi için lütfen [KV önbelleği README dosyasına](../../ch04/03_kv-cache) bakın.
