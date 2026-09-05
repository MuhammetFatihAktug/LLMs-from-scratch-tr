# GPT'yi Llama'ya Dönüştürmek

> 🇹🇷 **Türkçe çeviri.** Orijinal İngilizce sürüm: [README.md](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch05/07_gpt_to_llama/README.md) · Kod ve çıktı blokları birebir korunmuştur.



Bu klasör, 4. ve 5. bölümlerdeki GPT uygulamasını Meta AI'ın Llama mimarisine dönüştüren kodu içerir; önerilen okuma sırası şöyledir:

- [converting-gpt-to-llama2.ipynb](converting-gpt-to-llama2.ipynb): GPT'yi adım adım Llama 2 7B'ye dönüştüren ve Meta AI'dan önceden eğitilmiş ağırlıkları yükleyen kodu içerir
- [converting-llama2-to-llama3.ipynb](converting-llama2-to-llama3.ipynb): Llama 2 modelini Llama 3, Llama 3.1 ve Llama 3.2'ye dönüştüren kodu içerir
- [standalone-llama32.ipynb](standalone-llama32.ipynb): Llama 3.2'yi uygulayan bağımsız bir not defteri

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/gpt-to-llama/gpt-and-all-llamas.webp">


&nbsp;
### Llama 3.2'yi `llms-from-scratch` paketiyle kullanmak

Llama 3.2 1B ve 3B modellerini kolayca kullanmak için, bu depodaki [pkg/llms_from_scratch](../../pkg/llms_from_scratch) kaynak koduna dayanan `llms-from-scratch` PyPI paketini de kullanabilirsiniz.

&nbsp;
#### 1) Kurulum

```bash
pip install llms_from_scratch blobfile
```

(Tokenizer'ı yüklemek için `blobfile` paketinin gerekli olduğunu unutmayın.)

&nbsp;
#### 2) Model ve metin üretimi ayarları

Hangi modelin kullanılacağını belirtin:

```python
MODEL_FILE = "llama3.2-1B-instruct.pth"
# MODEL_FILE = "llama3.2-1B-base.pth"
# MODEL_FILE = "llama3.2-3B-instruct.pth"
# MODEL_FILE = "llama3.2-3B-base.pth"
```

Kullanıcı tarafından tanımlanabilen temel metin üretimi ayarları. Önerilen 8192 token'lık bağlam boyutunun, metin üretimi örneği için yaklaşık 3 GB VRAM gerektirdiğini unutmayın.

```python
# Text generation settings
if "instruct" in MODEL_FILE:
    PROMPT = "What do llamas eat?"
else:
    PROMPT = "Llamas eat"

MAX_NEW_TOKENS = 150
TEMPERATURE = 0.
TOP_K = 1
```

&nbsp;
#### 3) Ağırlıkların indirilmesi ve yüklenmesi

Aşağıdaki kod, yukarıdaki model seçimine göre ağırlık dosyasını otomatik olarak indirir:

```python
import os
import requests

url = f"https://huggingface.co/rasbt/llama-3.2-from-scratch/resolve/main/{MODEL_FILE}"

if not os.path.exists(MODEL_FILE):
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    with open(MODEL_FILE, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print(f"Downloaded to {MODEL_FILE}")
```

Model ağırlıkları daha sonra şöyle yüklenir:

```python
import torch
from llms_from_scratch.llama3 import Llama3Model

if "1B" in MODEL_FILE:
    from llms_from_scratch.llama3 import LLAMA32_CONFIG_1B as LLAMA32_CONFIG
elif "3B" in MODEL_FILE:
    from llms_from_scratch.llama3 import LLAMA32_CONFIG_3B as LLAMA32_CONFIG
else:
    raise ValueError("Incorrect model file name")

model = Llama3Model(LLAMA32_CONFIG)
model.load_state_dict(torch.load(MODEL_FILE, weights_only=True, map_location="cpu"))

device = (
    torch.device("cuda") if torch.cuda.is_available() else
    torch.device("mps") if torch.backends.mps.is_available() else
    torch.device("cpu")
)
model.to(device)
```

&nbsp;
#### 4) Tokenizer'ı başlatmak

Aşağıdaki kod tokenizer'ı indirir ve başlatır:

```python
from llms_from_scratch.llama3 import Llama3Tokenizer, ChatFormat, clean_text

TOKENIZER_FILE = "tokenizer.model"

url = f"https://huggingface.co/rasbt/llama-3.2-from-scratch/resolve/main/{TOKENIZER_FILE}"

if not os.path.exists(TOKENIZER_FILE):
    urllib.request.urlretrieve(url, TOKENIZER_FILE)
    print(f"Downloaded to {TOKENIZER_FILE}")
    
tokenizer = Llama3Tokenizer("tokenizer.model")

if "instruct" in MODEL_FILE:
    tokenizer = ChatFormat(tokenizer)
```

&nbsp;
#### 5) Metin üretmek

Son olarak, aşağıdaki kodla metin üretebiliriz:

```python
import time

from llms_from_scratch.ch05 import (
    generate,
    text_to_token_ids,
    token_ids_to_text
)

torch.manual_seed(123)

start = time.time()

token_ids = generate(
    model=model,
    idx=text_to_token_ids(PROMPT, tokenizer).to(device),
    max_new_tokens=MAX_NEW_TOKENS,
    context_size=LLAMA32_CONFIG["context_length"],
    top_k=TOP_K,
    temperature=TEMPERATURE
)

total_time = time.time() - start
print(f"Time: {total_time:.2f} sec")
print(f"{int(len(token_ids[0])/total_time)} tokens/sec")

if torch.cuda.is_available():
    max_mem_bytes = torch.cuda.max_memory_allocated()
    max_mem_gb = max_mem_bytes / (1024 ** 3)
    print(f"Max memory allocated: {max_mem_gb:.2f} GB")

output_text = token_ids_to_text(token_ids, tokenizer)

if "instruct" in MODEL_FILE:
    output_text = clean_text(output_text)

print("\n\nOutput text:\n\n", output_text)
```

Llama 3.2 1B Instruct modelini kullanırken çıktı aşağıdakine benzer görünmelidir:

```
Time: 3.17 sec
50 tokens/sec
Max memory allocated: 2.91 GB


Output text:

 Llamas are herbivores, which means they primarily eat plants. Their diet consists mainly of:

1. Grasses: Llamas love to graze on various types of grasses, including tall grasses and grassy meadows.
2. Hay: Llamas also eat hay, which is a dry, compressed form of grass or other plants.
3. Alfalfa: Alfalfa is a legume that is commonly used as a hay substitute in llama feed.
4. Other plants: Llamas will also eat other plants, such as clover, dandelions, and wild grasses.

It's worth noting that the specific diet of llamas can vary depending on factors such as the breed,
```

&nbsp;
#### Uzman ipucu 1: FlashAttention ile çıkarımı hızlandırın

`Llama3Model` yerine, doğrudan yerine geçecek şekilde `Llama3ModelFast` kullanabilirsiniz. Daha fazla bilgi için [pkg/llms_from_scratch/llama3.py](../../pkg/llms_from_scratch/llama3.py) kodunu incelemenizi tavsiye ederim.

`Llama3ModelFast`, `GroupedQueryAttention` modülündeki sıfırdan yazdığım ölçeklenmiş nokta çarpımı kodunu, Ampere ve daha yeni GPU'larda `FlashAttention` kullanan PyTorch'un `scaled_dot_product` fonksiyonuyla değiştirir.

Aşağıdaki tablo bir A100 üzerindeki performans karşılaştırmasını gösterir:

|                 | Token/saniye | Bellek  |
| --------------- | ---------- | ------- |
| Llama3Model     | 42         | 2.91 GB |
| Llama3ModelFast | 54         | 2.91 GB |

&nbsp;
#### Uzman ipucu 2: derleme ile çıkarımı hızlandırın


4 kata varan hızlanma için şunu:

```python
model.to(device)
```

şununla değiştirin:

```python
model = torch.compile(model)
model.to(device)
```

Not: Derleme sırasında birkaç dakikalık kayda değer bir başlangıç maliyeti vardır ve hızlanma ilk `generate` çağrısından sonra devreye girer.

Aşağıdaki tablo, art arda yapılan `generate` çağrıları için bir A100 üzerindeki performans karşılaştırmasını gösterir:

|                 | Token/saniye | Bellek  |
| --------------- | ---------- | ------- |
| Llama3Model     | 170        | 3.12 GB |
| Llama3ModelFast | 177        | 3.61 GB |

&nbsp;
#### Uzman ipucu 3: derleme ile çıkarımı hızlandırın

Modeli bir CPU üzerinde çalıştırırken, doğrudan yerine geçen KV önbellekli `Llama3Model` sürümünü kullanarak çıkarım performansını kayda değer biçimde artırabilirsiniz. (KV önbellekleri hakkında daha fazla bilgi için [Understanding and Coding the KV Cache in LLMs from Scratch](https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms) yazıma bakın.)

```python
from llms_from_scratch.kv_cache.llama3 import Llama3Model
from llms_from_scratch.kv_cache.generate import generate_text_simple

model = Llama3Model(LLAMA32_CONFIG)
# ...
token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(PROMPT, tokenizer).to(device),
    max_new_tokens=MAX_NEW_TOKENS,
    context_size=LLAMA32_CONFIG["context_length"],
)
```

Tepe bellek kullanımının yalnızca Nvidia CUDA cihazları için listelendiğini unutmayın; çünkü hesaplaması daha kolaydır. Ancak diğer cihazlardaki bellek kullanımı benzer bir hassasiyet biçimi kullandığı için muhtemelen benzerdir ve KV önbelleği depolaması, üretilen 150 token'lık metin için burada daha da düşük bellek kullanımına yol açar (yine de farklı cihazlar matris çarpımını farklı uygulayabilir ve farklı tepe bellek gereksinimleri doğurabilir; ayrıca daha uzun bağlam uzunluklarında KV önbelleği belleği karşılanamaz ölçüde artabilir).

| Model       | Mod               | Donanım         | Token/saniye | GPU Belleği (VRAM) |
| ----------- | ----------------- | --------------- | ---------- | ----------------- |
| Llama3Model | Regular           | Mac Mini M4 CPU | 1          | -                 |
| Llama3Model | Regular compiled  | Mac Mini M4 CPU | 1          | -                 |
| Llama3Model | KV cache          | Mac Mini M4 CPU | 68         | -                 |
| Llama3Model | KV cache compiled | Mac Mini M4 CPU | 86         | -                 |
|             |                   |                 |            |                   |
| Llama3Model | Regular           | Mac Mini M4 GPU | 15         | -                 |
| Llama3Model | Regular compiled  | Mac Mini M4 GPU | Error      | -                 |
| Llama3Model | KV cache          | Mac Mini M4 GPU | 62         | -                 |
| Llama3Model | KV cache compiled | Mac Mini M4 GPU | Error      | -                 |
|             |                   |                 |            |                   |
| Llama3Model | Regular           | Nvidia A100 GPU | 42         | 2.91 GB           |
| Llama3Model | Regular compiled  | Nvidia A100 GPU | 170        | 3.12 GB           |
| Llama3Model | KV cache          | Nvidia A100 GPU | 58         | 2.87 GB           |
| Llama3Model | KV cache compiled | Nvidia A100 GPU | 161        | 3.61 GB           |

Yukarıdaki tüm ayarların aynı metin çıktılarını ürettiğinin test edildiğini unutmayın.
