# Sıfırdan Qwen3

> 🇹🇷 **Türkçe çeviri.** Orijinal İngilizce sürüm: [README.md](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch05/11_qwen3/README.md) · Kod, çıktı ve tablo verileri birebir korunmuştur.

Bu klasördeki [standalone-qwen3.ipynb](standalone-qwen3.ipynb) Jupyter not defteri, Qwen3 0.6B, 1.7B, 4B, 8B ve 32B modellerinin sıfırdan bir uygulamasını içerir.

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/qwen/qwen-overview.webp">


Bu klasördeki [standalone-qwen3-moe.ipynb](standalone-qwen3-moe.ipynb) ve [standalone-qwen3-moe-plus-kvcache.ipynb](standalone-qwen3-moe-plus-kvcache.ipynb) Jupyter not defterleri, Thinking, Instruct ve Coder model varyantları dahil olmak üzere 30B-A3B Uzmanlar Karışımı (MoE) modelinin sıfırdan bir uygulamasını içerir.

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/qwen/qwen3-coder-flash-overview.webp?123" width="430px">

&nbsp;
# Sıfırdan Qwen3 kodu

Bu klasördeki bağımsız not defterleri, sıfırdan yazılmış kodu doğrusal bir akışla içerir:

1. [standalone-qwen3.ipynb](standalone-qwen3.ipynb): Ek özellikler içermeyen yoğun (dense) Qwen3 modeli
2. [standalone-qwen3-plus-kvcache.ipynb](standalone-qwen3-plus-kvcache.ipynb): Yukarıdakinin aynısı, ancak daha iyi çıkarım verimliliği için KV önbellekli
3. [standalone-qwen3-moe.ipynb](standalone-qwen3-moe.ipynb): İlk not defteri gibi, ancak Uzmanlar Karışımı (MoE) varyantı
4. [standalone-qwen3-moe-plus-kvcache.ipynb](standalone-qwen3-moe-plus-kvcache.ipynb): Yukarıdakinin aynısı, ancak daha iyi çıkarım verimliliği için KV önbellekli

Alternatif olarak, kodu [burada](../../pkg/llms_from_scratch/) bir Python paketi hâline de getirdim (birim testleri ve CI dahil); aşağıda açıklandığı gibi çalıştırabilirsiniz.

&nbsp;
# Eğitim

`Qwen3Model` sınıfı, `GPTModel` sınıfına benzer bir tarzda uygulanmıştır; bu nedenle 5. bölümdeki eğitim ile 6. ve 7. bölümlerdeki ince ayar için doğrudan yerine kullanılabilir.


&nbsp;
# Qwen3'ü `llms-from-scratch` paketiyle kullanmak

Sıfırdan Qwen3 uygulamasını kolayca kullanmak için, bu depodaki [pkg/llms_from_scratch](../../pkg/llms_from_scratch) kaynak koduna dayanan `llms-from-scratch` PyPI paketini de kullanabilirsiniz.

&nbsp;
#### 1) Kurulum

```bash
pip install llms_from_scratch tokenizers
```

&nbsp;
#### 2) Model ve metin üretimi ayarları

Hangi modelin kullanılacağını belirtin:

```python
USE_REASONING_MODEL = True
# Uses the base model if USE_REASONING_MODEL = False

USE_INSTRUCT_MODEL = False
# Uses the instruct mode (without reasoning) if 
# USE_REASONING_MODEL = True
# USE_INSTRUCT_MODEL = True
# This setting does have no effect if USE_REASONING_MODEL = False


# Use
# USE_REASONING_MODEL = True
# For Qwen3 Coder Flash model as well
```

Kullanıcı tarafından tanımlanabilen temel metin üretimi ayarları. 150 token ile 0.6B model yaklaşık 1,5 GB bellek gerektirir.

```python
MAX_NEW_TOKENS = 150
TEMPERATURE = 0.
TOP_K = 1
```

&nbsp;
#### 3a) 0.6B modelin ağırlıklarının indirilmesi ve yüklenmesi

Aşağıdaki kod, yukarıdaki model seçimine (akıl yürütme veya temel model) göre ağırlık dosyasını otomatik olarak indirir. Bu bölümün 0.6B modele odaklandığını unutmayın. Daha büyük modellerden biriyle (1.7B, 4B, 8B veya 32B) çalışmak istiyorsanız bu bölümü atlayıp 3b) ile devam edin.

```python
from llms_from_scratch.qwen3 import download_from_huggingface

repo_id = "rasbt/qwen3-from-scratch"

if USE_REASONING_MODEL:
    filename = "qwen3-0.6B.pth"
    local_dir = "Qwen3-0.6B"    
else:
    filename = "qwen3-0.6B-base.pth"   
    local_dir = "Qwen3-0.6B-Base"

download_from_huggingface(
    repo_id=repo_id,
    filename=filename,
    local_dir=local_dir
)
```

Model ağırlıkları daha sonra şöyle yüklenir:

```python
from pathlib import Path
import torch

from llms_from_scratch.qwen3 import Qwen3Model, QWEN_CONFIG_06_B

model_file = Path(local_dir) / filename

model = Qwen3Model(QWEN_CONFIG_06_B)
model.load_state_dict(torch.load(model_file, weights_only=True, map_location="cpu"))

device = (
    torch.device("cuda") if torch.cuda.is_available() else
    torch.device("mps") if torch.backends.mps.is_available() else
    torch.device("cpu")
)
model.to(device);
```

&nbsp;
#### 3b) Daha büyük Qwen modellerinin ağırlıklarının indirilmesi ve yüklenmesi

Daha büyük Qwen modellerinden biriyle (örneğin 1.7B, 4B, 8B veya 32B) çalışmak istiyorsanız, 3a) altındaki kod yerine aşağıdaki kodu kullanın; bu ek kod bağımlılıkları gerektirir:

```bash
pip install safetensors huggingface_hub
```

Ardından aşağıdaki kodu kullanın (istediğiniz model boyutunu seçmek için `USE_MODEL` değerini uygun şekilde değiştirin)

```python
USE_MODEL = "1.7B"

if USE_MODEL == "1.7B":
    from llms_from_scratch.qwen3 import QWEN3_CONFIG_1_7B as QWEN3_CONFIG
elif USE_MODEL == "4B":
    from llms_from_scratch.qwen3 import QWEN3_CONFIG_4B as QWEN3_CONFIG
elif USE_MODEL == "8B":
    from llms_from_scratch.qwen3 import QWEN3_CONFIG_8B as QWEN3_CONFIG
elif USE_MODEL == "14B":
    from llms_from_scratch.qwen3 import QWEN3_CONFIG_14B as QWEN3_CONFIG
elif USE_MODEL == "32B":
    from llms_from_scratch.qwen3 import QWEN3_CONFIG_32B as QWEN3_CONFIG
elif USE_MODEL == "30B-A3B":
    from llms_from_scratch.qwen3 import QWEN3_CONFIG_30B_A3B as QWEN3_CONFIG
else:
    raise ValueError("Invalid USE_MODEL name.")
    
repo_id = f"Qwen/Qwen3-{USE_MODEL}"
local_dir = f"Qwen3-{USE_MODEL}"

if not USE_REASONING_MODEL:
  repo_id = f"{repo_id}-Base"
  local_dir = f"{local_dir}-Base"
```

Şimdi ağırlıkları indirip `model` içine yükleyin:

```python
from llms_from_scratch.qwen3 import (
    Qwen3Model,
    download_from_huggingface_from_snapshots,
    load_weights_into_qwen
)

device = (
    torch.device("cuda") if torch.cuda.is_available() else
    torch.device("mps") if torch.backends.mps.is_available() else
    torch.device("cpu")
)

with device:
    model = Qwen3Model(QWEN3_CONFIG)

weights_dict = download_from_huggingface_from_snapshots(
    repo_id=repo_id,
    local_dir=local_dir
)
load_weights_into_qwen(model, QWEN3_CONFIG, weights_dict)
model.to(device)  # only required for the MoE models
del weights_dict  # delete weight dictionary to free up disk space
```


&nbsp;

#### 4) Tokenizer'ı başlatmak

Aşağıdaki kod tokenizer'ı indirir ve başlatır:

```python
from llms_from_scratch.qwen3 import Qwen3Tokenizer

if USE_REASONING_MODEL:
    tok_filename = "tokenizer.json"    
else:
    tok_filename = "tokenizer-base.json"   

tokenizer = Qwen3Tokenizer(
    tokenizer_file_path=tokenizer_file_path,
    repo_id=repo_id,
    apply_chat_template=USE_REASONING_MODEL,
    add_generation_prompt=USE_REASONING_MODEL,
    add_thinking=not USE_INSTRUCT_MODEL
)
```



&nbsp;

#### 5) Metin üretmek

Son olarak, aşağıdaki kodla metin üretebiliriz:

```python
prompt = "Give me a short introduction to large language models."
input_token_ids = tokenizer.encode(prompt)
```





```python
from llms_from_scratch.ch05 import generate
import time

torch.manual_seed(123)

start = time.time()

output_token_ids = generate(
    model=model,
    idx=torch.tensor(input_token_ids, device=device).unsqueeze(0),
    max_new_tokens=150,
    context_size=QWEN_CONFIG_06_B["context_length"],
    top_k=1,
    temperature=0.
)

total_time = time.time() - start
print(f"Time: {total_time:.2f} sec")
print(f"{int(len(output_token_ids[0])/total_time)} tokens/sec")

if torch.cuda.is_available():
    max_mem_bytes = torch.cuda.max_memory_allocated()
    max_mem_gb = max_mem_bytes / (1024 ** 3)
    print(f"Max memory allocated: {max_mem_gb:.2f} GB")

output_text = tokenizer.decode(output_token_ids.squeeze(0).tolist())

print("\n\nOutput text:\n\n", output_text + "...")
```

Qwen3 0.6B akıl yürütme modelini kullanırken çıktı aşağıdakine benzer görünmelidir (bu, bir A100 üzerinde çalıştırılmıştır):

```
Time: 6.35 sec
25 tokens/sec
Max memory allocated: 1.49 GB


Output text:

 <|im_start|>user
Give me a short introduction to large language models.<|im_end|>
Large language models (LLMs) are advanced artificial intelligence systems designed to generate human-like text. They are trained on vast amounts of text data, allowing them to understand and generate coherent, contextually relevant responses. LLMs are used in a variety of applications, including chatbots, virtual assistants, content generation, and more. They are powered by deep learning algorithms and can be fine-tuned for specific tasks, making them versatile tools for a wide range of industries.<|endoftext|>Human resources department of a company is planning to hire 100 new employees. The company has a budget of $100,000 for the recruitment process. The company has a minimum wage of $10 per hour. The company has a total of...
```



Daha büyük modeller için, her token'ı üretilir üretilmez yazdıran akış (streaming) varyantını tercih edebilirsiniz:

```python
from llms_from_scratch.generate import generate_text_simple_stream

input_token_ids_tensor = torch.tensor(input_token_ids, device=device).unsqueeze(0)

for token in generate_text_simple_stream(
    model=model,
    token_ids=input_token_ids_tensor,
    max_new_tokens=150,
    eos_token_id=tokenizer.eos_token_id
):
    token_id = token.squeeze(0).tolist()
    print(
        tokenizer.decode(token_id),
        end="",
        flush=True
    )
```

```
 <|im_start|>user
Give me a short introduction to large language models.<|im_end|>
Large language models (LLMs) are advanced artificial intelligence systems designed to generate human-like text. They are trained on vast amounts of text data, allowing them to understand and generate coherent, contextually relevant responses. LLMs are used in a variety of applications, including chatbots, virtual assistants, content generation, and more. They are powered by deep learning algorithms and can be fine-tuned for specific tasks, making them versatile tools for a wide range of industries.<|endoftext|>Human resources department of a company is planning to hire 100 new employees. The company has a budget of $100,000 for the recruitment process. The company has a minimum wage of $10 per hour. The company has a total of...
```



&nbsp;

#### Uzman ipucu 1: derleme ile çıkarımı hızlandırın


4 kata varan hızlanma için şunu:

```python
model.to(device)
```

şununla değiştirin:

```python
model.to(device)
model = torch.compile(model)
```

Not: Derleme sırasında birkaç dakikalık kayda değer bir başlangıç maliyeti vardır ve hızlanma ilk `generate` çağrısından sonra devreye girer.

Aşağıdaki tablo, art arda yapılan `generate` çağrıları için bir A100 üzerindeki performans karşılaştırmasını gösterir:

|                          | Donanım         | Token/saniye | Bellek   |
| ------------------------ | ----------------|----------- | -------- |
| Qwen3Model 0.6B          | Nvidia A100 GPU | 25         | 1.49 GB  |
| Qwen3Model 0.6B compiled | Nvidia A100 GPU | 107        | 1.99 GB  |


&nbsp;
#### Uzman ipucu 2: KV önbelleği ile çıkarımı hızlandırın

Modeli bir CPU üzerinde çalıştırırken, doğrudan yerine geçen KV önbellekli `Qwen3Model` sürümünü kullanarak çıkarım performansını kayda değer biçimde artırabilirsiniz. (KV önbellekleri hakkında daha fazla bilgi için [Understanding and Coding the KV Cache in LLMs from Scratch](https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms) yazıma bakın.)

```python
from llms_from_scratch.kv_cache.qwen3 import Qwen3Model
from llms_from_scratch.kv_cache.generate import generate_text_simple

model = Qwen3Model(QWEN_CONFIG_06_B)
# ...
token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(PROMPT, tokenizer).to(device),
    max_new_tokens=MAX_NEW_TOKENS,
    context_size=QWEN_CONFIG_06_B["context_length"],
)
```

Tepe bellek kullanımının yalnızca Nvidia CUDA cihazları için listelendiğini unutmayın; çünkü hesaplaması daha kolaydır. Ancak diğer cihazlardaki bellek kullanımı benzer bir hassasiyet biçimi kullandığı için muhtemelen benzerdir ve KV önbelleği depolaması, üretilen 150 token'lık metin için burada daha da düşük bellek kullanımına yol açar (yine de farklı cihazlar matris çarpımını farklı uygulayabilir ve farklı tepe bellek gereksinimleri doğurabilir; ayrıca daha uzun bağlam uzunluklarında KV önbelleği belleği karşılanamaz ölçüde artabilir).

| Model           | Mod               | Donanım         | Token/saniye | GPU Belleği (VRAM) |
| --------------- | ----------------- | --------------- | ---------- | ----------------- |
| Qwen3Model 0.6B | Regular           | Mac Mini M4 CPU | 1          | -                 |
| Qwen3Model 0.6B | Regular compiled  | Mac Mini M4 CPU | 1          | -                 |
| Qwen3Model 0.6B | KV cache          | Mac Mini M4 CPU | 80         | -                 |
| Qwen3Model 0.6B | KV cache compiled | Mac Mini M4 CPU | 137        | -                 |
|                 |                   |                 |            |                   |
| Qwen3Model 0.6B | Regular           | Mac Mini M4 GPU | 21         | -                 |
| Qwen3Model 0.6B | Regular compiled  | Mac Mini M4 GPU | Error      | -                 |
| Qwen3Model 0.6B | KV cache          | Mac Mini M4 GPU | 28         | -                 |
| Qwen3Model 0.6B | KV cache compiled | Mac Mini M4 GPU | Error      | -                 |
|                 |                   |                 |            |                   |
| Qwen3Model 0.6B | Regular           | Nvidia A100 GPU | 26         | 1.49 GB           |
| Qwen3Model 0.6B | Regular compiled  | Nvidia A100 GPU | 107        | 1.99 GB           |
| Qwen3Model 0.6B | KV cache          | Nvidia A100 GPU | 25         | 1.47 GB           |
| Qwen3Model 0.6B | KV cache compiled | Nvidia A100 GPU | 90         | 1.48 GB           |

Yukarıdaki tüm ayarların aynı metin çıktılarını ürettiğinin test edildiğini unutmayın.



&nbsp;

#### Uzman ipucu 3: yığın hâlinde (batched) çıkarım

Verimi (throughput) yığın hâlinde çıkarımla daha da artırabiliriz. Artık daha fazla girdi dizisiyle çıkarım yaptığımız için bu bire bir karşılaştırma sayılmaz; yine de saniyedeki token verimini artırır, karşılığında bellek kullanımı yükselir.

Bu, yalnızca istemin hazırlanmasıyla ilgili küçük bir kod değişikliği gerektirir. Örneğin, aşağıdaki yığın istemini ele alalım:

```python
from llms_from_scratch.ch04 import generate_text_simple
from llms_from_scratch.qwen3 import Qwen3Model, QWEN_CONFIG_06_B
# ...

prompts = [
    "Give me a short introduction to neural networks.",
    "Give me a short introduction to machine learning.",
    "Give me a short introduction to deep learning models.",
    "Give me a short introduction to natural language processing.",
    "Give me a short introduction to generative AI systems.",
    "Give me a short introduction to transformer architectures.",
    "Give me a short introduction to supervised learning methods.",
    "Give me a short introduction to unsupervised learning.",
]

tokenized_prompts = [tokenizer.encode(p) for p in prompts]
max_len = max(len(t) for t in tokenized_prompts)
padded_token_ids = [
    t + [tokenizer.pad_token_id] * (max_len - len(t)) for t in tokenized_prompts
]
input_tensor = torch.tensor(padded_token_ids).to(device)

output_token_ids = generate_text_simple(
    model=model,
    idx=input_tensor,
    max_new_tokens=150,
    context_size=QWEN_CONFIG_06_B["context_length"],
)
```

KV önbellekli sürümün kodu da benzerdir; tek fark, doğrudan yerine geçen şu içe aktarmaları kullanmayı gerektirmesidir:

```python
from llms_from_scratch.kv_cache_batched.generate import generate_text_simple
from llms_from_scratch.kv_cache_batched.qwen3 import Qwen3Model
```


Aşağıdaki deneyler 8'lik bir yığın boyutuyla çalıştırılmıştır.

| Model            | Mod               | Donanım         | Yığın boyutu | Token/saniye | GPU Belleği (VRAM) |
| ---------------- | ----------------- | --------------- | ---------- | ---------- | ----------------- |
| Qwen3Model  0.6B | Regular           | Mac Mini M4 CPU | 8          | 2          | -                 |
| Qwen3Model 0.6B  | Regular compiled  | Mac Mini M4 CPU | 8          | -          | -                 |
| Qwen3Model 0.6B  | KV cache          | Mac Mini M4 CPU | 8          | 92         | -                 |
| Qwen3Model 0.6B  | KV cache compiled | Mac Mini M4 CPU | 8          | 128        | -                 |
|                  |                   |                 |            |            |                   |
| Qwen3Model 0.6B  | Regular           | Mac Mini M4 GPU | 8          | 36         | -                 |
| Qwen3Model 0.6B  | Regular compiled  | Mac Mini M4 GPU | 8          | -          | -                 |
| Qwen3Model 0.6B  | KV cache          | Mac Mini M4 GPU | 8          | 61         | -                 |
| Qwen3Model 0.6B  | KV cache compiled | Mac Mini M4 GPU | 8          | -          | -                 |
|                  |                   |                 |            |            |                   |
| Qwen3Model 0.6B  | Regular           | Nvidia A100 GPU | 8          | 184        | 2.19 GB           |
| Qwen3Model 0.6B  | Regular compiled  | Nvidia A100 GPU | 8          | 351        | 2.19 GB           |
| Qwen3Model 0.6B  | KV cache          | Nvidia A100 GPU | 8          | 140        | 3.13 GB           |
| Qwen3Model 0.6B  | KV cache compiled | Nvidia A100 GPU | 8          | 280        | 1.75 GB           |
