# Bölüm 8 Bonus Materyali: Damıtma Verisi Üretmek

> 🇹🇷 **Türkçe çeviri.** Orijinal İngilizce sürüm: [README.md](https://github.com/rasbt/reasoning-from-scratch/blob/main/ch08/02_generate_distillation_data/README.md) · Komut, kod ve çıktı blokları birebir korunmuştur.

Bu klasör, 8. bölümde ele alındığı gibi, daha küçük bir akıl yürütme modelini eğitmek için damıtma verisi olarak kullanılabilecek öğretmen (teacher) çıktılarını matematik problemleri için üretmeye yarayan betikleri içerir.

&nbsp;
**İçindekiler:**

- [Dosyalar](#dosyalar)
- [Girdi Veri Biçimi](#girdi-veri-biçimi)
- [Çıktı Biçimi](#çıktı-biçimi)
- [1. Ollama ile yerel üretim](#1-ollama-ile-yerel-üretim)
  - [1.1 Ollama kurulumu](#11-ollama-kurulumu)
  - [1.2 Ollama ile yerel veri üretimi](#12-ollama-ile-yerel-veri-üretimi)
  - [1.3 Ollama sorun giderme](#13-ollama-sorun-giderme)
    - [1.3.1 Ollama çalışmıyor](#131-ollama-çalışmıyor)
    - [1.3.2 Ollama modeli indirilmemiş](#132-ollama-modeli-indirilmemiş)
- [2. OpenRouter ile barındırılan üretim](#2-openrouter-ile-barındırılan-üretim)
  - [2.1 OpenRouter kurulumu](#21-openrouter-kurulumu)
  - [2.2 OpenRouter ile veri üretimi](#22-openrouter-ile-veri-üretimi)
- [Damıtma için veri kümeleri](#damıtma-için-veri-kümeleri)
- [Veri kümesi istatistikleri](#veri-kümesi-istatistikleri)
- [Öğretmen doğruluğu](#öğretmen-doğruluğu)
- [MATH-500 damıtma veri kümesi üretmek](#math-500-damıtma-veri-kümesi-üretmek)
- [12.000 MATH örneğinden oluşan bir damıtma veri kümesi üretmek](#12000-math-örneğinden-oluşan-bir-damıtma-veri-kümesi-üretmek)


&nbsp;
## Dosyalar

- [average_field_lengths_json.py](average_field_lengths_json.py): Üretilen veri kümelerinin temel istatistiklerini yazdıran yardımcı betik.
- [generate_with_ollama.py](generate_with_ollama.py): Damıtma için model yanıtlarını üretmek üzere Ollama'yı kullanır. Bu betik, yerelde çalıştırabileceğiniz daha küçük modellerden (örneğin Qwen3 4B, gpt-oss 20B, DeepSeek R1 32B vb.) damıtmak istiyorsanız önerilir. 
- [generate_with_openrouter.py](generate_with_openrouter.py): Damıtma için model yanıtlarını üretmek üzere OpenRouter API'si aracılığıyla modelleri kullanır. Bu, yerelde çalıştırılamayacak kadar büyük olan DeepSeek R1 (671B) veya Kimi K2.5 (1T) gibi daha büyük modeller kullanılırken önerilir.
- [math_train_sample.json](math_train_sample.json): Hızlı sağlık kontrolleri için küçük örnek veri kümesi.

&nbsp;
## Girdi Veri Biçimi

Her iki betik de `--math_json` ile bir JSON dosyası bekler. Her nesnede en azından şunlar bulunmalıdır:

- `problem` (metin): Matematik sorusu.
- `answer` (metin): Doğru (ground-truth) yanıt.

`level`, `type` ve `unique_id` gibi ek anahtarlar yok sayılır. Örnek bir yapı için, 6, 7 ve 8. bölümlerde kullandığımız [math_full_minus_math500.json](https://github.com/rasbt/math_full_minus_math500/blob/main/math_full_minus_math500.json) dosyasına dayanan [math_train_sample.json](math_train_sample.json) dosyasına bakabilirsiniz. 

Bunu 12.000 örneğin tamamına uygulamak için [math_full_minus_math500.json](https://github.com/rasbt/math_full_minus_math500/blob/main/math_full_minus_math500.json) dosyasını indirip betiklere `--math_json math_full_minus_math500.json` ile verin. Bunun uzun süreceğini unutmayın; bu nedenle dosyayı birkaç yüz veya bin örneğe kırpmanızı öneririm.


&nbsp;
## Çıktı Biçimi

Her iki betik de her satırı şöyle görünen bir JSON dizisi yazar:

```
{
  "problem": "...",             # The original "problem"
  "gtruth_answer": "...",       # The original "answer"
  "message_thinking": "...",    # The model's thinking stream
  "message_content": "..."      # The model's final answer
}
```

Notlar:

- Girdi JSON dosyasındaki orijinal `"answer"` alanı, belirsizliği önlemek için `"gtruth_answer"` olarak yeniden adlandırıldı (çünkü "answer" genel bir terimdir ve modelin yanıtına da işaret edebilir).
- Dosyalar her örnekten sonra artımlı olarak yazılır; böylece ara dosyalarla çalışmak veya koşuyu yarıda kesmek mümkündür.
- Betikler, yarıda kesilmiş bir koşuya devam etmek için bir `--resume` seçeneğine sahiptir.

&nbsp;
## 1. Ollama ile yerel üretim


- Ollama, LLM'leri verimli şekilde çalıştırmaya yarayan açık kaynaklı bir uygulamadır.
- Verimliliği en üst düzeye çıkarmak için LLM'leri saf C/C++ ile uygulayan llama.cpp ([https://github.com/ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp)) etrafında bir sarmalayıcıdır.
- Bunun, LLM'leri eğitmek veya ince ayar yapmak için değil, metin üretmek (çıkarım) için kullanılan bir araç olduğunu unutmayın.

&nbsp;
### 1.1 Ollama kurulumu


- Aşağıdaki kodu çalıştırmadan önce [https://ollama.com](https://ollama.com) adresini ziyaret edip talimatları izleyerek ollama'yı kurun (örneğin "Download" düğmesine tıklayıp işletim sisteminize uygun ollama uygulamasını indirin).
- macOS ve Windows kullanıcıları indirdikleri ollama uygulamasına tıklasın; komut satırı kullanımını kurmanızı isterse "evet" deyin.
- Linux kullanıcıları ollama web sitesinde verilen kurulum komutunu kullanabilir.
- Ollama'yı bilgisayarımızda çalıştırmanın 3 yolu vardır:


&nbsp;
**1. `ollama serve`**

- Bu, ollama arka ucunu genellikle `http://localhost:11434` adresinde bir sunucu olarak çalıştırır. API üzerinden çağırana kadar bir model yüklemez. Ollama'yı Python üzerinden kullanmak istiyorsak istediğimiz budur.

&nbsp;
**2. `ollama run deepseek-r1:8b`**

- Bu pratik bir sarmalayıcıdır. Sunucu zaten çalışmıyorsa başlatır, ardından modeli (ilk seferde) indirir ve bizi modelle sohbet edebileceğimiz etkileşimli bir terminale bırakır. Arka planda aynı sunucu API'sini kullanır.
- `deepseek-r1:8b` modeli, `--max_new_tokens 8192` token ayarıyla yaklaşık 30 GB RAM gerektirecektir.
  - Daha fazla RAM'iniz varsa, daha yüksek kaliteli yanıtlar için daha büyük modelleri denemenizi öneririm, örneğin `deepseek-r1:32b` (yaklaşık 60 GB gerektirir)
  - Daha az RAM'iniz varsa daha küçük bir model seçmeyi deneyin; daha küçük R1 modellerinin listesini [burada](https://ollama.com/library/deepseek-r1) bulabilirsiniz. Ayrıca bir DeepSeek modeli kullanmak yerine, ilginizi çekebilecek başka modelleri seçmek için [Ollama web sitesindeki](https://ollama.com/) "Search model" alanını kullanmaktan çekinmeyin.
  - Alternatif olarak, RAM kullanımını azaltmak için `--max_new_tokens 8192` değerini `--max_new_tokens 2048` olarak düşürebilirsiniz, ancak bu bazı yanıtları erken kesebilir.

&nbsp;
**3. Ollama masaüstü uygulaması**

- Bu, aynı arka ucu otomatik olarak çalıştırır ve üzerine bir grafik arayüz (GUI) sunar (yukarıdaki şekilde gösterildiği gibi).
  Ayrıca varsayılan ayarlar (sistem istemi, temperature, durdurma dizileri) uygular; bu da yanıtların ham API kullanımından neden farklı göründüğünü açıklayabilir.

&nbsp;
### 1.2 Ollama ile yerel veri üretimi

```bash
uv run generate_with_ollama.py \
  --math_json math_train_sample.json \
  --dataset_size 5 \
  --model deepseek-r1:8b \
  --max_new_tokens 8192 \
  --out_file sample_ollama_outputs.json
```

`uv` kullanıcısı değilseniz, `uv run` yerine `python` yazın.

Beklenen çıktı şöyle olmalıdır:

```
Loading model: deepseek-r1:8b
Using CUDA:0
Model ready
5/5 | MATH-500: 5/5 | ETA: 00s        
Total time: 3.2 min

Wrote 5 rows to: /home/rasbt/reasoning-from-scratch-codedev/ch08/sample_ollama_outputs.json
```

Elde edilen [sample_ollama_outputs.json](sample_ollama_outputs.json) dosyasındaki kayıtlar şöyledir:

```json
  {
    "problem": "A rectangular band formation...",
    "gtruth_answer": "98",
    "message_thinking": "I need to find the largest number of...",
    "message_content": "The function is continuous..."
  },
```

`"message_thinking"` alanı düşünce zinciri açıklamasını, `"message_content"` ise nihai yanıtı içerir. Örneğin bunlar şöyle birleştirilebilir:

```python
complete_answer = f"<think>{data['message_thinking']}</think>\n\n{data['message_content']}"
```

Yani,

```
"<think>I need to find the largest number of...</think>

The function is continuous..."
```

&nbsp;
### 1.3 Ollama sorun giderme

Aşağıda, Ollama veri üretim betiğini çalıştırırken karşılaşılan bazı yaygın sorunlar yer alıyor.

&nbsp;
#### 1.3.1 Ollama çalışmıyor

Şuna benzer bir hata görürseniz: 

```
Loading model: deepseek-r1:32b
Using CUDA:0
Traceback (most recent call last):
  File "/home/rasbt/reasoning-from-scratch-codedev/ch08/generate_with_ollama.py", line 379, in <module>
    query_ollama_chat(
  File "/home/rasbt//reasoning-from-scratch-codedev/ch08/generate_with_ollama.py", line 235, in query_ollama_chat
    raise RuntimeError(
RuntimeError: Failed to query Ollama after 3 attempt(s). Last error: <urlopen error [Errno 111] Connection refused>
```

`ollama serve` komutunun (farklı bir terminal sekmesinde) çalıştığından emin olun.

&nbsp;
#### 1.3.2 Ollama modeli indirilmemiş

Aşağıdaki hatayı görürseniz:

```
Loading model: deepseek-r1:8b
Using CUDA:0
Traceback (most recent call last):
  File "/home/rasbt/reasoning-from-scratch-codedev/ch08/generate_with_ollama.py", line 379, in <module>
    query_ollama_chat(
  File "/home/rasbt/reasoning-from-scratch-codedev/ch08/generate_with_ollama.py", line 235, in query_ollama_chat
    raise RuntimeError(
RuntimeError: Failed to query Ollama after 3 attempt(s). Last error: HTTP 404 from Ollama at http://localhost:11434/api/chat: {"error":"model 'deepseek-r1:8b' not found"}
```

bu, modelin henüz indirilmediği anlamına gelir. Bu durumda ayrı bir terminalde `ollama run deepseek-r1:8b` komutunu çalıştırın; bu, modeli indirip bir sohbet başlatacaktır. Modeli sohbette deneyip ardından `\bye` ile çıkabilirsiniz.


&nbsp;
## 2. OpenRouter ile barındırılan üretim

Modelleri yerelde çalıştırmak istiyorsanız Ollama kullanışlıdır. Ancak donanımımızda yerelde çalıştırılamayacak kadar büyük olan birkaç büyük model vardır (671 milyar parametreli DeepSeek R1 modeli gibi). Bu durumlar için, bulutta barındırılan çok çeşitli açık ağırlıklı ve tescilli LLM'leri ChatGPT benzeri bir API üzerinden kullanmamızı sağlayan [OpenRouter](https://openrouter.ai) hizmetini öneririm. 

Bu yazının yazıldığı tarihte, [DeepSeek R1](https://openrouter.ai/deepseek/deepseek-r1) 1 milyon girdi token'ı için \$0.70 ve 1 milyon çıktı token'ı için \$2.50 tutarındadır. OpenRouter'da çok daha ucuz (ve daha hızlı) pek çok model olduğunu unutmayın; daha yeni [DeepSeek V3.2](https://openrouter.ai/deepseek/deepseek-v3.2) modeli bile 1 milyon çıktı token'ı için yalnızca $0.40 tutar.

Bunu söyledikten sonra, basit bir maliyet hesabı yapalım. Ortalama 11 token'lık girdi istemi uzunluğu ve ortalama 1524 token'lık yanıt uzunluğu göz önüne alındığında, 1000 MATH sorusunun yanıtlarını üretmek yaklaşık $3.82 tutar.

Ayrıntılı dökümü şöyle:

- Toplam girdi token'ı: 11 × 1000 = 11,000 
- Toplam çıktı token'ı: 1524 × 1000 = 1,524,000 
- Girdi maliyeti: `(11,000 / 1,000,000) × $0.70 = $0.0077`
- Çıktı maliyeti: `(1,524,000 / 1,000,000) × $2.50 = $3.81`
- Toplam maliyet: `$3.81 + $0.0077 ≈ $3.82`

&nbsp;
### 2.1 OpenRouter kurulumu

Kurulum oldukça basittir. Tek yapmanız gereken [OpenRouter](https://openrouter.ai/) üzerinde bir hesap oluşturmak, [https://openrouter.ai/settings/keys](https://openrouter.ai/settings/keys) adresinden bir API anahtarı üretmek ve API anahtarını güvenli bir yerde saklamaktır (ör. bir parola yöneticisi).


&nbsp;
### 2.2 OpenRouter ile veri üretimi

OpenRouter betiği, API anahtarını bir ortam değişkeni olarak başa eklememiz dışında Ollama betiğine benzer şekilde çalışır:

```bash
OPENROUTER_API_KEY="YOUR_API_KEY" uv run generate_with_openrouter.py \
  --math_json math_train_sample.json \
  --dataset_size 5 \
  --model deepseek/deepseek-r1 \
  --num_processes 1 \
  --out_file sample_openrouter_outputs.json
```

`uv` kullanıcısı değilseniz, `uv run` yerine `python` yazın.

Çıktı şöyle görünür:

```
Loading model: deepseek/deepseek-r1
Using OpenRouter API: https://openrouter.ai/api/v1/chat/completions
Model ready
5/5 | MATH-500: 5/5 | ETA: 00s        
Total time: 2.2 min

Wrote 5 rows to: /Users/sebastian/Developer/reasoning-from-scratch/ch08/02_generate_distillation_data/sample_openrouter_outputs.json
```

[sample_openrouter_outputs.json](sample_openrouter_outputs.json) çıktı dosyası, Ollama betiği tarafından üretilenle aynı yapıya sahiptir.

**İpucu:** Çok fazla veri üretiyorsanız, bu sıralı damıtma sürecini çalıştırmak çok yavaş olabilir (ör. DeepSeek R1 ile 12.000 yanıt için ~100 saat). Bu durumda `--num_processes` ile birden çok paralel veri üretim iş parçacığı çalıştırmanızı öneririm. Örneğin, DeepSeek R1 modelleriyle `--num_processes 50` kullanmak çalışma süresini 100 saatten yaklaşık 2 saate düşürür.


&nbsp;
## Damıtma için veri kümeleri

Yukarıda açıklanan OpenRouter yaklaşımıyla üretilmiş bir veri kümesi koleksiyonunu burada bulabilirsiniz: [https://huggingface.co/datasets/rasbt/math_distill](https://huggingface.co/datasets/rasbt/math_distill).

&nbsp;
## Veri kümesi istatistikleri

Veri kümesi istatistiklerini kontrol etmek için [average_field_lengths_json.py] betiğini kullanın:

```bash
uv run average_field_lengths_json.py \
--json_path sample_openrouter_outputs.json
```

```
tokenizer-reasoning.json: 100% (10 MiB / 10 MiB)
Records: 5
Tokenizer: reasoning
Field             AvgTokens  MinTokens  MaxToken  Count
gtruth_answer          9.40          9        10      5
message_content      196.00        166       259      5
message_thinking     933.20        449      1676      5
problem               77.80         30       121      5
```

&nbsp;
## Öğretmen doğruluğu

Veri kümesini üreten modelin (yani öğretmenin) doğruluğunu hesaplamak için [../../ch03/02_math500-verifier-scripts/evaluate_json.py](../../ch03/02_math500-verifier-scripts/evaluate_json.py) betiğini kullanın:

```bash
uv run ../../ch03/02_math500-verifier-scripts/evaluate_json.py \
--json_path sample_openrouter_outputs.json \
--gtruth_answer gtruth_answer \
--generated_text message_content
```

```
Accuracy: 100.0% (5/5)
```

&nbsp;
## MATH-500 damıtma veri kümesi üretmek

500 örneklik MATH-500 kümesi için öğretmen yanıtları üretmek üzere `--math_json` seçeneğini atlayabilirsiniz; her iki betik de `math500_test.json` dosyasını otomatik olarak yükler (ve ilk kullanımda yerel bir kopyasını kaydeder).

**Ollama**

```bash
uv run generate_with_ollama.py \
  --dataset_size 500 \
  --model deepseek-r1:8b \
  --max_new_tokens 8192 \
  --out_file math500_ollama_distill.json
```

**OpenRouter**

```bash
OPENROUTER_API_KEY="YOUR_API_KEY" uv run generate_with_openrouter.py \
  --dataset_size 500 \
  --model deepseek/deepseek-r1 \
  --num_processes 1 \
  --out_file math500_openrouter_distill.json
```

&nbsp;
## 12.000 MATH örneğinden oluşan bir damıtma veri kümesi üretmek

Bu, 6, 7 ve 8. bölümlerdeki aynı, örtüşmeyen 12.000 örneklik eğitim kümesini kullanır. Henüz elinizde yoksa, önce indirin:

```bash
curl -fL -o math_full_minus_math500.json \
https://raw.githubusercontent.com/rasbt/math_full_minus_math500/refs/heads/main/math_full_minus_math500.json
```

**Ollama**

```bash
uv run generate_with_ollama.py \
  --math_json math_full_minus_math500.json \
  --dataset_size 12000 \
  --model deepseek-r1:8b \
  --max_new_tokens 8192 \
  --resume \
  --out_file math12000_ollama_distill.json
```

**OpenRouter**

```bash
OPENROUTER_API_KEY="YOUR_API_KEY" uv run generate_with_openrouter.py \
  --math_json math_full_minus_math500.json \
  --dataset_size 12000 \
  --model deepseek/deepseek-r1 \
  --num_processes 50 \
  --resume \
  --out_file math12000_openrouter_distill.json
```

Büyük OpenRouter çalıştırmaları için, hesap sınırlarınıza ve istediğiniz verime bağlı olarak `--num_processes` değerini azaltın veya artırın.
