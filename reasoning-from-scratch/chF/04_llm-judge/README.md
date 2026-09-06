
# Hakem Olarak LLM (LLM-as-a-judge)

> 🇹🇷 **Türkçe çeviri.** Orijinal İngilizce sürüm: [README.md](https://github.com/rasbt/reasoning-from-scratch/blob/main/chF/04_llm-judge/README.md) · Komut ve çıktı blokları birebir korunmuştur.

Bu bonus materyal, gpt-oss:20b modelinin (açık kaynaklı Ollama kütüphanesi aracılığıyla) Qwen3 0.6B temel ve akıl yürütme varyantlarını MATH-500 üzerinde değerlendirdiği bir "hakem olarak LLM" yaklaşımını uygular.

<img src="https://sebastianraschka.com/images/reasoning-from-scratch-images/appendix-f/Appendix_F_F06_raschka.webp" width="500px">





- Ollama, LLM'leri verimli şekilde çalıştırmaya yarayan açık kaynaklı bir uygulamadır
- Verimliliği en üst düzeye çıkarmak için LLM'leri saf C/C++ ile uygulayan llama.cpp ([https://github.com/ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp)) etrafında bir sarmalayıcıdır
- Bunun, LLM'leri eğitmek veya ince ayar yapmak için değil, metin üretmek (çıkarım) için kullanılan bir araç olduğunu unutmayın
- Aşağıdaki kodu çalıştırmadan önce [https://ollama.com](https://ollama.com) adresini ziyaret edip talimatları izleyerek ollama'yı kurun (örneğin "Download" düğmesine tıklayıp işletim sisteminize uygun ollama uygulamasını indirin)
- macOS ve Windows kullanıcıları indirdikleri ollama uygulamasına tıklasın; komut satırı kullanımını kurmanızı isterse "evet" deyin
- Linux kullanıcıları ollama web sitesinde verilen kurulum komutunu kullanabilir
- Ollama'yı bilgisayarımızda çalıştırmanın 3 yolu vardır:



**1. `ollama serve`**

- Bu, ollama arka ucunu genellikle `http://localhost:11434` adresinde bir sunucu olarak çalıştırır. API üzerinden çağırana kadar bir model yüklemez. Ollama'yı Python üzerinden kullanmak istiyorsak istediğimiz budur.

**2. `ollama run gpt-oss:20b`**

- Bu pratik bir sarmalayıcıdır. Sunucu zaten çalışmıyorsa başlatır, ardından modeli (ilk seferde) indirir ve bizi modelle sohbet edebileceğimiz etkileşimli bir terminale bırakır. Arka planda aynı sunucu API'sini kullanır.

**3. Ollama masaüstü uygulaması**

- Bu, aynı arka ucu otomatik olarak çalıştırır ve üzerine bir grafik arayüz (GUI) sunar (yukarıdaki şekilde gösterildiği gibi).
Ayrıca varsayılan ayarlar (sistem istemi, temperature, durdurma dizileri) uygular; bu da yanıtların ham API kullanımından neden farklı göründüğünü açıklayabilir.



## Kullanım



Seçenekler ve varsayılan değerler aşağıda gösterilmiştir.

<br>

---

**Not**: `uv` kullanıcısı değilseniz, aşağıdaki örneklerde `uv run ...py` yerine `python ...py` yazın.

---



```bash
uv run ollama-judge.py --help
usage: ollama-judge.py [-h] [--device DEVICE]
                       [--which_model {base,reasoning}]
                       [--dataset_size DATASET_SIZE]
                       [--max_new_tokens MAX_NEW_TOKENS]
                       [--url URL]
                       [--judge_model JUDGE_MODEL]

options:
  -h, --help            show this help message and
                        exit
  --device DEVICE       Device e.g., "cpu",
                        "cuda", "cuda:0", "mps".
  --which_model {base,reasoning}
                        Candidate variant to use.
                        Defaults to "base".
  --dataset_size DATASET_SIZE
                        Number of MATH-500
                        examples to evaluate.
                        Default: 10
  --max_new_tokens MAX_NEW_TOKENS
                        Max new tokens for
                        candidate generation.
                        Default: 2048
  --url URL             Ollama chat endpoint for
                        the judge. Default: "http:
                        //localhost:11434/api/chat
                        "
  --judge_model JUDGE_MODEL
                        Judge model name (Ollama).
                        Used only for scoring.
                        Default: "gpt-oss:20b"
```



**Temel model**

```bash
➜  uv run ollama-judge.py
Using Apple Silicon GPU (MPS)
Model: base
Device: mps
✓ qwen3/qwen3-0.6B-base.pth already up-to-date
✓ qwen3/tokenizer-base.json already up-to-date
Ollama running: True
[1/10] score=5
[2/10] score=1
[3/10] score=5
[4/10] score=5
[5/10] score=3
[6/10] score=5
[7/10] score=5
[8/10] score=3
[9/10] score=5
[10/10] score=1

Summary
-------
Average score: 3.800 over 10 example(s)
Counts: 1:2 2:0 3:2 4:0 5:6
```

**Akıl yürütme modeli**

```bash
➜  uv run ollama-judge.py --which_model reasoning
Using Apple Silicon GPU (MPS)
Model: reasoning
Device: mps
✓ qwen3/qwen3-0.6B-reasoning.pth already up-to-date
✓ qwen3/tokenizer-reasoning.json already up-to-date
Ollama running: True
[1/10] score=5
[2/10] score=5
[3/10] score=5
[4/10] score=5
[5/10] score=4
[6/10] score=5
[7/10] score=5
[8/10] score=1
[9/10] score=5
[10/10] score=3

Summary
-------
Average score: 4.300 over 10 example(s)
Counts: 1:1 2:0 3:1 4:1 5:7
```
