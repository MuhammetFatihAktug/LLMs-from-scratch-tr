# Bölüm 8 Bonus Materyali: Qwen3'ü Yerel Bir Hugging Face Sarmalayıcısıyla Kullanmak

> 🇹🇷 **Türkçe çeviri.** Orijinal İngilizce sürüm: [README.md](https://github.com/rasbt/reasoning-from-scratch/blob/main/ch08/06_use_via_huggingface/wrapper_approach/README.md) · Komutlar birebir korunmuştur.

Bu klasör, sıfırdan yazılmış [`Qwen3Model`](../../../reasoning_from_scratch/qwen3.py) modelinin, uyumluluk için ince bir yerel `PreTrainedModel` sınıfı içine sarılarak Hugging Face `transformers` kütüphanesiyle nasıl kullanılacağını gösterir.

Bu, şunları kullanmanızı sağlar:

- `model.generate(...)`
- `transformers.Trainer`

ve bunları doğrudan bu depodaki yerel `.pth` model dosyalarıyla (temel Qwen3 ağırlıkları ve 6-8. bölümlerdeki uyumlu kontrol noktaları dahil) çalıştırabilirsiniz.

&nbsp;
## Dosyalar

- [hf_wrapper.py](hf_wrapper.py): kitap boyunca kullandığımız sıfırdan yazılmış `Qwen3Model` etrafındaki yerel `PreTrainedModel` sarmalayıcısı
- [hf_inference.py](hf_inference.py): sarmalayıcıyı ve deponun tokenizer'ını kullanarak metin üretimi
- [hf_trainer.py](hf_trainer.py): sarmalayıcıyı ve 8. bölümün damıtma JSON biçimini kullanan `Trainer` örneği

---

**Not**: `uv` kullanıcısı değilseniz, aşağıdaki örneklerde `uv run ...py` yerine `python ...py` yazın.

---

&nbsp;
## Bu Sarmalayıcı Ne Yapar

Sarmalayıcı, modeli bu depoya yerel tutar ve Hugging Face API'sine uyarlar.

Somut olarak şunları yapar:

- yerel bir `.pth` model dosyasını doğrudan `Qwen3Model` içine yükler
- bu modeli bir `PreTrainedModel` arayüzüne sarar
- `Trainer` ile uyumlu bir `forward(...)` metodu sunar
- `model.generate(...)` kullanımını mümkün kılar
- deponun `Qwen3Tokenizer` sınıfını kullanmaya devam eder

Neden? Bazı okurlar, bu depodaki sıfırdan yazılmış koda göre çok daha fazla özelliğe sahip olan `transformers` içinde modelleri daha ileri düzeyde incelemeyi merak ediyordu.

&nbsp;
## Kısıtlar

Bu, sıfırdan yazılmış modelin etrafında küçük ve yerel bir sarmalayıcıdır.

Önemli sonuçları:

- `reasoning_from_scratch` paketinin kurulu olduğu ortamlar için tasarlanmıştır
- bir `AutoTokenizer.from_pretrained(...)` iş akışı sunmaz
- `config.json` ve tokenizer dosyalarını içeren yeniden kullanılabilir bir model dizini oluşturmaz
- üretim bilinçli olarak basit tutulmuştur; bu nedenle sıfırdan yazılmış KV önbelleğini Hugging Face önbellek sınıflarına uyarlamak yerine tüm ön eki (prefix) yeniden hesaplar; tam destek istiyorsanız [../export_approach](../export_approach) yaklaşımına geçmeniz gerekir

Bu kısıtların kodu kısa tuttuğunu ve bu depo içindeki yerel kullanıma odakladığını unutmayın.

&nbsp;
## Adım 1: Bağımlılıkları kurun

Bu rehber, depo bağımlılıklarına ek olarak Hugging Face Transformers kullanır.

```bash
pip install transformers accelerate
```

Veya `uv` kullanıyorsanız:

```bash
uv add --dev transformers accelerate
```

&nbsp;
## Adım 2: Yerel sarmalanmış çıkarımı çalıştırın

Temel modeli sarmalayıcı üzerinden çalıştırmak için şunu kullanın:

```bash
  uv run hf_inference.py \
    --tokenizer_kind base \
    --prompt "If x + 7 = 19, what is x?"
```

Akıl yürütme varyantını çalıştırmak için şunu kullanın:

```bash
  uv run hf_inference.py \
    --tokenizer_kind reasoning \
    --prompt "If x + 7 = 19, what is x?"
```

Bunun yerine yerel bir kontrol noktası çalıştırmak için:

```bash
uv run hf_inference.py \
  --tokenizer_kind reasoning \
  --model_path ../../04_train_with_distillation/checkpoints/distill/qwen3-0.6B-distill-step00004-epoch1.pth \
  --prompt "If x + 7 = 19, what is x?"
```

`--model_path` belirtilmezse, betik seçilen `--tokenizer_kind` değeri için varsayılan temel veya akıl yürütme modelini indirir. `--model_path` verilirse, temel Qwen3 `.pth` dosyasını ya da 6-8. bölümlerde üretilen herhangi bir uyumlu kontrol noktasını gösterebilir.

Dahili olarak çıkarım betiği şunları yapar:

1. yerel sarmalayıcı modelini oluşturur
2. seçilen `.pth` model dosyasını sarmalanmış `Qwen3Model` içine yükler
3. istemi deponun tokenizer'ıyla token'lara ayırır
4. `model.generate(...)` çağrısını yapar

&nbsp;
## Adım 3: `Trainer` ile eğitime devam edin

Aynı sarmalayıcı `transformers.Trainer` ile de kullanılabilir:

```bash
uv run hf_trainer.py \
  --tokenizer_kind reasoning \
  --model_path ../../04_train_with_distillation/checkpoints/distill/qwen3-0.6B-distill-step00004-epoch1.pth \
  --data_path ../../02_generate_distillation_data/sample_openrouter_outputs.json \
  --dataset_size 5 \
  --validation_size 1 \
  --epochs 1 \
  --logging_steps 1
```

Çıkarımda olduğu gibi, `--model_path` temel Qwen3 ağırlıklarını veya uyumlu bir 6-8. bölüm kontrol noktasını gösterebilir.

Eğitici (trainer), 8. bölümün başka yerlerinde kullanılan yalnızca-yanıt (answer-only) hedefini korur:

- istem token'ları maskelenir
- kayba yalnızca yanıt token'ları katkıda bulunur
- akıl yürütme modu, öğretmen izlerini `<think>...</think>` içine sarar

Girdi JSON biçimi, [../../02_generate_distillation_data](../../02_generate_distillation_data) klasöründe üretilen damıtma verisiyle eşleşir.

&nbsp;
