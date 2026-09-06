# Bölüm 8 Bonus Materyali: Qwen3'ü Hugging Face ile Kullanmak

> 🇹🇷 **Türkçe çeviri.** Orijinal İngilizce sürüm: [README.md](https://github.com/rasbt/reasoning-from-scratch/blob/main/ch08/06_use_via_huggingface/README.md) · Bu klasör, bu depodaki sıfırdan yazılmış [`Qwen3Model`](../../reasoning_from_scratch/qwen3.py) modelini ve uyumlu `.pth` kontrol noktalarını Hugging Face `transformers` ile kullanmanın iki yolunu içerir.

Her iki yaklaşım da Hugging Face tarzı çıkarım ve eğitim yapmanıza olanak tanır. Fark, yeniden kullanılabilir bir Hugging Face model dizini mi yoksa mevcut PyTorch modelinin etrafında daha hafif bir yerel sarmalayıcı mı istediğinizdir.

&nbsp;
## Yaklaşımlar


&nbsp;
### 1) `wrapper_approach`

[./wrapper_approach](./wrapper_approach) yaklaşımı modeli yerel bir `.pth` dosyası olarak tutar ve Hugging Face API'sinin bazı bölümleriyle çalışabilmesi için `Qwen3Model` sınıfını ince bir yerel `PreTrainedModel` içine sarar.

Şunları istiyorsanız bu yaklaşımı kullanın:

- en az miktarda ek kod
- bu depo içinde yerel deneyler
- dışa aktarma (export) adımı olmadan `model.generate(...)` ve `transformers.Trainer`
- temel modeli veya 6-8. bölüm kontrol noktalarını doğrudan `.pth` dosyasından yükleme


&nbsp;
### 2) `export_approach`

[./export_approach](./export_approach) yaklaşımı, sıfırdan yazılmış Qwen3 ağırlıklarını veya uyumlu bir kontrol noktasını Hugging Face uyumlu bir model klasörüne dönüştürür.

Şunları istiyorsanız bu yaklaşımı kullanın:

- `config.json`, tokenizer dosyaları ve ağırlıkları içeren kaydedilmiş bir model dizini
- `AutoConfig`, `AutoTokenizer` ve `AutoModelForCausalLM`
- Hugging Face modellerinin genellikle paketlenme biçimine daha yakın bir iş akışı



&nbsp;
## Hangisi Kullanılmalı?

- Öğrenme amaçlıysanız ve hedefiniz `transformers` ile daha hafif bir yerel entegrasyonsa [wrapper_approach](wrapper_approach) seçeneğini tercih edin.
- Hedefiniz bir Hugging Face model paketi oluşturmak ve hesaplama performansını optimize etmekse [export_approach](export_approach) seçeneğini tercih edin.
