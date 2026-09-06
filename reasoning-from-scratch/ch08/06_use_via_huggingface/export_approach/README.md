# Bölüm 8 Bonus Materyali: Sıfırdan Qwen3 Kodunu Hugging Face Transformers ile Kullanmak

> 🇹🇷 **Türkçe çeviri.** Orijinal İngilizce sürüm: [README.md](https://github.com/rasbt/reasoning-from-scratch/blob/main/ch08/06_use_via_huggingface/export_approach/README.md) · Kod ve komut blokları birebir korunmuştur.

Bu klasör, sıfırdan yazılmış [`Qwen3Model`](../../../reasoning_from_scratch/qwen3.py) modelinin ve 6-8. bölümlerde oluşturulan herhangi bir uyumlu `.pth` kontrol noktasının Hugging Face Transformers uyumlu bir klasöre nasıl dönüştürüleceğini ve bunun Hugging Face çıkarım fonksiyonları ile `Trainer` kullanılarak nasıl çalıştırılacağını gösterir.

Dışa aktarma (export), özel bir `transformers` mimarisi olarak uygulanmıştır; bu nedenle `AutoConfig`, `AutoTokenizer`, `AutoModelForCausalLM`, `model.generate(...)` ve `Trainer` gibi standart Hugging Face API'leriyle çalışır. Ancak özel kod olduğu için `trust_remote_code=True` ile yükleyin.

&nbsp;
## Dosyalar

- [hf_export.py](hf_export.py): sıfırdan yazılmış Qwen3 ağırlıklarını veya kaydedilmiş bir `.pth` kontrol noktasını bir Hugging Face model klasörüne dönüştürür
- [hf_inference.py](hf_inference.py): `AutoModelForCausalLM` ile metin üretimi çalıştırır
- [hf_trainer.py](hf_trainer.py): dışa aktarılmış bir modelin eğitimine, 8. bölümün damıtma JSON biçimi üzerinde `transformers.Trainer` ile devam eder
- [hf_qwen3.py](hf_qwen3.py): dışa aktarılan Qwen3 mimarisi için özel Hugging Face `PretrainedConfig` ve `PreTrainedModel` uygulaması

Dışa aktarma betikleri, Hugging Face'e özgü model kodunu bu klasörde yerel tutar ve 3. bölüm istem şablonu, RoPE yardımcıları ile Qwen3 indirme fonksiyonları için [`reasoning_from_scratch`](../../../reasoning_from_scratch) paketinden paylaşılan yardımcıları içe aktarır. (Kurulum ayrıntıları için bkz. [bölüm 2 kurulum talimatları](../../../ch02/02_setup-tips/python-instructions.md).)

---

**Not**: `uv` kullanıcısı değilseniz, aşağıdaki örneklerde `uv run ...py` yerine `python ...py` yazın.

---

&nbsp;
## Adım 1: Bağımlılıkları kurun

Bu rehber, depo bağımlılıklarına ek olarak Hugging Face Transformers kullanır. `transformers.Trainer` için ayrıca `accelerate` paketine de ihtiyacınız var.

```bash
pip install transformers accelerate
```

Veya `uv` kullanıyorsanız:

```bash
uv add --dev transformers accelerate
```

&nbsp;
## Adım 2: Standart (vanilla) Qwen3 modelini dışa aktarın

Orijinal temel modeli bir Hugging Face klasörü olarak dışa aktarmak için şunu çalıştırın:

```bash
uv run hf_export.py \
  --output_dir hf-qwen3-base \
  --tokenizer_kind "base"  # or use "reasoning"
```

Ham `.pth` model ve tokenizer dosyaları zaten yerelinizde varsa, indirmeden kaçınabilirsiniz:

```bash
uv run hf_export.py \
  --output_dir hf-qwen3-base \
  --tokenizer_kind base \
  --model_path ../../../ch02/01_main-chapter-code/qwen3/qwen3-0.6B-base.pth \
  --tokenizer_path ../../../ch02/01_main-chapter-code/qwen3/tokenizer-base.json
```

Aynısı 6-8. bölümlerin kontrol noktası `.pth` dosyalarıyla da çalışır.

Dışa aktarılan klasör şunları içerecektir:

- `config.json`
- `generation_config.json`
- tokenizer dosyaları
- model ağırlıkları (varsayılan olarak `model.safetensors`)
- `trust_remote_code=True` tarafından gereken, kopyalanmış özel bir Python modülü

&nbsp;
### Dışa aktarma kodu ne yapar

Dışa aktarıcı, modeli resmî Hugging Face Qwen uygulamasına çevirmez ve öğrenilmiş ağırlıkları değiştirmez. Yaptığı şudur:

1. sıfırdan yazılmış `Qwen3Model` mimarisini yeniden üreten özel bir Hugging Face `PretrainedConfig` ve `PreTrainedModel` oluşturur
2. orijinal `.pth` `state_dict` dosyasını, eğitilebilir parametreleri yeniden adlandırmadan veya yeniden şekillendirmeden doğrudan bu özel Hugging Face modeline yükler
3. sonucu standart Hugging Face klasör biçiminde kaydeder; böylece `AutoConfig`, `AutoTokenizer`, `AutoModelForCausalLM`, `generate(...)` ve `Trainer` bunu yükleyebilir

Eklenen veya sarmalanan başlıca şeyler şunlardır:

- bir Hugging Face yapılandırma dosyası (`config.json`)
- `transformers` ile uyumlu bir `forward(...)` imzasına sahip bir Hugging Face model sınıfı
- Hugging Face tokenizer dosyaları
- Hugging Face üretim meta verileri (`generation_config.json`)
- `trust_remote_code=True` tarafından yüklenen özel bir Python kaynak dosyası

Dışa aktarma sırasında küçük bir ek ayrıntı daha var. Şöyle ki, sıfırdan yazılmış kontrol noktaları yalnızca eğitilebilir ağırlıkları kaydederken, Hugging Face dışa aktarımı önceden hesaplanmış RoPE `cos` ve `sin` buffer'larını da paketler; böylece dışa aktarılan modelin yeniden yüklenmesi sayısal olarak tutarlı olur.

`--tokenizer_kind reasoning` için dışa aktarıcı, akıl yürütme sohbet şablonunu da tokenizer'a ekler; böylece çıkarım betikleri istemleri beklenen sohbet biçimine otomatik olarak sarabilir.

Bu özel Hugging Face modülü, kurulu [`reasoning_from_scratch`](../../../reasoning_from_scratch) paketini içe aktardığı için, dışa aktarılan klasör bu paket Python ortamında kurulu olduğu sürece uyumludur.

&nbsp;
## Adım 3: Kaydedilmiş bir kontrol noktasını dışa aktarın

Aynı dışa aktarıcı, 8. bölümün damıtma kontrol noktaları veya bu depo tarafından üretilen başka herhangi bir uyumlu `.pth` dosyası için de çalışır.

Örneğin, akıl yürütme tokenizer'ıyla bir 8. bölüm kontrol noktası eğittiyseniz:

```bash
uv run hf_export.py \
  --output_dir hf-qwen3-distill \
  --model_path ../../04_train_with_distillation/checkpoints/distill/qwen3-0.6B-distill-step00004-epoch1.pth \
  --tokenizer_kind reasoning
```

Önemli notlar:

- 8. bölüm damıtma kontrol noktaları ve akıl yürütme tokenizer'ıyla eğitilmiş diğer kontrol noktaları için `--tokenizer_kind reasoning` kullanın.
- Temel tokenizer ile eğitilmiş kontrol noktaları için `--tokenizer_kind base` kullanın.
- Eşleşen tokenizer JSON dosyası zaten diskinizde varsa, indirmeden kaçınmak için `--tokenizer_path` ile verebilirsiniz.

&nbsp;
## Adım 4: Hugging Face çıkarımını çalıştırın

Dışa aktarımdan sonra, `AutoTokenizer` ve `AutoModelForCausalLM` ile çıkarım çalıştırın:

```bash
uv run hf_inference.py \
  --model_dir hf-qwen3-base \
  --prompt "If x + 7 = 19, what is x?"
```

Dahili olarak betik şunları yapar:

1. dışa aktarılan modeli `trust_remote_code=True` ile yükler
2. istemi 3. bölümdeki aynı matematik istem şablonuyla biçimlendirir
3. dışa aktarılan model akıl yürütme tokenizer'ını kullandığında akıl yürütme sohbet sarmalayıcısını otomatik olarak uygular
4. `model.generate(...)` çağrısını yapar

Doğrudan ham Hugging Face API'sini tercih ediyorsanız, eşdeğer kalıp şudur:

```python
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("hf-qwen3-base")
config = AutoConfig.from_pretrained("hf-qwen3-base", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "hf-qwen3-base",
    trust_remote_code=True,
)
```

&nbsp;
## Adım 5: `Trainer` ile eğitime devam edin

Dışa aktarılmış bir kontrol noktasının eğitimine, [`../../04_train_with_distillation`](../../04_train_with_distillation) klasöründe kullanılan aynı JSON biçimi üzerinde Hugging Face `Trainer` ile devam edebilirsiniz.

Örnek:

```bash
uv run hf_trainer.py \
  --model_dir hf-qwen3-base \
  --data_path ../../02_generate_distillation_data/sample_openrouter_outputs.json \
  --dataset_size 5 \
  --validation_size 1 \
  --epochs 1 \
  --logging_steps 1 \
  --save_steps 10
```

Betik, sıfırdan yazılmış damıtma kodunda kullanılan aynı yalnızca-yanıt (answer-only) eğitim hedefini korur:

- istem token'ları kaybın dışında maskelenir
- kayba yalnızca damıtılmış yanıt token'ları katkıda bulunur
- akıl yürütme dışa aktarımlarında betik, öğretmen izlerini nihai yanıttan önce `<think>...</think>` içine sarar



&nbsp;
## Dışa aktarımı başka bir yerde yüklemek

Dışa aktardıktan sonra klasörü başka bir makineye kopyalayabilir veya Hugging Face Hub'a yükleyip orada da yükleyebilirsiniz; yeter ki o ortamda `reasoning_from_scratch` kurulu olsun:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "path-or-hub-repo",
    trust_remote_code=True,
)
model = AutoModelForCausalLM.from_pretrained(
    "path-or-hub-repo",
    trust_remote_code=True,
)
```
