# Bölüm 3: Akıl Yürütme Modellerini Değerlendirmek

> 🇹🇷 **Türkçe çeviri.** Orijinal İngilizce sürüm: [README.md](https://github.com/rasbt/reasoning-from-scratch/blob/main/ch03/02_math500-verifier-scripts/README.md) · Komut, çıktı ve tablo verileri birebir korunmuştur.

&nbsp;


&nbsp;
## Bonus materyaller

- [evaluate_math500.py](evaluate_math500.py): modelleri MATH-500 veri kümesinde değerlendirmek için bağımsız betik
- [evaluate_math500_batched.py](evaluate_math500_batched.py): yukarıdakiyle aynı, ancak üretim sırasında birden çok örneği paralel işler (daha yüksek verim için)
- [evaluate_json.py](evaluate_json.py): kaydedilmiş JSON/JSONL kayıt dosyalarını değerlendirir ve doğruluğu raporlar

Her iki değerlendirme betiği de kod tekrarını önlemek için [`reasoning_from_scratch`](../../reasoning_from_scratch) paketinden işlevsellik içe aktarır. (Kurulum ayrıntıları için bkz. [bölüm 2 kurulum talimatları](../../ch02/02_setup-tips/python-instructions.md).)



<br>

---

**Not**: `uv` kullanıcısı değilseniz, aşağıdaki örneklerde `uv run ...py` yerine `python ...py` yazın.

---



&nbsp;

## `evaluate_math500.py` kullanımı

Şununla çalıştırın:

```bash
python evaluate_math500.py
```

Veya `uv` ile:


```bash
uv run evaluate_math500.py
```

Seçenekler:

```bash
uv run evaluate_math500.py --help

options:
  -h, --help            show this help message and exit
  --device DEVICE       Device to use: "auto" (default) or any torch device string
                        (e.g., "cpu", "cuda", "cuda:0", "mps").
  --which_model {base,reasoning}
                        Model variant to load (default: "base").
  --dataset_size DATASET_SIZE
                        Number of MATH-500 examples to evaluate (default: 10).
  --max_new_tokens MAX_NEW_TOKENS
                        Max new tokens to generate (default: 2048).
  --compile             Enable torch.compile.
  --verbose             Print per-sample correctness while evaluating.
```

&nbsp;
## `evaluate_math500_batch.py` kullanımı

Bu sürüm yığınlamayı (batching) üretimin kendisine kadar genişletir ve paralel kod çözmeyi mümkün kılar:

```bash
uv run evaluate_math500_batched.py --help
```

Ek seçenekler:

```bash
  --batch_size BATCH_SIZE
                        Number of examples to generate in parallel (default: 4).
  --disable_efficient_mode
                        Use a simpler batched inference method. Slower and more
                        memory-intensive, but easier to debug.
```


&nbsp;


**Uygulama notu:**
Varsayılan olarak yığınlanmış üretim, bir durdurma (stop) token'ı üreten diziler için durur. `--disable_efficient_mode` ile tüm diziler, en uzun olanı bitene kadar devam eder. Bu yalnızca hesaplama verimliliğini etkiler, niteliksel sonuçları değil; çünkü durdurma token'ından sonraki token'lar atılır.

&nbsp;

**İpucu (MPS cihazları):**
Şununla çalıştırın:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run evaluate_math500_batched.py
```

Verimli yığınlanmış çıkarımda kullanılan bazı PyTorch işlemleri MPS'te henüz desteklenmiyor. Geri dönüş (fallback) olarak `--disable_efficient_mode` seçeneğini de kullanabilirsiniz.



&nbsp;

- `evaluate_math500.py --dataset_size 500`


| Cihaz / Veri kümesi boyutu                   | Temel model | Akıl yürütme modeli |
| ------------------------------------------- | ---------- | --------------- |
| **Mac Mini M4 CPU** (500 örnek, sıralı) | 43.6 min | Çalıştırılmadı (aşırı ısınma) |
| **Mac Mini M4 GPU** (500 örnek, sıralı) | 37.5 min | Çalıştırılmadı (aşırı ısınma) |
| **DGX Spark** (500 örnek, sıralı) | 10.0 min  | 182.2 min      |
| **H100 GPU** (500 örnek, sıralı) | 13.3 min  | 185.4 min      |

<br>
<br>

- `evaluate_math500_batched.py --dataset_size 500 --batch_size 128`

| Cihaz / Veri kümesi boyutu                                   | Temel model | Akıl yürütme modeli |
| ------------------------------------------------------------ | ---------- | --------------- |
| **Mac Mini M4 CPU** (500 örnek, yığınlanmış, `--batch_size 128`) | 167.2 min | Çalıştırılmadı (aşırı ısınma) |
| **Mac Mini M4 GPU** (500 örnek, yığınlanmış, `--batch_size 128`) | Error*     | Error           |
| **DGX Spark** (500 örnek, yığınlanmış, `--batch_size 128`)    | 16.3 min  | 119.3 min      |
| **H100 GPU** (500 örnek, yığınlanmış, `--batch_size 128`)     | 3.3 min   | 14.6 min       |



- Temel modelin doğruluğu %15,6 (78/500); akıl yürütme modelinin doğruluğu %50,8 (254/500).


&nbsp;
## `evaluate_json.py` kullanımı

Zaten kaydedilmiş kayıtlarınız varsa ve yalnızca doğruluğu (yeniden) hesaplamak istiyorsanız bunu kullanın:

```bash
uv run evaluate_json.py --json_path math500_base-mps-evaluate-script.jsonl
# Accuracy 15.6% (78/500)

İsteğe bağlı anahtarlar:

```bash
uv run evaluate_json.py \
  --json_path my_records.json \
  --gtruth_answer "gtruth_answer" \
  --generated_text "generated_text"
```
