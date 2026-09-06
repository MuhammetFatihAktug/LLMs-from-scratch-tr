# MiniMax Damıtma Sağlayıcısı

> 🇹🇷 **Türkçe çeviri.** Orijinal İngilizce sürüm: [README.md](https://github.com/rasbt/reasoning-from-scratch/blob/main/ch08/02_generate_distillation_data/other_providers/minimax/README.md) · Komutlar birebir korunmuştur.

Bu klasör, 8. bölümdeki damıtma verisi üretimi için MiniMax'e özgü barındırılan (hosted) üretim betiğini içerir.

Aşağıdaki komutları `ch08/02_generate_distillation_data/` dizininden çalıştırın; böylece `math_train_sample.json` gibi göreli yollar yazıldığı gibi çalışmaya devam eder.

Girdi ve çıktı JSON biçimleri, [ana README](../../README.md#input-data-format) dosyasında belgelenenlerle aynıdır.

&nbsp;
## Dosyalar

- [generate_with_minimax.py](generate_with_minimax.py): Damıtma için model yanıtlarını üretmek üzere MiniMax'in bulut API'sini kullanır. MiniMax, OpenAI uyumlu bir API üzerinden MiniMax-M3 (512K bağlam penceresi) gibi modeller sunar.

&nbsp;
## MiniMax kurulumu

1. [MiniMax Platform](https://platform.minimaxi.com/) üzerinde bir hesap oluşturun
2. Hesap ayarlarınızdan bir API anahtarı üretin
3. API anahtarını güvenli bir yerde saklayın (ör. bir parola yöneticisi)

Mevcut modeller:
- `MiniMax-M3` — 512K bağlam penceresi ve 128K maksimum çıktıya sahip en güncel model (varsayılan)
- `MiniMax-M2.7` — 1M bağlam penceresine sahip önceki nesil
- `MiniMax-M2.7-highspeed` — M2.7'nin verim için optimize edilmiş daha hızlı varyantı

&nbsp;
## MiniMax ile veri üretimi

MiniMax betiği, OpenRouter betiğine benzer şekilde çalışır:

```bash
MINIMAX_API_KEY="YOUR_API_KEY" uv run other_providers/minimax/generate_with_minimax.py \
  --math_json math_train_sample.json \
  --dataset_size 5 \
  --model MiniMax-M3 \
  --num_processes 1 \
  --out_file sample_minimax_outputs.json
```

`uv` kullanıcısı değilseniz, `uv run` yerine `python` yazın.

Çıktı dosyası, Ollama ve OpenRouter betikleri tarafından üretilenlerle aynı yapıya sahiptir.

**Not:** MiniMax, temperature parametresinin (0.0, 1.0] aralığında olmasını gerektirir. Betik, bu aralığın dışındaki değerleri otomatik olarak sınırlar (clamp).

&nbsp;
## MATH-500 damıtma veri kümesi üretmek

500 örneklik MATH-500 kümesi için öğretmen (teacher) yanıtları üretmek üzere `--math_json` seçeneğini atlayabilirsiniz; MiniMax betiği `math500_test.json` dosyasını otomatik olarak yükler (ve ilk kullanımda yerel bir kopyasını kaydeder).

```bash
MINIMAX_API_KEY="YOUR_API_KEY" uv run other_providers/minimax/generate_with_minimax.py \
  --dataset_size 500 \
  --model MiniMax-M3 \
  --num_processes 1 \
  --out_file math500_minimax_distill.json
```

&nbsp;
## 12.000 MATH örneğinden oluşan bir damıtma veri kümesi üretmek

Bu, 6, 7 ve 8. bölümlerdeki aynı, örtüşmeyen 12.000 örneklik eğitim kümesini kullanır. Henüz elinizde yoksa, önce indirin:

```bash
curl -fL -o math_full_minus_math500.json \
https://raw.githubusercontent.com/rasbt/math_full_minus_math500/refs/heads/main/math_full_minus_math500.json
```

```bash
MINIMAX_API_KEY="YOUR_API_KEY" uv run other_providers/minimax/generate_with_minimax.py \
  --math_json math_full_minus_math500.json \
  --dataset_size 12000 \
  --model MiniMax-M3 \
  --num_processes 50 \
  --resume \
  --out_file math12000_minimax_distill.json
```

Büyük çalıştırmalar için, hesap sınırlarınıza ve istediğiniz verime bağlı olarak `--num_processes` değerini azaltın veya artırın.
