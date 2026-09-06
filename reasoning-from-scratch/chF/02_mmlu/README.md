
# MMLU Kıyaslaması

> 🇹🇷 **Türkçe çeviri.** Orijinal İngilizce sürüm: [README.md](https://github.com/rasbt/reasoning-from-scratch/blob/main/chF/02_mmlu/README.md) · Komut, çıktı ve matematiksel formül blokları birebir korunmuştur.

Bu bonus materyal, modelleri MMLU üzerinde değerlendirmek için üç farklı yöntem uygular.
- Yöntem 1, sezgisel bir giriş olarak tasarlanmıştır
- Yöntem 2, pratikte en yaygın kullanılan yöntemdir
- Yöntem 3, akıl yürütme modelleri için daha uygun olan daha sağlam bir yöntemdir

- Kodun [MMLU veri kümesini](https://huggingface.co/datasets/cais/mmlu) Hugging Face model merkezinden yüklediğini unutmayın. Bu nedenle kodu çalıştırmadan önce `datasets` Python kütüphanesini kurmanız gerekir:

```python
pip install datasets
```

veya

```python
uv add datasets
```

- Aşağıdaki bölümlerde MMLU değerlendirme yöntemlerini (`"high_school_mathematics"`) alt kümesine uyguluyoruz

- Başka pek çok ilginç alt küme olduğunu unutmayın; bu, basitlik ve verimlilik için seçilmiştir; örneğin şunları kullanabilirsiniz:

  - Diğer mevcut alt kümeleri listelemek için `--subsets list` kullanın

  - Birden çok alt küme seçmek için örneğin `--subsets "astronomy,high_school_mathematics"` kullanın

  - Tüm alt kümeler üzerinde değerlendirmek için `--subsets "all"` kullanın

(Basitlik ve kod okunabilirliği adına, 5 örnekli (5-shot) yerine sıfır örnekli (zero-shot) bir kuruluma odaklandığımızı unutmayın.)

<br>

---

**Not**: `uv` kullanıcısı değilseniz, aşağıdaki örneklerde `uv run ...py` yerine `python ...py` yazın.

---

&nbsp;

## Yöntem 1: MMLU harf eşleştirme

- Modelin yanıtı üretmesine izin veriyoruz
- Üretilen ilk A/B/C/D harfini çıkarıp doğru yanıtla karşılaştırıyoruz
- Bu en sezgisel yöntemdir, ancak dezavantajı modelin A/B/C/D harfiyle yanıt vermeyebilmesidir

<br>

<img src="https://sebastianraschka.com/images/reasoning-from-scratch-images/bonus/mmlu/method_1.webp" width=700>

<br>

```bash
➜  02_mmlu git:(main) ✗ uv run 1_letter_matching.py --which_model base     
Using Apple Silicon GPU (MPS)
Using device: mps
✓ qwen3/qwen3-0.6B-base.pth already up-to-date
✓ qwen3/tokenizer-base.json already up-to-date
MMLU 50 acc=0.240 [high_school_mathematics]
MMLU 100 acc=0.200 [high_school_mathematics]
MMLU 150 acc=0.193 [high_school_mathematics]
MMLU 200 acc=0.235 [high_school_mathematics]
MMLU 250 acc=0.224 [high_school_mathematics]

MMLU letter accuracy: 58/270 = 21.48% in 69.1s
{'accuracy': 0.21481481481481482, 'num_examples': 270, 'subsets': ['high_school_mathematics'], 'split': 'test'}
```

```bash
➜  02_mmlu git:(main) ✗ uv run 1_letter_matching.py --which_model reasoning
Using Apple Silicon GPU (MPS)
Using device: mps
qwen3-0.6B-reasoning.pth: 100% (1433 MiB / 1433 MiB)
tokenizer-reasoning.json: 100% (10 MiB / 10 MiB)
MMLU 50 acc=0.220 [high_school_mathematics]
MMLU 100 acc=0.230 [high_school_mathematics]
MMLU 150 acc=0.220 [high_school_mathematics]
MMLU 200 acc=0.210 [high_school_mathematics]
MMLU 250 acc=0.216 [high_school_mathematics]

MMLU letter accuracy: 57/270 = 21.11% in 43.6s
{'accuracy': 0.2111111111111111, 'num_examples': 270, 'subsets': ['high_school_mathematics'], 'split': 'test'}
```



&nbsp;

## Yöntem 2: Log-olasılık (log-probability) puanlaması

- İstemi modelden geçirip bir sonraki token için log-olasılıkları (log-prob) alıyoruz (log-prob tartışması için 4. bölüme bakın)
- Ardından her harf seçeneği için, o harfi eklersek hangi token kimliğinin ilk görüneceğini hesaplıyoruz
- Sonra bu dört log-olasılığı karşılaştırıp en yükseğini (max) seçiyoruz

<br>

<img src="https://sebastianraschka.com/images/reasoning-from-scratch-images/bonus/mmlu/method_2.webp" width=700>

<br>

```bash
➜  02_mmlu git:(main) ✗ uv run 2_logprob.py --which_model base 
Using Apple Silicon GPU (MPS)
Using device: mps
✓ qwen3/qwen3-0.6B-base.pth already up-to-date
✓ qwen3/tokenizer-base.json already up-to-date
MMLU 50 acc=0.360 [high_school_mathematics]
MMLU 100 acc=0.420 [high_school_mathematics]
MMLU 150 acc=0.400 [high_school_mathematics]
MMLU 200 acc=0.370 [high_school_mathematics]
MMLU 250 acc=0.344 [high_school_mathematics]

MMLU letter accuracy (log-prob): 93/270 = 34.44% in 22.5s
{'accuracy': 0.34444444444444444, 'num_examples': 270, 'subsets': ['high_school_mathematics'], 'split': 'test'}
```

```bash
➜  02_mmlu git:(main) ✗ uv run 2_logprob.py --which_model reasoning
Using Apple Silicon GPU (MPS)
Using device: mps
✓ qwen3/qwen3-0.6B-reasoning.pth already up-to-date
✓ qwen3/tokenizer-reasoning.json already up-to-date
MMLU 50 acc=0.220 [high_school_mathematics]
MMLU 100 acc=0.230 [high_school_mathematics]
MMLU 150 acc=0.220 [high_school_mathematics]
MMLU 200 acc=0.210 [high_school_mathematics]
MMLU 250 acc=0.216 [high_school_mathematics]

MMLU letter accuracy (log-prob): 57/270 = 21.11% in 22.4s
{'accuracy': 0.2111111111111111, 'num_examples': 270, 'subsets': ['high_school_mathematics'], 'split': 'test'}
```



&nbsp;

## Yöntem 3: Öğretmen zorlaması (teacher forcing)

- A/B/C/D harflerinin her birinin log-olasılığına bakmak yerine, (özellikle akıl yürütme modelleri için) daha sağlam bir puanlama yöntemi, harfi tam yanıt dizesiyle birlikte modele vermektir
- Örneğimizde yanıt dizeleri "A. 7", "B. 11", "C. 16", "D. 8" şeklindedir
- Bu yöntem, talihsiz bir terim olan "teacher forcing" (öğretmen zorlaması) adıyla bilinir
- Bu yöntem en güvenilir olanıdır, ancak dezavantajı yöntem 2'deki log-olasılık yaklaşımından 4 kat daha uzun sürmesidir (çünkü modele 4 yanıt varyantının tümünü veriyoruz)

<br>

<img src="https://sebastianraschka.com/images/reasoning-from-scratch-images/bonus/mmlu/method_3.webp" width=700>

<br>

```bash
➜  02_mmlu git:(main) ✗ uv run 3_teacher_forcing.py --which_model base 
Using Apple Silicon GPU (MPS)
Using device: mps
✓ qwen3/qwen3-0.6B-base.pth already up-to-date
✓ qwen3/tokenizer-base.json already up-to-date
MMLU 50 acc=0.360 [high_school_mathematics]
MMLU 100 acc=0.310 [high_school_mathematics]
MMLU 150 acc=0.307 [high_school_mathematics]
MMLU 200 acc=0.315 [high_school_mathematics]
MMLU 250 acc=0.312 [high_school_mathematics]

MMLU letter accuracy (teacher-forced): 86/270 = 31.85% in 67.9s
{'accuracy': 0.31851851851851853, 'num_examples': 270, 'subsets': ['high_school_mathematics'], 'split': 'test'}
```

```bash
➜  02_mmlu git:(main) ✗ uv run 3_teacher_forcing.py --which_model reasoning
Using Apple Silicon GPU (MPS)
Using device: mps
✓ qwen3/qwen3-0.6B-reasoning.pth already up-to-date
✓ qwen3/tokenizer-reasoning.json already up-to-date
MMLU 50 acc=0.240 [high_school_mathematics]
MMLU 100 acc=0.250 [high_school_mathematics]
MMLU 150 acc=0.267 [high_school_mathematics]
MMLU 200 acc=0.255 [high_school_mathematics]
MMLU 250 acc=0.280 [high_school_mathematics]

MMLU letter accuracy (teacher-forced): 78/270 = 28.89% in 68.8s
{'accuracy': 0.28888888888888886, 'num_examples': 270, 'subsets': ['high_school_mathematics'], 'split': 'test'}
```



## Rastgele tahmin temel çizgisi

- Bu rastgele tahmin temel çizgisi, yalnızca yukarıdaki sayıları bir perspektife oturtmak içindir

- Tüm yanıtlar arasında düzgün (eşit) olasılıkla rastgele tahmin yapan bir modelin $25\%$ doğruluk elde etmesi beklenir

- Ancak rastgele tahmin eden biri için $25\%$ değerinden sapmalar bekleyebiliriz (örneklem büyüklüğüne bağlı olarak)

- Örneğin, bir değerlendirme koşusunu $n$ sorudan $K$ tanesi doğru olacak şekilde binom dağılımıyla modelleyebiliriz:

  - $p=\tfrac14$ ve $n=$ soru sayısı olmak üzere $K \sim \mathrm{Binomial}(n,p)$.
  - Doğruluk $A = K/n$.

- Bunu $n=270$ olan *high_school_mathematics* alt kümesi için adım adım ele alalım

- Genel olarak binom dağılımının özellikleri şunlardır:

  - Ortalama: $\mathbb{E}[K] = np$
  - Standart sapma: $\sigma_K = \sqrt{np(1-p)}$

- $A=K/n$ doğruluğu için:

  - Ortalama: $\mathbb{E}[A] = p = 0.25$
  - Standart sapma: $\sigma_A = \sqrt{\tfrac{p(1-p)}{n}}$

- $n=270$ yerine konduğunda:

  - $\mathbb{E}[A] = 25\%$
  - $\sigma_A = \sqrt{\tfrac{0.25\cdot 0.75}{270}} \approx 2.64\%$

- Bir standart sapmalık ($\pm 1\sigma$) doğruluk sınırlarını sayılara çevirelim:

  - Alt sınır: $K \le \lfloor 270\,(0.25-0.02636)\rfloor = 60$
  - Üst sınır: $K \ge \lceil 270\,(0.25+0.02636)\rceil = 75$
  - (Bandın içi $K=61,\dots,74$; eşdeğer olarak $A\in[22.36\%,\,27.64\%]$)

- Dolayısıyla bu sınırın dışına düşme olasılığı şudur:

  $$
  z = \pm\,\frac{75-67.5}{\sqrt{270\cdot 0.25\cdot 0.75}} \approx \pm 1.054, \qquad
  \Pr(|A-0.25|>0.02636) \approx 2\bigl(1-\Phi(1.054)\bigr) \approx 0.292.
  $$

  Yani rastgele tahmin koşularının yaklaşık %29,2'si %22,36'nın altında veya %27,64'ün üzerindedir

- Bu, modelin (düzgün dağılım varsayımıyla) rastgele tahmin ettiği durumların yaklaşık $29.2\%$ kadarında $22.36\%$ altında veya $27.64\%$ üzerinde bir doğruluk elde ettiğimiz anlamına gelir
- Aşağıda deneysel bir bakış yer alıyor:


```bash
➜  02_mmlu git:(main) ✗ uv run 0_random_guessing_baseline.py --subset "high_school_mathematics"
Subset: high_school_mathematics | split: test | n=270
Gold distribution provided in the dataset:
  A: 57 (21.11%)
  B: 71 (26.30%)
  C: 71 (26.30%)
  D: 71 (26.30%)

Random guessing over 10,000 trials (uniform A/B/C/D, seed=42):
  Mean accuracy: 24.98%
  Std dev across trials: 2.65%

Selected quantiles of accuracy:
  1% quantile: 18.889%
  5% quantile: 20.741%
  25% quantile: 23.333%
  50% quantile: 24.815%
  75% quantile: 26.667%
  95% quantile: 29.259%
  99% quantile: 31.111%

Full frequency table of accuracies (rounded):
  0.160: 1 times (0.01%)
  0.170: 11 times (0.11%)
  0.180: 38 times (0.38%)
  0.190: 124 times (1.24%)
  0.200: 302 times (3.02%)
  0.210: 562 times (5.62%)
  0.220: 612 times (6.12%)
  0.230: 1254 times (12.54%)
  0.240: 1619 times (16.19%)
  0.250: 1096 times (10.96%)
  0.260: 1525 times (15.25%)
  0.270: 1248 times (12.48%)
  0.280: 572 times (5.72%)
  0.290: 565 times (5.65%)
  0.300: 281 times (2.81%)
  0.310: 132 times (1.32%)
  0.320: 28 times (0.28%)
  0.330: 24 times (0.24%)
  0.340: 5 times (0.05%)
  0.360: 1 times (0.01%)
```
