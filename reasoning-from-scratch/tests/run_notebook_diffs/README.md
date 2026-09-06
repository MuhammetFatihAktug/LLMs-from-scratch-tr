# Not defteri çıktı karşılaştırması

> 🇹🇷 **Türkçe çeviri.** Orijinal İngilizce sürüm: [README.md](https://github.com/rasbt/reasoning-from-scratch/blob/main/tests/run_notebook_diffs/README.md) · Komutlar birebir korunmuştur.

Bunlar, not defterlerini iki farklı PyTorch sürümüyle çalıştıran ve olası tutarsızlıkları incelemek için değişen hücre kaynaklarının ve çıktılarının Markdown raporunu oluşturan kolaylık betikleridir. Esasen elle yapılan uyumluluk kontrolleri içindir ve pytest ya da GitHub CI tarafından çalıştırılmaz.

&nbsp;
## Not defterlerini iki PyTorch sürümüyle çalıştırmak

Depo kökünden şunu çalıştırın:

```bash
uv run python tests/run_notebook_diffs/run.py \
  --torch-version 2.7.1 \
  --torch-version 2.13.0 \
  ch02/01_main-chapter-code/ch02_main.ipynb
```

`uv`, her sürüm için temel bağımlılıkların kurulu olduğu yalıtılmış birer ortam oluşturur. 

Ek not defteri bağımlılıkları için `--with` kullanılabilir:

```bash
uv run python tests/run_notebook_diffs/run.py \
  --torch-version 2.7.1 \
  --torch-version 2.13.0 \
  --with transformers \
  --with datasets \
  ch08/01_main-chapter-code/ch08_main.ipynb
```

PyTorch sürümlerinden biri varsayılan Python sürümü için bir wheel sağlamıyorsa `--python 3.11` kullanılabilir.

Ayrıca tek bir komutta birden çok not defteri verilebileceğini unutmayın. Örneğin,

```bash
notebooks=(
  ch02/01_main-chapter-code/ch02_main.ipynb
  ch02/01_main-chapter-code/ch02_exercise-solutions.ipynb

  ch03/01_main-chapter-code/ch03_main.ipynb
  ch03/01_main-chapter-code/ch03_exercise-solutions.ipynb
  ch03/03_advanced-parser/compare_with_current_parser.ipynb

  ch04/01_main-chapter-code/ch04_main.ipynb
  ch04/01_main-chapter-code/ch04_exercise-solutions.ipynb

  ch05/01_main-chapter-code/ch05_main.ipynb
  ch05/01_main-chapter-code/ch05_exercise-solutions.ipynb

  ch06/01_main-chapter-code/ch06_main.ipynb
  ch06/01_main-chapter-code/ch06_exercise-solutions.ipynb

  # ch07/01_main-chapter-code/ch07_main.ipynb
  # ch07/01_main-chapter-code/ch07_exercise-solutions.ipynb

  ch08/01_main-chapter-code/ch08_main.ipynb
  ch08/01_main-chapter-code/ch08_exercise-solutions.ipynb

  chC/01_main-chapter-code/chC_main.ipynb
  chD/chD_main.ipynb
  chE/chE_main.ipynb
  # chF/01_main-chapter-code/chF_main.ipynb
)

UV_PYTHON=3.13 uv run python tests/run_notebook_diffs/run.py \
  --allow-errors \
  --torch-version 2.7.1 \
  --torch-version 2.13.0 \
  "${notebooks[@]}"
```

Her not defteri kendi dizininden çalışır; böylece göreli yollar Jupyter'daki gibi davranır.

&nbsp;
## Sonuçlar

Sonuçlar varsayılan olarak `tests/run_notebook_diffs/results/` altına yazılır:

```text
results/
└── ch02__01_main-chapter-code__ch02_main/
    ├── torch-2.7.1.ipynb
    ├── torch-2.13.0.ipynb
    └── comparison.md
```

`comparison.md` dosyasının yapısı şöyledir:

---

#### Hücre 23

##### Çıktılar

```diff
--- left outputs
+++ right outputs
@@ -2,6 +2,6 @@
   {
     "name": "stdout",
     "output_type": "stream",
-    "text": "PyTorch version 2.7.1\nApple Silicon GPU\n"
+    "text": "PyTorch version 2.13.0\nApple Silicon GPU\n"
   }
 ]
```

#### Hücre 87

##### Çıktılar

```diff
--- left outputs
+++ right outputs
@@ -207,6 +207,6 @@
   {
     "name": "stdout",
     "output_type": "stream",
-    "text": "\n\nTime: 8.20 sec\n5 tokens/sec\n"
+    "text": "\n\nTime: 8.16 sec\n5 tokens/sec\n"
   }
 ]
```

---



`results/` dizini Git tarafından yok sayılır. İsteğe bağlı olarak başka bir yere yazmak için `--output-dir PATH` kullanabilirsiniz.

Çalıştırma, varsayılan olarak bir not defteri hatasında durur. Eğitim amacıyla bilinçli olarak hata içeren bazı not defterlerinde kalan hücrelerle devam etmek için `--allow-errors` kullanın. 

Hücre başına varsayılan 1 saatlik bir zaman aşımı vardır; bunu `--timeout SECONDS` ile değiştirebilirsiniz.

&nbsp;
## Mevcut not defterlerini karşılaştırmak

Elinizde çalıştırılmış not defterleri zaten varsa, karşılaştırıcı bağımsız olarak da kullanılabilir:

```bash
uv run python tests/run_notebook_diffs/compare.py \
  first.ipynb second.ipynb --output comparison.md
```

