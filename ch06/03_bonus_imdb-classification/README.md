# 50 Bin IMDb Film Yorumunun Duygusunu Sınıflandıran Ek Deneyler

> 🇹🇷 **Türkçe çeviri.** Orijinal İngilizce sürüm: [README.md](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch06/03_bonus_imdb-classification/README.md) · Komut, tablo ve çıktı blokları birebir korunmuştur.

## Genel Bakış

Bu klasör, 6. bölümdeki (kod çözücü tarzı) GPT-2 (2018) modelini [BERT (2018)](https://arxiv.org/abs/1810.04805), [RoBERTa (2019)](https://arxiv.org/abs/1907.11692) ve [ModernBERT (2024)](https://arxiv.org/abs/2412.13663) gibi kodlayıcı (encoder) tarzı LLM'lerle karşılaştırmak için ek deneyler içerir. 6. bölümdeki küçük SPAM veri kümesi yerine, IMDb'den 50 bin film yorumundan oluşan veri kümesini ([veri kümesi kaynağı](https://ai.stanford.edu/~amaas/data/sentiment/)) ikili sınıflandırma hedefiyle kullanıyor ve yorumu yazanın filmi beğenip beğenmediğini tahmin ediyoruz. Bu dengeli bir veri kümesidir; dolayısıyla rastgele bir tahmin %50 doğruluk vermelidir.



|         | Model                           | Test accuracy |
| ------- | ------------------------------- | ------------- |
| **1.1** | 124M GPT-2 Baseline             | 91.88%        |
| **1.2** | 124M GPT-2 Baseline (with Muon) | 92.40%        |
| **2**   | 340M BERT                       | 90.89%        |
| **3**   | 66M DistilBERT                  | 91.40%        |
| **4**   | 355M RoBERTa                    | 92.95%        |
| **5**   | 304M DeBERTa-v3                 | 94.69%        |
| **6**   | 149M ModernBERT Base            | 93.79%        |
| **7**   | 395M ModernBERT Large           | 95.07%        |
| **8**   | Logistic Regression Baseline    | 88.85%        |




&nbsp;
## Adım 1: Bağımlılıkları Kurun

Ek bağımlılıkları şu komutla kurun:

```bash
pip install -r requirements-extra.txt
```

&nbsp;
## Adım 2: Veri Kümesini İndirin

Kodlar, bir film yorumunun olumlu mu olumsuz mu olduğunu tahmin etmek için IMDb'den 50 bin film yorumunu ([veri kümesi kaynağı](https://ai.stanford.edu/~amaas/data/sentiment/)) kullanır.

`train.csv`, `validation.csv` ve `test.csv` veri kümelerini oluşturmak için aşağıdaki kodu çalıştırın:

```bash
python download_prepare_dataset.py
```

&nbsp;
## Adım 3: Modelleri Çalıştırın

&nbsp;
### 1) 124M GPT-2 Temel Çizgi

6. bölümde kullanılan 124M GPT-2 modeli; önceden eğitilmiş ağırlıklarla başlar ve tüm ağırlıklara ince ayar yapar:

```bash
python train_gpt.py --trainable_layers "all" --num_epochs 1
```

```
Ep 1 (Step 000000): Train loss 3.706, Val loss 3.853
Ep 1 (Step 000050): Train loss 0.682, Val loss 0.706
...
Ep 1 (Step 004300): Train loss 0.199, Val loss 0.285
Ep 1 (Step 004350): Train loss 0.188, Val loss 0.208
Training accuracy: 95.62% | Validation accuracy: 95.00%
Training completed in 9.48 minutes.

Evaluating on the full datasets ...

Training accuracy: 95.64%
Validation accuracy: 92.32%
Test accuracy: 91.88%
```

<br>

Alternatif [train_gpt_muon.py](train_gpt_muon.py) betiği, aynı kodu gömme dışındaki katmanlar için PyTorch'un yeni Muon optimize edicisiyle çalıştırır. Muon hakkında daha fazla bilgi için orijinal [makaleye](https://arxiv.org/abs/2502.16982) ve [../../ch05/18_muon](../../ch05/18_muon) klasörüne bakın.


```bash
python train_gpt_muon.py --trainable_layers "all" --num_epochs 1
```

```
Ep 1 (Step 000000): Train loss 2.659, Val loss 3.237
Ep 1 (Step 000050): Train loss 0.919, Val loss 0.799
...
Training accuracy: 98.12% | Validation accuracy: 91.88%
Training completed in 23.01 minutes.

Evaluating on the full datasets ...

Training accuracy: 97.45%
Validation accuracy: 92.52%
Test accuracy: 92.40%
```

Gözlem: Muon daha iyi/daha hızlı optimize ediyor gibi görünüyor, ancak bu burada eğitim kümesinde daha fazla aşırı öğrenmeye (overfitting) de yol açıyor.

Not: Bu farklı bir GPU'da çalıştırıldığı için eğitim süreleri doğrudan karşılaştırılabilir değildir.

<br>

---

<br>

&nbsp;
### 2) 340M BERT


340 milyon parametreli, kodlayıcı tarzı bir [BERT](https://arxiv.org/abs/1810.04805) modeli:

```bash
python train_bert_hf.py --trainable_layers "all" --num_epochs 1 --model "bert"
```

```
Ep 1 (Step 000000): Train loss 0.848, Val loss 0.775
Ep 1 (Step 000050): Train loss 0.655, Val loss 0.682
...
Ep 1 (Step 004300): Train loss 0.146, Val loss 0.318
Ep 1 (Step 004350): Train loss 0.204, Val loss 0.217
Training accuracy: 92.50% | Validation accuracy: 88.75%
Training completed in 7.65 minutes.

Evaluating on the full datasets ...

Training accuracy: 94.35%
Validation accuracy: 90.74%
Test accuracy: 90.89%
```

<br>

---

<br>

&nbsp;
### 3) 66M DistilBERT

66 milyon parametreli, kodlayıcı tarzı bir [DistilBERT](https://arxiv.org/abs/1910.01108) modeli (340 milyon parametreli BERT modelinden damıtılmıştır); önceden eğitilmiş ağırlıklarla başlar ve yalnızca son transformer bloğu ile çıkış katmanlarını eğitir:



```bash
python train_bert_hf.py --trainable_layers "all" --num_epochs 1 --model "distilbert"
```

```
Ep 1 (Step 000000): Train loss 0.693, Val loss 0.688
Ep 1 (Step 000050): Train loss 0.452, Val loss 0.460
...
Ep 1 (Step 004300): Train loss 0.179, Val loss 0.272
Ep 1 (Step 004350): Train loss 0.199, Val loss 0.182
Training accuracy: 95.62% | Validation accuracy: 91.25%
Training completed in 4.26 minutes.

Evaluating on the full datasets ...

Training accuracy: 95.30%
Validation accuracy: 91.12%
Test accuracy: 91.40%
```
<br>

---

<br>

&nbsp;
### 4) 355M RoBERTa

355 milyon parametreli, kodlayıcı tarzı bir [RoBERTa](https://arxiv.org/abs/1907.11692) modeli; önceden eğitilmiş ağırlıklarla başlar ve yalnızca son transformer bloğu ile çıkış katmanlarını eğitir:


```bash
python train_bert_hf.py --trainable_layers "last_block" --num_epochs 1 --model "roberta"
```

```
Ep 1 (Step 000000): Train loss 0.695, Val loss 0.698
Ep 1 (Step 000050): Train loss 0.670, Val loss 0.690
...
Ep 1 (Step 004300): Train loss 0.083, Val loss 0.098
Ep 1 (Step 004350): Train loss 0.170, Val loss 0.086
Training accuracy: 98.12% | Validation accuracy: 96.88%
Training completed in 11.22 minutes.

Evaluating on the full datasets ...

Training accuracy: 96.23%
Validation accuracy: 94.52%
Test accuracy: 94.69%
```

<br>

---

<br>

&nbsp;
### 5) 304M DeBERTa-v3

304 milyon parametreli, kodlayıcı tarzı bir [DeBERTa-v3](https://arxiv.org/abs/2111.09543) modeli. DeBERTa-v3, ayrıştırılmış (disentangled) dikkat ve geliştirilmiş konum kodlamasıyla önceki sürümlerin üzerine çıkar.


```bash
python train_bert_hf.py --trainable_layers "all" --num_epochs 1 --model "deberta-v3-base"
```

```
Ep 1 (Step 000000): Train loss 0.689, Val loss 0.694
Ep 1 (Step 000050): Train loss 0.673, Val loss 0.683
...
Ep 1 (Step 004300): Train loss 0.126, Val loss 0.149
Ep 1 (Step 004350): Train loss 0.211, Val loss 0.138
Training accuracy: 92.50% | Validation accuracy: 94.38%
Training completed in 7.20 minutes.

Evaluating on the full datasets ...

Training accuracy: 93.44%
Validation accuracy: 93.02%
Test accuracy: 92.95%
```

<br>

---

<br>



&nbsp;
### 6) 149M ModernBERT Base

[ModernBERT (2024)](https://arxiv.org/abs/2412.13663), verimliliği ve performansı artırmak için paralel artık bağlantılar (residual connections) ve kapılı doğrusal birimler (GLU) gibi mimari iyileştirmeler içeren, optimize edilmiş bir BERT yeniden uygulamasıdır. BERT'in orijinal ön eğitim hedeflerini korurken modern donanımda daha hızlı çıkarım ve daha iyi ölçeklenebilirlik sağlar.

```bash
python train_bert_hf.py --trainable_layers "all" --num_epochs 1 --model "modernbert-base"
```



```
Ep 1 (Step 000000): Train loss 0.699, Val loss 0.698
Ep 1 (Step 000050): Train loss 0.564, Val loss 0.606
...
Ep 1 (Step 004300): Train loss 0.086, Val loss 0.168
Ep 1 (Step 004350): Train loss 0.160, Val loss 0.131
Training accuracy: 95.62% | Validation accuracy: 93.75%
Training completed in 10.27 minutes.

Evaluating on the full datasets ...

Training accuracy: 95.72%
Validation accuracy: 94.00%
Test accuracy: 93.79%
```

<br>

---

<br>


&nbsp;
### 7) 395M ModernBERT Large

Yukarıdakiyle aynı, ancak daha büyük ModernBERT varyantı kullanılıyor.

```bash
python train_bert_hf.py --trainable_layers "all" --num_epochs 1 --model "modernbert-large"
```



```
Ep 1 (Step 000000): Train loss 0.666, Val loss 0.662
Ep 1 (Step 000050): Train loss 0.548, Val loss 0.556
...
Ep 1 (Step 004300): Train loss 0.083, Val loss 0.115
Ep 1 (Step 004350): Train loss 0.154, Val loss 0.116
Training accuracy: 96.88% | Validation accuracy: 95.62%
Training completed in 27.69 minutes.

Evaluating on the full datasets ...

Training accuracy: 97.04%
Validation accuracy: 95.30%
Test accuracy: 95.07%
```




<br>

---

<br>

&nbsp;
### 8) Lojistik Regresyon Temel Çizgisi

Temel çizgi olarak scikit-learn tabanlı bir [lojistik regresyon](https://sebastianraschka.com/blog/2022/losses-learned-part1.html) sınıflandırıcısı:


```bash
python train_sklearn_logreg.py
```

```
Dummy classifier:
Training Accuracy: 50.01%
Validation Accuracy: 50.14%
Test Accuracy: 49.91%


Logistic regression classifier:
Training Accuracy: 99.80%
Validation Accuracy: 88.62%
Test Accuracy: 88.85%
```
