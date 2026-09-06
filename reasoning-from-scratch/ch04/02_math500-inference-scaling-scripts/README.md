# Bölüm 4: Çıkarım Zamanı Ölçeklendirme ile Akıl Yürütmeyi İyileştirmek

> 🇹🇷 **Türkçe çeviri.** Orijinal İngilizce sürüm: [README.md](https://github.com/rasbt/reasoning-from-scratch/blob/main/ch04/02_math500-inference-scaling-scripts/README.md) · Komutlar ve sonuç tabloları birebir korunmuştur.


&nbsp;
## Bonus materyaller

- [cot_prompting_math500.py](cot_prompting_math500.py): modelleri MATH-500 veri kümesinde düşünce zinciri (chain-of-thought) istemiyle değerlendirmek için bağımsız betik
- [self_consistency_math500.py](self_consistency_math500.py): modelleri MATH-500 veri kümesinde öz tutarlılık (self-consistency) örneklemesiyle değerlendirmek için bağımsız betik
- [run_all_experiments_math500.sh](run_all_experiments_math500.sh): Aşağıdaki bu README'de listelenen tüm deneyleri (4-12. satırlar) çalıştıran pratik bir bash betiği

Her iki değerlendirme betiği de kod tekrarını önlemek için [`reasoning_from_scratch`](../../reasoning_from_scratch) paketinden işlevsellik içe aktarır. (Kurulum ayrıntıları için bkz. [bölüm 2 kurulum talimatları](../../ch02/02_setup-tips/python-instructions.md).)



<br>

---

**Not**: `uv` kullanıcısı değilseniz, aşağıdaki örneklerde `uv run ...py` yerine `python ...py` yazın.

---



&nbsp;

## Düşünce zinciri (chain-of-thought) istemi

[`cot_prompting_math500.py`](self_consistency_math500.py) betiği, 4. bölümdeki düşünce zinciri istemi yöntemini uygular.

&nbsp;

<img src="https://sebastianraschka.com/images/reasoning-from-scratch-images/ch04/CH04_F04_raschka.webp" width=600>

&nbsp;

Aşağıdaki tablo bu yaklaşımı (3. satır) 3. bölümdeki temel çizgilerle karşılaştırır:

|    | Yöntem                                       | Model     | Doğruluk | Süre       |
|----|----------------------------------------------|-----------|----------|------------|
| 1  | Temel çizgi (bölüm 3), açgözlü kod çözme     | Base      | 15.2%    | 10.1 min   |
| 2  | Temel çizgi (bölüm 3), açgözlü kod çözme     | Reasoning | 48.2%    | 182.1 min  |
| 3  | Düşünce zinciri istemi ("CoT")               | Base      | 40.6%    | 84.5 min   |

Tabloda gösterilen doğruluk değerleri ve çalışma süreleri, MATH-500 test kümesindeki 500 örneğin tamamı üzerinde bir "cuda" GPU (DGX Spark) kullanılarak hesaplanmıştır.

Birinci satırdaki deneyi çalıştırmak için şunu kullanın:

```bash
python cot_prompting_math500.py \
--which_model "base" \
--dataset_size 500
```

Veya `uv` ile:


```bash
uv run cot_prompting_math500.py \
--which_model "base" \
--dataset_size 500
```

Ek seçenekler için `--help` bayrağını kullanın.



&nbsp;
## Öz tutarlılık (self-consistency) örneklemesi

[`self_consistency_math500.py`](self_consistency_math500.py) betiği, 4. bölümdeki örnekleme yöntemini uygular.

(İsteğe bağlı olarak, tüm `--num_samples` değerini daha hızlı işleme için tek bir yığın hâlinde çalıştıran bir [`self_consistency_math500_batched.py`](self_consistency_math500_batched.py) varyantı vardır. Ancak bunun daha fazla hesaplama belleği gerektirdiğini unutmayın.)

&nbsp;

<img src="https://sebastianraschka.com/images/reasoning-from-scratch-images/ch04/CH04_F17_raschka.webp" width=600>

&nbsp;

Aşağıdaki tablo bu yaklaşımı (4-12. satırlar) 3. bölümdeki temel çizgilerle (1-2. satırlar) karşılaştırır:

|      | Yöntem                                    | Model     | Doğruluk | Süre      |
| ---- | ----------------------------------------- | --------- | -------- | --------- |
| 1    | Temel çizgi (bölüm 3), açgözlü kod çözme  | Base      | 15.2%    | 10.1 min  |
| 2    | Temel çizgi (bölüm 3), açgözlü kod çözme  | Reasoning | 48.2%    | 182.1 min |
| 3    | Düşünce zinciri istemi ("CoT")            | Base      | 40.6%    | 84.5 min  |
| 4    | Temperature ve top-p ("Top-p")            | Base      | 17.8%    | 30.7 min  |
| 5    | "Top-p" + Öz tutarlılık (n=3)             | Base      | 29.6%    | 97.6 min  |
| 6    | "Top-p" + Öz tutarlılık (n=5)             | Base      | 27.8%    | 116.8 min |
| 7    | "Top-p" + Öz tutarlılık (n=10)            | Base      | 31.6%    | 300.4 min |
| 8    | "Top-p" + "CoT"                           | Base      | 33.4%    | 129.2 min |
| 9    | Öz tutarlılık (n=3) + "Top-p" + "CoT"     | Base      | 42.2%    | 211.6 min |
| 10   | Öz tutarlılık (n=5) + "Top-p" + "CoT"     | Base      | 48.0%    | 452.9 min |
| 11   | Öz tutarlılık (n=10) + "Top-p" + "CoT"    | Base      | 52.0%    | 862.6 min |
| 12   | Öz tutarlılık (n=3) + "Top-p" + "CoT"     | Reasoning | 55.2%    | 544.4 min |

Tabloda gösterilen doğruluk değerleri ve çalışma süreleri, MATH-500 test kümesindeki 500 örneğin tamamı üzerinde bir "cuda" GPU (DGX Spark) kullanılarak hesaplanmıştır.

Aşağıdaki kodlar, 4-12. satırlardaki öz tutarlılık deneylerinin nasıl çalıştırılacağını gösterir (`uv` kullanıcısı değilseniz `uv run` yerine `python` yazın).

**4. satır:**

```bash
uv run self_consistency_math500.py \
    --which_model "base" \
    --temperature 0.9 \
    --top_p 0.9 \
    --num_samples 1 \
    --dataset_size 500
```

**5. satır:**

```bash
uv run self_consistency_math500.py \
    --which_model "base" \
    --temperature 0.9 \
    --top_p 0.9 \
    --num_samples 3 \
    --dataset_size 500
```

**6. satır:**

```bash
uv run self_consistency_math500.py \
    --which_model "base" \
    --temperature 0.9 \
    --top_p 0.9 \
    --num_samples 5 \
    --dataset_size 500
```

**7. satır:**

```bash
uv run self_consistency_math500.py \
    --which_model "base" \
    --temperature 0.9 \
    --top_p 0.9 \
    --num_samples 10 \
    --dataset_size 500
```

**8. satır:**

```bash
uv run self_consistency_math500.py \
    --which_model "base" \
    --temperature 0.9 \
    --top_p 0.9 \
    --num_samples 1 \
    --dataset_size 500 \
    --prompt_suffix "\n\nExplain step by step."
```

**9. satır:**

```bash
uv run self_consistency_math500.py \
    --which_model "base" \
    --temperature 0.9 \
    --top_p 0.9 \
    --num_samples 3 \
    --dataset_size 500 \
    --prompt_suffix "\n\nExplain step by step."
```

**10. satır:**

```bash
uv run self_consistency_math500.py \
    --which_model "base" \
    --temperature 0.9 \
    --top_p 0.9 \
    --num_samples 5 \
    --dataset_size 500 \
    --prompt_suffix "\n\nExplain step by step."
```

**11. satır:**

```bash
uv run self_consistency_math500.py \
    --which_model "base" \
    --temperature 0.9 \
    --top_p 0.9 \
    --num_samples 10 \
    --dataset_size 500 \
    --prompt_suffix "\n\nExplain step by step."
```

**12. satır:**

```bash
uv run self_consistency_math500.py \
    --which_model "reasoning" \
    --temperature 0.9 \
    --top_p 0.9 \
    --num_samples 3 \
    --dataset_size 500 \
    --prompt_suffix "\n\nExplain step by step."
```


Ek seçenekler için `--help` bayrağını kullanın.
