# Bölüm 5: Öz İyileştirme (Self-Refinement) ile Çıkarım Zamanı Ölçeklendirme

> 🇹🇷 **Türkçe çeviri.** Orijinal İngilizce sürüm: [README.md](https://github.com/rasbt/reasoning-from-scratch/blob/main/ch05/02_math500-more-inference-scaling-scripts/README.md) · Komutlar ve sonuç tabloları birebir korunmuştur.


&nbsp;
## Bonus materyaller

- [self_refinement_math500.py](self_refinement_math500.py): modelleri MATH-500 veri kümesinde öz iyileştirme ile değerlendirmek için bağımsız betik

Betik, kod tekrarını önlemek için [`reasoning_from_scratch`](../../reasoning_from_scratch) paketinden işlevsellik içe aktarır. (Kurulum ayrıntıları için bkz. [bölüm 2 kurulum talimatları](../../ch02/02_setup-tips/python-instructions.md).)



<br>

---

**Not**: `uv` kullanıcısı değilseniz, aşağıdaki örneklerde `uv run ...py` yerine `python ...py` yazın.

---



&nbsp;

## Öz iyileştirme (self-refinement)

[`self_refinement_math500.py`](self_refinement_math500.py) betiği, 5. bölümdeki öz iyileştirme yöntemini uygular.


&nbsp;

<img src="https://sebastianraschka.com/images/reasoning-from-scratch-images/ch05/CH05_F21_raschka.webp" width=600>

&nbsp;



| #  | Yöntem          | Puanlayıcı | Yineleme  | Model     | Doğruluk | Süre      |
|----|-----------------|-----------|------------|-----------|----------|-----------|
| 1  | Temel çizgi (böl.3) | -         | -          | Base      | 15.2%    | 10.1 min  |
| 2  | Öz iyileştirme  | None      | 1          | Base      | 25.0%    | 84.8 min  |
| 3  | Öz iyileştirme  | None      | 2          | Base      | 22.0%    | 165.4 min |
|    |                 |           |            |           |          |           |
| 4  | Öz iyileştirme  | Heuristic | 1          | Base      | 21.6%    | 84.7 min  |
| 5  | Öz iyileştirme  | Heuristic | 2          | Base      | 20.8%    | 151.4 min |
|    |                 |           |            |           |          |           |
| 6  | Öz iyileştirme  | Logprob   | 1          | Base      | 21.4%    | 85.3 min  |
| 7  | Öz iyileştirme  | Logprob   | 2          | Base      | 22.0%    | 165.3 min |
|    |                 |           |            |           |          |           |
| 8  | Öz iyileştirme  | Logp-ex   | 1          | Base      | 20.4%    | 85.0 min  |
| 9  | Öz iyileştirme  | Logp-ex   | 2          | Base      | 21.2%    | 160.2 min |
|    |                 |           |            |           |          |           |
| 10 | Temel çizgi (böl.3) | -         | -          | Reasoning | 48.2%    | 182.1 min |
| 11 | Öz iyileştirme  | None      | 1          | Reasoning | 56.6%    | 498.8 min |
| 12 | Öz iyileştirme  | None      | 2          | Reasoning | 56.6%    | 713.9 min |
|    |                 |           |            |           |          |           |
| 13 | Öz iyileştirme  | Heuristic | 1          | Reasoning | 57.8%    | 498.6 min |
| 14 | Öz iyileştirme  | Heuristic | 2          | Reasoning | 57.8%    | 713.9 min |
|    |                 |           |            |           |          |           |
| 15 | Öz iyileştirme  | Logprob   | 1          | Reasoning | 48.4%    | 499.7 min |
| 16 | Öz iyileştirme  | Logprob   | 2          | Reasoning | 48.6%    | 753.0 min |

Tabloda gösterilen doğruluk değerleri ve çalışma süreleri, MATH-500 test kümesindeki 500 örneğin tamamı üzerinde bir "cuda" GPU (DGX Spark) kullanılarak hesaplanmıştır.

Aşağıdaki kodlar, 4-12. satırlardaki öz tutarlılık deneylerinin nasıl çalıştırılacağını gösterir (`uv` kullanıcısı değilseniz `uv run` yerine `python` yazın).

**2. satır:**

```bash
uv run self_refinement_math500.py \
    --which_model "base" \
    --temperature 0.7 \
    --top_p 0.9 \
    --dataset_size 500 \
    --iterations 1 \
    --scoring "none"
```

**3. satır:**

```bash
uv run self_refinement_math500.py \
    --which_model "base" \
    --temperature 0.7 \
    --top_p 0.9 \
    --dataset_size 500 \
    --iterations 2 \
    --scoring "none"
```

**4. satır:**

```bash
uv run self_refinement_math500.py \
    --which_model "base" \
    --temperature 0.7 \
    --top_p 0.9 \
    --dataset_size 500 \
    --iterations 1 \
    --scoring "heuristic"
```

**5. satır:**

```bash
uv run self_refinement_math500.py \
    --which_model "base" \
    --temperature 0.7 \
    --top_p 0.9 \
    --dataset_size 500 \
    --iterations 2 \
    --scoring "heuristic"
```

**6. satır:**

```bash
uv run self_refinement_math500.py \
    --which_model "base" \
    --temperature 0.7 \
    --top_p 0.9 \
    --dataset_size 500 \
    --iterations 1 \
    --scoring "logprob"
```

**7. satır:**

```bash
uv run self_refinement_math500.py \
    --which_model "base" \
    --temperature 0.7 \
    --top_p 0.9 \
    --dataset_size 500 \
    --iterations 2 \
    --scoring "logprob"
```

**8. satır:**

```bash
uv run self_refinement_math500.py \
    --which_model "base" \
    --temperature 0.7 \
    --top_p 0.9 \
    --dataset_size 500 \
    --iterations 1 \
    --scoring "logprob_extract"
```

**9. satır:**

```bash
uv run self_refinement_math500.py \
    --which_model "base" \
    --temperature 0.7 \
    --top_p 0.9 \
    --dataset_size 500 \
    --iterations 2 \
    --scoring "logprob_extract"
```

**11. satır:**

```bash
uv run self_refinement_math500.py \
    --which_model "reasoning" \
    --temperature 0.7 \
    --top_p 0.9 \
    --dataset_size 500 \
    --iterations 1 \
    --scoring "none"
```

**12. satır:**

```bash
uv run self_refinement_math500.py \
    --which_model "reasoning" \
    --temperature 0.7 \
    --top_p 0.9 \
    --dataset_size 500 \
    --iterations 2 \
    --scoring "none"
```

**13. satır:**

```bash
uv run self_refinement_math500.py \
    --which_model "reasoning" \
    --temperature 0.7 \
    --top_p 0.9 \
    --dataset_size 500 \
    --iterations 1 \
    --scoring "heuristic"
```

**14. satır:**

```bash
uv run self_refinement_math500.py \
    --which_model "reasoning" \
    --temperature 0.7 \
    --top_p 0.9 \
    --dataset_size 500 \
    --iterations 2 \
    --scoring "heuristic"
```

**15. satır:**

```bash
uv run self_refinement_math500.py \
    --which_model "reasoning" \
    --temperature 0.7 \
    --top_p 0.9 \
    --dataset_size 500 \
    --iterations 1 \
    --scoring "logprob"
```

**16. satır:**

```bash
uv run self_refinement_math500.py \
    --which_model "reasoning" \
    --temperature 0.7 \
    --top_p 0.9 \
    --dataset_size 500 \
    --iterations 2 \
    --scoring "logprob"
```




&nbsp;

## Puanlayıcı temelli eşitlik bozucu ile öz tutarlılık

[`self_consistency_scorer_math500.py`](self_consistency_scorer_math500.py) betiği, öz tutarlılığı 5. bölümde uygulanan puanlayıcılara dayalı bir eşitlik bozma (tie-breaking) mekanizmasıyla genişletir.


&nbsp;

<img src="https://sebastianraschka.com/images/reasoning-from-scratch-images/appendix-b/majority-vote.webp" width=600>

&nbsp;



|   | Yöntem                                   | Model | Doğruluk | Süre      |
|---|------------------------------------------|-------|----------|-----------|
| 1 | CoT istemli bölüm 4 temel çizgisi        | Base  | 33.4%    | 129.2 min |
| 2 | Öz tutarlılık (n=3) + çoğunluk oyu       | Base  | 43.2%    | 328.2 min |
| 3 | Öz tutarlılık (n=3) + sezgisel (heuristic) | Base | 43.4%   | 326.5 min |
| 4 | Öz tutarlılık (n=3) + ort. logprob       | Base  | 44.8%    | 327.7 min |


Tabloda gösterilen doğruluk değerleri ve çalışma süreleri, MATH-500 test kümesindeki 500 örneğin tamamı üzerinde bir "cuda" GPU (DGX Spark) kullanılarak hesaplanmıştır.

Aşağıdaki kodlar, 2-4. satırlardaki öz tutarlılık deneylerinin nasıl çalıştırılacağını gösterir (`uv` kullanıcısı değilseniz `uv run` yerine `python` yazın).

**2. satır:**

```bash
uv run self_consistency_scorer_math500.py \
    --which_model "base" \
    --temperature 0.9 \
    --top_p 0.9 \
    --num_samples 3 \
    --dataset_size 500 \
    --prompt_suffix "\n\nExplain step by step." \
    --scoring "none"
```

**3. satır:**

```bash
uv run self_consistency_scorer_math500.py \
    --which_model "base" \
    --temperature 0.9 \
    --top_p 0.9 \
    --num_samples 3 \
    --dataset_size 500 \
    --prompt_suffix "\n\nExplain step by step." \
    --scoring "heuristic"
```

**4. satır:**

```bash
uv run self_consistency_scorer_math500.py \
    --which_model "base" \
    --temperature 0.9 \
    --top_p 0.9 \
    --num_samples 3 \
    --dataset_size 500 \
    --prompt_suffix "\n\nExplain step by step." \
    --scoring "logprob"
```

&nbsp;

## Best-of-N

[`self_consistency_scorer_math500.py`](self_consistency_scorer_math500.py) betiği, Best-of-N çıkarım ölçeklendirme yaklaşımını uygular.

Best-of-N, birden çok yanıt üretmemiz açısından öz tutarlılığa benzer. Ancak nihai yanıtı çoğunluk oyuyla seçmek yerine, üretilen tüm yanıtları bir puanlama fonksiyonuyla puanlarız.

[`best_of_n_math500.py`](best_of_n_math500.py) betiği, öz tutarlılığı 5. bölümde uygulanan puanlayıcılara dayalı bir eşitlik bozma mekanizmasıyla genişletir.


&nbsp;

<img src="https://sebastianraschka.com/images/reasoning-from-scratch-images/appendix-b/best-of-n.webp" width=600>

&nbsp;

|   | Yöntem                                   | Model | Doğruluk | Süre      |
|---|------------------------------------------|-------|----------|-----------|
| 1 | Düşünce zinciri istemli temel çizgi      | Base  | 33.4%    | 129.2 min |
| 2 | Best-of-N (n=3) + sezgisel (heuristic)   | Base  | 40.6%    | 327.7 min |
| 3 | Best-of-N (n=3) + ort. logprob           | Base  | 43.2%    | 330.2 min |


Tabloda gösterilen doğruluk değerleri ve çalışma süreleri, MATH-500 test kümesindeki 500 örneğin tamamı üzerinde bir "cuda" GPU (DGX Spark) kullanılarak hesaplanmıştır.

Aşağıdaki kodlar, 2. ve 3. satırlardaki öz tutarlılık deneylerinin nasıl çalıştırılacağını gösterir (`uv` kullanıcısı değilseniz `uv run` yerine `python` yazın).

**2. satır:**

```bash
uv run best_of_n_math500.py \
    --which_model "base" \
    --temperature 0.9 \
    --top_p 0.9 \
    --num_samples 3 \
    --dataset_size 500 \
    --prompt_suffix "\n\nExplain step by step."
    --scoring "heuristic"
)
```

**3. satır:**

```bash
uv run best_of_n_math500.py \
    --which_model "base" \
    --temperature 0.9 \
    --top_p 0.9 \
    --num_samples 3 \
    --dataset_size 500 \
    --prompt_suffix "\n\nExplain step by step."
    --scoring "logprob"
)
```
s