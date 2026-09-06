# Bölüm 6: Pekiştirmeli Öğrenme ile Akıl Yürütme Modellerini Eğitmek

> 🇹🇷 **Türkçe çeviri.** Orijinal İngilizce sürüm: [README.md](https://github.com/rasbt/reasoning-from-scratch/blob/main/ch06/02_rlvr_grpo_scripts_intro/README.md) · Komutlar ve sonuç tabloları birebir korunmuştur.

&nbsp;

&nbsp;
## Bonus materyaller

- [rlvr_grpo_original_no_kl.py](rlvr_grpo_original_no_kl.py): Doğrulanabilir ödüllerle pekiştirmeli öğrenme (RLVR) kullanarak bir akıl yürütme modeli eğitmek üzere orijinal GRPO algoritmasını uygulayan betik. Algoritma [DeepSeek R1](https://arxiv.org/abs/2501.12948) tarafından kullanılmış ve ilk olarak [DeepSeekMath](https://arxiv.org/abs/2402.03300) makalesinde önerilmiştir. Ancak bu betik KL ıraksama terimini atlar ([DAPO](https://arxiv.org/abs/2503.14476), [Dr. GRPO](https://arxiv.org/abs/2503.20783), [Olmo 3](https://arxiv.org/abs/2512.13961) ve diğerlerinde önerildiği gibi)
  - KL ıraksama terimi, eğitilen modelin orijinal modelden çok fazla sapmamasını sağlar, ancak performansa zarar verebilir (özellikle matematik görevlerinde)
  - Bu betik kavramsal olarak 6. bölümdekiyle aynı kodu uygular; ancak iki küçük performans ayarı içerir:
    1. Verimi %20 artırmak için `torch.multinomial` örnekleyicisindeki `.cpu()` dönüşümünün kaldırılması; bunun nasıl uygulandığına dair daha fazla bilgi için bkz. [PR #178](https://github.com/rasbt/reasoning-from-scratch/pull/178)
    2. `--skip-zero-advantage-updates` bayrağı kullanıldığında tüm ödüller eşitse model güncellemesinin atlanması; bu, eğitimi daha da hızlandırır ve bellek gereksinimlerini düşürebilir (çünkü `--max_new_tokens` sınırını aşan uzun diziler en maliyetli olanlardır ve doğru yanıt üretilmeden token sınırına ulaşıldığı için genellikle sıfır ödülle sonuçlanır); bunun nasıl uygulandığına dair daha fazla bilgi için bkz. [PR #186](https://github.com/rasbt/reasoning-from-scratch/pull/186)
    - Betiği yukarıda belirtilen bu iki iyileştirme olmadan görmek isterseniz, orijinal koda [buradan](https://github.com/rasbt/reasoning-from-scratch/blob/da009e41aacb17a433968cf84a4a6cf2a0fa4655/ch06/02_rlvr_grpo_scripts_intro/rlvr_grpo_original_no_kl.py) bakabilirsiniz
- [rlvr_grpo_original_no_kl_batched.py](rlvr_grpo_original_no_kl_batched.py): Yukarıdakiyle aynı, ancak yığınlar hâlinde eğitimi destekler. Ancak bunun bellek gereksinimlerini artırdığını ve dolayısıyla rollout sayısını ile rollout uzunluklarını düşürmeyi gerektirebileceğini unutmayın. Kullanımı, `--num_batches` eklemesi dışında yukarıdaki betikle aynıdır.
  - 3. bölümdeki [evaluate_math500_batched.py](https://github.com/rasbt/reasoning-from-scratch/blob/main/ch03/02_math500-verifier-scripts/evaluate_math500_batched.py) kodunun aksine, bu kodun [qwen3_batched.py](https://github.com/rasbt/reasoning-from-scratch/blob/main/reasoning_from_scratch/qwen3_batched.py) dosyasından `Qwen3Model` içe aktarmasına gerek olmadığını unutmayın; ayrıntılı açıklama için bkz. [PR #179](https://github.com/rasbt/reasoning-from-scratch/pull/179) 

- [rlvr_grpo_original_no_kl_batched_fsdp.py](rlvr_grpo_original_no_kl_batched_fsdp.py): Yukarıdakiyle aynı, ancak PyTorch'un FSDP özelliğiyle birden çok GPU üzerinde eğitimi destekler. Birden çok GPU'ya erişiminiz varsa eğitim için önerilen betik budur. Kullanımı, `--num_gpus` eklemesi dışında yukarıdaki betikle aynıdır.

Betikler, kod tekrarını önlemek için [`reasoning_from_scratch`](../../reasoning_from_scratch) paketinden bazı işlevleri içe aktarır. (Kurulum ayrıntıları için bkz. [bölüm 2 kurulum talimatları](../../ch02/02_setup-tips/python-instructions.md).) Ancak bu durumda kod, daha kolay incelenip değiştirilebilmesi için bölümün temel fonksiyonlarını da yeniden uygular.



<br>

---

**Not**: `uv` kullanıcısı değilseniz, aşağıdaki örneklerde `uv run ...py` yerine `python ...py` yazın.

---


&nbsp;

|      | Yöntem                                 | Adım | Maks token | Rollout sayısı | MATH-500 Doğr. | Ort. token sayısı |
| ---- | -------------------------------------- | ---- | ---------- | ------------ | ------------ | --------------- |
| 1    | Base (bölüm 3)                         | -    |            |              | 15.2%        | 78.85           |
| 2    | Reasoning (bölüm 3)                    | -    |            |              | 48.2%        | 1369.79         |
| 3    | GRPO original (bölüm 7)                | 50   | 512        | 8            | 33.4%        | 910.33          |
| 4    | GRPO original (bölüm 7)                | 100  | 512        | 8            | 0.4%         | 1168.05         |
| 5    | GRPO original but no KL (bu bölüm)     | 50   | 512        | 8            | 47.4%        | 586.11          |
| 6    | GRPO original but no KL (bu bölüm)     | 100  | 512        | 8            | 44.0%        | 555.95          |
| 7    | GRPO Olmo 3 mod. (bölüm 7)             | 50   | 512        | 8            | 46.4%        | 601.61          |
| 8    | GRPO Olmo 3 mod. (bölüm 7)             | 100  | 512        | 8            | 45.4%        | 589.51          |
| 9    | GRPO DeepSeek V3.2 mod. (bölüm 7)      | 50   | 512        | 8            | 44.2%        | 618.49          |
| 10   | GRPO DeepSeek V3.2 mod. (bölüm 7)      | 100  | 512        | 8            | 45.2%        | 676.96          |

Kontrol noktaları her 50 adımda bir kaydedilir. Bir betiği KeyboardInterrupt ile keserseniz, son adımı da bir kontrol noktası olarak kaydeder.

Gerekli hesaplama belleği açısından daha erişilebilir olması için eğitimin yalnızca 512 üretilmiş token'a (yukarıdaki tablodaki maks token) izin verdiğini unutmayın.

Ancak değerlendirme betiği (3. bölümdekiyle aynı yöntem) 2048 üretilmiş token'a kadar izin verir ve yukarıdaki tablodaki "Ort. token sayısı" sütunu, MATH-500 test veri kümesi genelinde ortalama kaç token kullanıldığını ölçer. (Eğitim, MATH veri kümesinde MATH-500 test kümesiyle örtüşmeyen 12.000 örnek üzerinde yapılır. Daha fazla ayrıntı için bkz. [https://github.com/rasbt/math_full_minus_math500](https://github.com/rasbt/math_full_minus_math500).)

**1. satır**

```bash
uv run ../../ch03/02_math500-verifier-scripts/evaluate_math500.py \
--dataset_size 500 \
--which_model base
```

- İpucu: Betiğin makinenize özgü toplam çalışma süresine dair bir tahmin görmek için yukarıdaki komuta `--show_eta` ekleyebilirsiniz

**2. satır**

```bash
uv run ../../ch03/02_math500-verifier-scripts/evaluate_math500.py \
--dataset_size 500 \
--which_model reasoning
```

**3. ve 4. satırlar**

```bash
uv run ../../ch07/02_rlvr_grpo_scripts_advanced/rlvr_grpo_original.py \
--num_rollouts 8 \
--max_new_tokens 512 
```

Ardından modeli değerlendirmek için üretilen kontrol noktası üzerinde `evaluate_math500.py` betiğini çalıştırın. Örneğin,

```bash
uv run ../../ch03/02_math500-verifier-scripts/evaluate_math500.py \
--dataset_size 500 \
--which_model base \
--checkpoint_path checkpoints/rlvr_grpo_original/qwen3-0.6B-rlvr-grpo-step00050.pth
```

**5. ve 6. satırlar**

```bash
uv run rlvr_grpo_original_no_kl.py \
--num_rollouts 8 \
--steps 100 \
--max_new_tokens 512
```

**7. ve 8. satırlar**

```bash
uv run ../../ch07/02_rlvr_grpo_scripts_original/rlvr_grpo_olmo3.py \
--num_rollouts 8 \
--max_new_tokens 512 
```

**9. ve 10. satırlar**

```bash
uv run ../../ch07/02_rlvr_grpo_scripts_original/rlvr_grpo_deepseek_v32.py \
--num_rollouts 8 \
--max_new_tokens 512 
```


<br>

RAM'iniz kısıtlıysa, rollout sayısını (`--num_rollouts`) veya yanıt uzunluğunu (`--max_new_tokens`) düşürmeyi değerlendirin. Aşağıdaki tablo referans olarak bazı kaynak gereksinimlerini listeler.



| num_rollouts | max_new_tokens | Gerekli RAM (GB) |
| ------------ | -------------- | ----------------- |
| 8            | 1024           | 30.50 GB          |
| 8            | 512            | 20.31 GB          |
| 8            | 256            | 15.60 GB          |
| 4            | 1024           | 12.80 GB          |
| 4            | 512            | 14.60 GB          |
| 4            | 256            | 10.59 GB          |


Token veya rollout sayısını düşürmenin performansı büyük olasılıkla olumsuz etkileyeceğini unutmayın. Düşük bir rollout sayısı kullanıyorsanız, `--accum_steps` değerini 1'den 2 veya 4'e çıkararak (gradyan biriktirme) eğitim kararlılığını bir miktar iyileştirebilirsiniz; ancak bu daha fazla hesaplama süresi gerektirir. 

Bu ayarlarla orijinal ("vanilla") GRPO yönteminin 50 adımdan fazlası için pek kararlı olmadığını unutmayın; 50 adımdan fazla eğitmek istiyorsanız 7. bölümdeki geliştirilmiş sürümleri değerlendirebilirsiniz.


<br>

Orijinal GRPO algoritmasının, eğitimi kararlı hâle getirmek ve iyileştirmek için çeşitli şekillerde geliştirilebileceğini unutmayın; bu [sonraki bölümün](../../ch07) konusudur.



&nbsp;
## Eğitim koşularını grafikleştirmek

[plot_metrics.py](plot_metrics.py) betiği, CSV biçimindeki eğitim koşularını grafikleştirmek için kullanılabilir. `logs` klasöründe 200 adımlık örnek bir koşu bulunur (günlük dosyası, `--max_new_tokens 2048` artırımı dışında varsayılan ayarlarla oluşturulmuştur):

```bash
uv run plot_metrics.py \
--csv logs/rlvr_grpo_original_no_kl_metrics.csv \
--moving_average 20
```

(`--moving_average 20` ayarı, daha yumuşak bir eğilim çizgisi için geçmiş adımların %20'si üzerinden ortalama alır.)

<img src="https://sebastianraschka.com/images/reasoning-from-scratch-images/ch06/other/plot.webp?1" width="600px">
