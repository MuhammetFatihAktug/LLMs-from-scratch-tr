# Bölüm 8 Bonus Materyali: Damıtma ile Eğitim

> 🇹🇷 **Türkçe çeviri.** Orijinal İngilizce sürüm: [README.md](https://github.com/rasbt/reasoning-from-scratch/blob/main/ch08/04_train_with_distillation/README.md) · Komut, kod ve sonuç tabloları birebir korunmuştur.

Bu klasör, 8. bölümde ele alındığı gibi Qwen3 0.6B modelini öğretmen (teacher) tarafından üretilmiş akıl yürütme izleri üzerinde eğitmek için basit bir damıtma betiği içerir.

&nbsp;
## Dosyalar

- [distill.py](distill.py): Qwen3 0.6B modelini JSON biçimindeki damıtma verisi üzerinde eğitir (biçim hakkında daha fazlası sonraki bölümde).
  - Varsayılan olarak temel modeli temel tokenizer ile eğitir
  - `--use_think_tokens` verirseniz, akıl yürütme tokenizer'ını kullanır ve 8. bölümde yapıldığına benzer şekilde akıl yürütme izini nihai yanıttan önce `<think>...</think>` içine sarar
  - Her dönemden (epoch) sonra `checkpoints/distill/` klasörüne bir kontrol noktası kaydeder ve eğitim ölçütlerini `logs/distill_metrics.csv` dosyasına ekler
  - Temel model yerine `--checkpoint_path` (isteğe bağlı) ile başlatırsanız, mevcut bir kontrol noktasından devam edebilirsiniz
- [distill_batched.py](distill_batched.py): Yukarıdaki betiğin yığınlanmış (batched) sürümü.
  - Dolgu (padding) farkındalıklı yığınlanmış Qwen3 uygulamasını kullanır; böylece farklı uzunluktaki örnekler birlikte eğitilebilir
  - Optimizasyon adımı başına birden çok örnek işlemek için bir `--batch_size` argümanı ekler
  - Kontrol noktalarını `checkpoints/distill_batched/` klasörüne kaydeder ve ölçütleri `logs/distill_batched_metrics.csv` dosyasına ekler
  - Elbette, yığınlanmış varyantın (yığın boyutuna bağlı olarak) çok daha fazla GPU belleği kullandığını unutmayın

Betik, model yükleme ve istem biçimlendirme kodunu tekrarlamamak için [`reasoning_from_scratch`](../../reasoning_from_scratch) paketinden paylaşılan işlevleri içe aktarır. (Kurulum ayrıntıları için bkz. [bölüm 2 kurulum talimatları](../../ch02/02_setup-tips/python-instructions.md).)


<br>

---

**Not**: `uv` kullanıcısı değilseniz, aşağıdaki örneklerde `uv run ...py` yerine `python ...py` yazın.

---


&nbsp;
## Girdi veri biçimi

Girdi, [`../02_generate_distillation_data`](../02_generate_distillation_data) tarafından üretilen JSON çıktısıdır. Her satır şöyle görünmelidir:

```json
{
  "problem": "Compute 1/2 + 1/6.",
  "gtruth_answer": "2/3",
  "message_thinking": "I will rewrite the fractions with a common denominator.",
  "message_content": "The final answer is \\boxed{\\tfrac{2}{3}}."
}
```

Eğitim için yalnızca şu alanlar kullanılır:

- `problem`: 3. bölümde kullanılan aynı matematik istem şablonuna yerleştirilir
- `message_content`: zorunlu; denetimli (supervised) hedef yanıt olarak kullanılır
- `message_thinking`: isteğe bağlı; varsa `message_content` öncesine eklenir

Eksik veya hatalı biçimli alanlara sahip satırlar otomatik olarak atlanır ve `--max_seq_len` değerinden uzun örnekler eğitim/doğrulama ayrımından önce filtrelenir.


&nbsp;
## Örnek koşu

Hızlı bir sağlık kontrolü (sanity check) için, önceki klasörde üretilen küçük bir örnek üzerinde eğitim yapabilirsiniz:

```bash
uv run distill.py \
  --data_path ../02_generate_distillation_data/sample_openrouter_outputs.json \
  --dataset_size 5 \
  --validation_size 1 \
  --epochs 2 \
  --log_every 1
```

Bu şunları yapacaktır:

- temel Qwen3 0.6B ağırlıklarını yükler
- istem/yanıt çiftlerini token'lara ayırır
- doğrulama için 1 örnek ayırır
- her dönemden sonra `checkpoints/distill/` klasörüne bir kontrol noktası kaydeder
- CSV ölçütlerini `logs/distill_metrics.csv` dosyasına yazar

Bunun yerine açık akıl yürütme etiketleri ve akıl yürütme tokenizer'ı ile eğitmek istiyorsanız `--use_think_tokens` ekleyin:

```bash
uv run distill.py \
  --data_path ../02_generate_distillation_data/sample_openrouter_outputs.json \
  --dataset_size 5 \
  --validation_size 1 \
  --epochs 2 \
  --log_every 1 \
  --use_think_tokens
```

Bunun yerine yığınlar hâlinde eğitmek istiyorsanız şunu çalıştırın:

```bash
uv run distill_batched.py \
  --data_path ../02_generate_distillation_data/sample_openrouter_outputs.json \
  --dataset_size 5 \
  --validation_size 1 \
  --epochs 2 \
  --batch_size 2 \
  --log_every 1
```


&nbsp;
## Faydalı seçenekler

```bash
uv run distill.py --help
```

Önemli argümanlar:

- `--data_path`: damıtma JSON dosyasının yolu
- `--dataset_size`: ayırmadan önce veri kümesini kırpar (`0` tüm satırları kullanır)
- `--validation_size`: doğrulama örneklerinin mutlak sayısı
- `--epochs`: eğitim bölümü üzerinden geçiş sayısı
- `--batch_size`: `distill_batched.py` içinde optimizasyon adımı başına örnek sayısı
- `--lr`: AdamW öğrenme oranı
- `--max_seq_len`: istem + yanıt dizisi bu sınırdan uzun olan örnekleri atar
- `--checkpoint_path`: daha önceki bir damıtma kontrol noktasından başlatır
- `--grad_clip_norm`: isteğe bağlı gradyan kırpma
- `--use_think_tokens`: akıl yürütme tokenizer'ına ve `<think>...</think>` biçimlendirmesine geçer

Uygulamalı örnekler için aşağıdaki "Deneyler" bölümüne bakın.

&nbsp;

## Damıtılmış bir kontrol noktasını değerlendirmek

Eğitimden sonra, 3. bölümdeki değerlendirme betiğini kullanarak bir kontrol noktasını MATH-500 üzerinde değerlendirebilirsiniz.

`--use_think_tokens` olmadan eğittiyseniz, bir `base` model olarak değerlendirin:

```bash
uv run ../../ch03/02_math500-verifier-scripts/evaluate_math500.py \
  --dataset_size 500 \
  --which_model base \
  --checkpoint_path checkpoints/distill/qwen3-0.6B-distill-step00004-epoch1.pth
```

**Önemli:** `--use_think_tokens` ile eğittiyseniz, akıl yürütme tokenizer'ının kullanılması için bir `reasoning` model olarak değerlendirin:

```bash
uv run ../../ch03/02_math500-verifier-scripts/evaluate_math500.py \
  --dataset_size 500 \
  --which_model reasoning \
  --checkpoint_path checkpoints/distill/qwen3-0.6B-distill-step00004-epoch1.pth
```


&nbsp;
## Deneyler

8. bölümde kullanılan damıtma veri kümeleri, Hugging Face deposumdan [rasbt/math_distill](https://huggingface.co/datasets/rasbt/math_distill) adresinden edinilebilir. 8. bölümde bunlar, bölümleri (partition) indiren bir yardımcı fonksiyonla yüklenir, örneğin:

````python
from reasoning_from_scratch.ch08 import load_distill_data

_ = load_distill_data(
    partition="deepseek-r1-math-train.json",
    local_path="deepseek-r1-math-train.json"
)
_ = load_distill_data(
    partition="qwen3-235b-a22b-math-train.json",
    local_path="qwen3-235b-a22b-math-train.json"
)
````



Aşağıdaki deneyler için, o veri kümesi koleksiyonundaki `deepseek-r1-math-train.json` ve `qwen3-235b-a22b-math-train.json` dosyalarını kullandım.


&nbsp;

|      | Öğretmen verisi                      | Dönem | MATH-500 Doğr. | Son doğr. kaybı |
| ---- | ------------------------------------ | ----- | ------------ | -------------- |
| 1    | Base (bölüm 3)                       | -     | 15.2%        | -              |
| 2    | Reasoning (bölüm 3)                  | -     | 48.2%        | -              |
| 3    | DeepSeek R1 damıtma verisi           | 1     | 30.6%        | 0.5436         |
| 4    | DeepSeek R1 damıtma verisi           | 2     | 32.4%        | 0.5349         |
| 5    | DeepSeek R1 damıtma verisi           | 3     | 33.6%        | 0.5343         |
| 6    | Qwen3 235B A22B damıtma verisi       | 1     | 45.0%        | 0.4043         |
| 7    | Qwen3 235B A22B damıtma verisi       | 2     | 43.8%        | 0.3963         |
| 8    | Qwen3 235B A22B damıtma verisi       | 3     | 44.2%        | 0.3948         |

Eğitim bir H100'de yaklaşık 30 dakika, bir DGX Spark'ta yaklaşık 3 saat sürer ve 15 GB'a kadar RAM kullanır.

Aşağıda, tabloda raporlanan sonuçları yeniden üretmek için kod parçaları yer alıyor.

&nbsp;
**1. satır**

```bash
uv run ../../ch03/02_math500-verifier-scripts/evaluate_math500.py \
--dataset_size 500 \
--which_model base
```

&nbsp;
**2. satır**

```bash
uv run ../../ch03/02_math500-verifier-scripts/evaluate_math500.py \
--dataset_size 500 \
--which_model reasoning
```

&nbsp;
**3, 4 ve 5. satırlar**

```bash
uv run distill.py \
--data_path deepseek-r1-math-train.json \
--validation_size 25 \
--epochs 3 \
--lr 1e-5 \
--max_seq_len 2048 \
--use_think_tokens \
--grad_clip 1.0
```

Ardından, dönem kontrol noktalarını değerlendirmek için şunu çalıştırın:

&nbsp;
```bash
uv run ../../ch03/02_math500-verifier-scripts/evaluate_math500.py \
--dataset_size 500 \
--which_model reasoning \
--max_new_tokens 4096 \
--checkpoint_path run-1/checkpoints/distill/qwen3-0.6B-distill-step06682-epoch1.pth
```

4. ve 5. satırlar için kontrol noktası yolunu sırasıyla `...step13364-epoch2.pth` ve `...step20046-epoch3.pth` ile değiştirin.

&nbsp;
**6, 7 ve 8. satırlar**

```bash
uv run distill.py \
--data_path qwen3-235b-a22b-math-train.json \
--validation_size 25 \
--epochs 3 \
--lr 1e-5 \
--max_seq_len 2048 \
--use_think_tokens \
--grad_clip 1.0
```

Ardından, dönem kontrol noktalarını değerlendirmek için şunu çalıştırın:

```bash
uv run ../../ch03/02_math500-verifier-scripts/evaluate_math500.py \
--dataset_size 500 \
--which_model reasoning \
--max_new_tokens 4096 \
--checkpoint_path run_11/checkpoints/distill/qwen3-0.6B-distill-step05746-epoch1.pth
```

7. ve 8. satırlar için kontrol noktası yolunu sırasıyla `...step11492-epoch2.pth` ve `...step17238-epoch3.pth` ile değiştirin.
