# Optimize Edilmiş Qwen3

> 🇹🇷 **Türkçe çeviri.** Orijinal İngilizce sürüm: [README.md](https://github.com/rasbt/reasoning-from-scratch/blob/main/ch02/03_optimized-LLM/README.md) · Kod blokları ve ölçüm tabloları birebir korunmuştur.

Bu kitapta kullanılan sıfırdan Qwen3 uygulaması, verimli olmak (hem CPU'da hem GPU'da) ile yalın kalmak arasında bir denge kurarken insan tarafından kolayca okunabilir olmayı da sürdürür.

Alternatif olarak, biraz daha GPU-verimli olan isteğe bağlı `Qwen3Model` doğrudan yerine geçen (drop-in) sürümünü kullanabilirsiniz. [`qwen3_optimized.py`](../../reasoning_from_scratch/qwen3_optimized.py) içindeki optimize edilmiş sürüm (Ek C'de daha ayrıntılı ele alınır), [`qwen3.py`](../../reasoning_from_scratch/qwen3.py) içindeki temel uygulamadan iki temel açıdan farklıdır:

- Dikkati özel bir uygulama yerine PyTorch'un yerleşik `torch.nn.functional.scaled_dot_product` fonksiyonuyla uygular.
- Anahtar/değer tensörlerini önceden tahsis eden değiştirilmiş bir `KVCache` sunar. Bu, bellek kullanımını artırır ancak çalışma sırasında tekrar tekrar yeni depolama alanı ayrılmasını önler.


Farkları incelemek için [`qwen3.py`](../../reasoning_from_scratch/qwen3.py) ve [`qwen3_optimized.py`](../../reasoning_from_scratch/qwen3_optimized.py) dosyalarını yan yana açmanızı ve/veya bir dosya farkına (diff) bakmanızı öneririm:

<br>

![](https://sebastianraschka.com/images/reasoning-from-scratch-images/bonus/optimized-LLM/vscode.webp)

<br>

&nbsp;
## Nasıl kullanılır

Optimize edilmiş kod, aşağıda gösterildiği gibi ana bölümlerde kullanılan kodun doğrudan yerine kullanılabilir.

**Önce:**

```python
from reasoning_from_scratch.qwen3 import Qwen3Model
from reasoning_from_scratch.ch02 import generate_text_basic_stream_cache
```


**Sonra:**

```python
from reasoning_from_scratch.qwen3_optimized import Qwen3Model
from reasoning_from_scratch.ch02 import generate_text_basic_stream_cache
```

&nbsp;
## Karşılaştırmalar nasıl çalıştırılır

Sisteminizdeki performansı değerlendirmek için bu klasördeki [`compare_inference.py`](compare_inference.py) fonksiyonunu kullanabilirsiniz:

```python
python compare_inference.py
```

veya

```python
uv run compare_inference.py
```

Ardından şu bayrakları ekleyin:

- `--device`: Cihazı seçer, ör. `cpu`, `mps` veya `cuda`
- `--cache`: KV önbelleğini etkinleştirir
- `--compile`: `torch.compile` kullanır
- `--reasoning`: Temel model yerine Qwen3 akıl yürütme varyantını kullanır. Temel model, verilen isteme yanıt olarak yaklaşık 50 token üretir. Akıl yürütme varyantı yaklaşık 2000 token üretir.
- `--optimize`: `qwen3.py` içindeki standart model yerine `qwen3_optimized.py` içindeki optimize edilmiş modeli kullanır.

<br>

&nbsp;
### Standart model


| Model    | Mod | Komut | Donanım | Token/saniye | GPU Belleği (VRAM) |
| -------- | ----------------- | ------------------------------- | --------------- | ------------- | ----------------- |
| qwen3.py | Regular           | --device cpu                    | Mac Mini M4 CPU | 6             | -                 |
| qwen3.py | Regular compiled  | --device cpu --compile          | Mac Mini M4 CPU | 6             | -                 |
| qwen3.py | KV cache          | --device cpu --cache            | Mac Mini M4 CPU | 28            | -                 |
| qwen3.py | KV cache compiled | --device cpu --compile --cache  | Mac Mini M4 CPU | 68            | -                 |
|          |                   |                                 |                 |               |                   |
| qwen3.py | Regular           | --device mps                    | Mac Mini M4 GPU | 17            | -                 |
| qwen3.py | Regular compiled  | --device mps --compile          | Mac Mini M4 GPU | InductorError | -                 |
| qwen3.py | KV cache          | --device mps --cache            | Mac Mini M4 GPU | 18            | -                 |
| qwen3.py | KV cache compiled | --device mps --compile --cache  | Mac Mini M4 GPU | InductorError | -                 |
|          |                   |                                 |                 |               |                   |
| qwen3.py | Regular           | --device cuda                   | NVIDIA H100 GPU | 51            | 1.55 GB           |
| qwen3.py | Regular compiled  | --device cuda --compile         | NVIDIA H100 GPU | 164           | 1.81 GB           |
| qwen3.py | KV cache          | --device cuda --cache           | NVIDIA H100 GPU | 48            | 1.52 GB           |
| qwen3.py | KV cache compiled | --device cuda --compile --cache | NVIDIA H100 GPU | 141           | 1.81 GB           |

<br>

&nbsp;
### Optimize edilmiş model

| Model              | Mod               | Komut                                       | Donanım         | Token/saniye | GPU Belleği (VRAM) |
| ------------------ | ----------------- | ------------------------------------------- | --------------- | ---------- | ----------------- |
| qwen3_optimized.py | Regular           | --optimized --device cpu                    | Mac Mini M4 CPU | 5          | -                 |
| qwen3_optimized.py | Regular compiled  | --optimized --device cpu --compile          | Mac Mini M4 CPU | 7          | -                 |
| qwen3_optimized.py | KV cache          | --optimized --device cpu --cache            | Mac Mini M4 CPU | 49         | -                 |
| qwen3_optimized.py | KV cache compiled | --optimized --device cpu --compile --cache  | Mac Mini M4 CPU | 51         | -                 |
|                    |                   |                                             |                 |            |                   |
| qwen3_optimized.py | Regular           | --optimized --device mps                    | Mac Mini M4 GPU | 21         | -                 |
| qwen3_optimized.py | Regular compiled  | --optimized --device mps --compile          | Mac Mini M4 GPU | NameError  | -                 |
| qwen3_optimized.py | KV cache          | --optimized --device mps --cache            | Mac Mini M4 GPU | 29         | -                 |
| qwen3_optimized.py | KV cache compiled | --optimized --device mps --compile --cache  | Mac Mini M4 GPU | 38         | -                 |
|                    |                   |                                             |                 |            |                   |
| qwen3_optimized.py | Regular           | --optimized --device cuda                   | NVIDIA H100 GPU | 55         | 1.50 GB           |
| qwen3_optimized.py | Regular compiled  | --optimized --device cuda --compile         | NVIDIA H100 GPU | 173        | 1.81 GB           |
| qwen3_optimized.py | KV cache          | --optimized --device cuda --cache           | NVIDIA H100 GPU | 56         | 5.85 GB           |
| qwen3_optimized.py | KV cache compiled | --optimized --device cuda --compile --cache | NVIDIA H100 GPU | 177        | 5.85 GB           |

<br>

Yukarıdaki 2 tabloyu karşılaştırdığımızda, optimize edilmiş varyantın çoğu durumda token/saniye açısından belirgin biçimde daha hızlı olduğunu görebiliyoruz.

Ancak, KV önbellekli derlenmiş sürüm kullanıldığında optimize edilmemiş sürümün (68 tok/sn) optimize edilmiş sürümden (51 tok/sn) daha hızlı olduğunu unutmayın.

Optimize edilmiş sürüm ayrıca optimize edilmemiş sürüme (1,5 GB) kıyasla daha fazla temel RAM kullanır (KV önbellekli hâlde 5,85 GB). Bunun nedeni, desteklenen maksimum bağlam uzunluğu için KV değerlerini tutan tensörleri önceden tahsis etmesidir. (Yani optimize edilmemiş sürümü 41 bin bağlam uzunluğuna sahip bir istemde çalıştırırsanız, RAM kullanımı yaklaşık olarak benzer olurdu.)

**Belki de en iyi öneri şudur: CPU kullanırken optimize edilmemiş sürümü (`--cache` ve `--compile` ile), GPU kullanırken optimize edilmiş sürümü (`--cache` ve `--compile` ile) tercih edin.**
