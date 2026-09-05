# Çok Başlı Gizil Dikkat (Multi-Head Latent Attention, MLA)

> 🇹🇷 **Türkçe çeviri.** Orijinal İngilizce sürüm: [README.md](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch04/05_mla/README.md) · Komut ve çıktı blokları birebir korunmuştur.

Bu bonus materyal, klasik Çok Başlı Dikkat (MHA) yerine Çok Başlı Gizil Dikkat (MLA) kullanıldığında elde edilen bellek tasarrufunu gösterir.

&nbsp;
## Giriş

[../04_gqa](../04_gqa) bölümünde, MHA için bir hesaplama verimliliği çözümü olarak Gruplanmış Sorgu Dikkati'ni (GQA) ele almıştık. Ablasyon çalışmaları ([orijinal GQA makalesi](https://arxiv.org/abs/2305.13245) ve [Llama 2 makalesi](https://arxiv.org/abs/2307.09288) gibi) GQA'nın LLM modelleme performansı açısından standart MHA ile karşılaştırılabilir sonuçlar verdiğini gösteriyor.

[DeepSeek V2, V3 ve R1](https://arxiv.org/abs/2412.19437) tarafından kullanılan Çok Başlı Gizil Dikkat (MLA) ise, KV önbellekleme ile de özellikle iyi uyum sağlayan farklı bir bellek tasarrufu stratejisi sunar. GQA gibi anahtar ve değer başlarını paylaştırmak yerine MLA, anahtar ve değer tensörlerini KV önbelleğinde saklamadan önce daha düşük boyutlu bir uzaya sıkıştırır.

Çıkarım zamanında, aşağıdaki şekilde gösterildiği gibi, bu sıkıştırılmış tensörler kullanılmadan önce orijinal boyutlarına geri izdüşürülür. Bu, fazladan bir matris çarpımı ekler ancak bellek kullanımını azaltır.

&nbsp;

![MLA](https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/mla-memory/1.webp)

&nbsp;

(Yan not olarak, sorgular da sıkıştırılır, ancak yalnızca eğitim sırasında; çıkarımda değil.)

Bu arada, daha önce belirtildiği gibi MLA DeepSeek V3 ile gelen yeni bir şey değildir; [DeepSeek V2 selefi](https://arxiv.org/abs/2405.04434) de bunu kullanmış (hatta tanıtmış) idi. Ayrıca V2 makalesi, DeepSeek ekibinin neden GQA yerine MLA'yı seçtiğini açıklayabilecek birkaç ilginç ablasyon çalışması içerir (aşağıdaki şekle bakınız).

&nbsp;

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/mla-memory/2.webp" alt="GQA" width="500px" />

&nbsp;

Yukarıdaki şekilde görüldüğü gibi, GQA MHA'dan daha kötü performans gösteriyor gibi görünürken, MLA MHA'dan daha iyi modelleme performansı sunuyor; DeepSeek ekibinin GQA yerine MLA'yı seçmesinin nedeni muhtemelen budur. (MLA ile GQA arasındaki "Token Başına KV Önbelleği" tasarrufu karşılaştırmasını da görmek ilginç olurdu!)

Bir sonraki mimari bileşene geçmeden önce bu bölümü özetlemek gerekirse: MLA, modelleme performansı açısından MHA'yı hafifçe geride bırakırken KV önbelleği bellek kullanımını azaltan zekice bir numaradır.

&nbsp;
## MLA Bellek Tasarrufu

Bellek tasarrufu esas olarak KV depolamasına yansır. KV depolama boyutunu şu formülle hesaplayabiliriz:

bayt ≈ batch_size × seqlen × n_layers × latent_dim × eleman_başına_bayt

Buna karşılık MHA KV önbelleği belleği şöyle hesaplanır:

bayt ≈ batch_size × seqlen × n_layers × embed_dim × 2 (K,V) × eleman_başına_bayt

Bu şu anlama gelir: MLA'da, yukarıdaki şekilde gösterildiği gibi tam anahtar ve değer vektörleri yerine yalnızca sıkıştırılmış gizil (latent) temsili sakladığımız için "embed_dim × 2 (K,V)" ifadesini "latent_dim" değerine indiriyoruz.



MHA yerine MLA kullanarak ne kadar bellek tasarrufu sağlayabileceğinizi görmek üzere bunu farklı model yapılandırmalarına uygulamak için bu klasördeki [memory_estimator_mla.py](memory_estimator_mla.py) betiğini kullanabilirsiniz:

```bash
➜ uv run memory_estimator_mla.py \
  --context_length 8192 \
  --emb_dim 2048 \
  --n_heads 24 \
  --n_layers 48 \
  --n_kv_groups 4 \
  --batch_size 1 \
  --dtype bf16 \
  --latent_dim 1024
==== Config ====
context_length   : 8192
emb_dim          : 2048
n_heads          : 24
n_layers         : 48
n_kv_groups      : 4
latent_dim       : 1024
batch_size       : 1
dtype            : bf16 (2 Bytes/elem)
head_dim         : 86
GQA n_kv_heads   : 6

==== KV-cache totals across all layers ====
MHA total KV cache  : 3.25 GB
GQA total KV cache  : 0.81 GB
MLA total KV cache  : 0.81 GB
Ratio (MHA / GQA)   : 4.00x
Savings (GQA vs MHA): 75.00%
Ratio (MHA / MLA)   : 4.03x
Savings (MLA vs MHA): 75.19%
```

Yukarıdaki sıkıştırmanın (`--emb_dim 2048 -> latent_dim 1024`), GQA'dakine benzer bir tasarruf elde etmek için seçildiğini unutmayın. Pratikte sıkıştırma, dikkatle incelenmesi gereken bir hiperparametredir; `latent_dim` değerini çok küçük seçmek modelleme performansını olumsuz etkileyebilir (GQA'da `n_kv_groups` değerini çok yüksek seçmeye benzer şekilde).

MHA yerine MLA kullanıldığındaki tasarruf, aşağıdaki grafikte farklı `latent_dim` değerleri için bağlam uzunluğunun bir fonksiyonu olarak ayrıca gösterilmiştir:

&nbsp;

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/mla-memory/3.webp?2" alt="GQA" width="500px" />

&nbsp;

Grafiği `uv run plot_memory_estimates_mla.py` komutuyla yeniden üretebilirsiniz.



&nbsp;
## MLA Kod Örnekleri

Bu klasördeki [gpt_with_kv_mha.py](gpt_with_kv_mha.py) ve [gpt_with_kv_mla.py](gpt_with_kv_mla.py) betikleri, bir GPT modeli uygulaması bağlamında MHA ve MLA bellek kullanımını karşılaştırmak için uygulamalı örnekler sunar.

Buradaki MLA kodu [https://huggingface.co/bird-of-paradise/deepseek-mla](https://huggingface.co/bird-of-paradise/deepseek-mla) uygulamasından esinlenmiştir.

MLA'nın [GQA](../04_gqa) ile birlikte de kullanılabileceğini, ancak basitlik adına burada bunun yapılmadığını belirtelim. (Şu anda bunu yapan öne çıkan bir LLM de bilmiyorum.)

Ayrıca modelin eğitilmediğini ve dolayısıyla anlamsız metin ürettiğini unutmayın. Yine de 5-7. bölümlerdeki standart GPT modelinin yerine doğrudan kullanabilir ve eğitebilirsiniz.

Son olarak, bu uygulama [başka bir bonus bölümde](../03_kv-cache) açıklanan KV önbelleğini kullanır; böylece bellek tasarrufu daha belirgin hâle gelir.

```bash
uv run gpt_with_kv_mha.py \
--max_new_tokens 32768 \
--n_heads 24 \
--n_layers 12 \
--emb_dim 768

...

Time: 453.81 sec
72 tokens/sec
Max memory allocated: 1.54 GB
```

```bash
uv run gpt_with_kv_mla.py \
--max_new_tokens 32768 \
--n_heads 24 \
--n_layers 12 \
--emb_dim 768 \
--latent_dim 192 # (768×2)/192 = 8× compression

...

Time: 487.21 sec
67 tokens/sec
Max memory allocated: 0.68 GB
```

Yukarıdaki grafiklerdeki kadar büyük bir tasarruf görmememizin iki nedeni var:

1. Modelin üretimi makul bir sürede bitirmesi için daha küçük bir yapılandırma kullanıyorum.
2. Daha da önemlisi, burada yalnızca dikkat mekanizmasına değil, modelin tamamına bakıyoruz; belleğin çoğunu modeldeki tam bağlantılı katmanlar kaplıyor (ancak bu ayrı bir analizin konusu).
