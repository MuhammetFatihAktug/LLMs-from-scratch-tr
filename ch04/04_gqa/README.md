# Gruplanmış Sorgu Dikkati (Grouped-Query Attention, GQA)

> 🇹🇷 **Türkçe çeviri.** Orijinal İngilizce sürüm: [README.md](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch04/04_gqa/README.md) · Komut ve çıktı blokları birebir korunmuştur.

Bu bonus materyal, klasik Çok Başlı Dikkat (Multi-Head Attention, MHA) yerine Gruplanmış Sorgu Dikkati (GQA) kullanıldığında elde edilen bellek tasarrufunu gösterir.

&nbsp;
## Giriş

Gruplanmış Sorgu Dikkati (GQA), son yıllarda Çok Başlı Dikkat'e (MHA) kıyasla hesaplama ve parametre açısından daha verimli bir alternatif olarak yeni standart hâline geldi. Yeni bir fikir olmadığını, 2023 tarihli [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) çalışmasına dayandığını belirtelim. Hatta eski dostumuz Llama 2 serisinin daha büyük varyantları bile bunu kullanıyordu.

İşte GQA'nın kısa bir özeti. Her başın kendi anahtar ve değer kümesine de sahip olduğu MHA'dan farklı olarak GQA, bellek kullanımını azaltmak için birden fazla başı aynı anahtar ve değer izdüşümlerini (projection) paylaşacak şekilde gruplandırır.

Örneğin, aşağıdaki şekilde ayrıntılı gösterildiği gibi, 3 anahtar-değer grubu ve 6 dikkat başı varsa, 1. ve 2. başlar bir anahtar-değer kümesini paylaşırken, 3. ve 4. başlar ile 5. ve 6. başlar da sırasıyla başka birer kümeyi paylaşır.

&nbsp;

![GQA](https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/gqa-memory/1.webp?1)

&nbsp;

Anahtar ve değerlerin bu şekilde paylaşılması, toplam anahtar ve değer hesaplama sayısını azaltır; bu da daha düşük bellek kullanımına ve daha yüksek verimliliğe yol açar.

Özetle, GQA'nın arkasındaki temel fikir, anahtar ve değer başlarını birden çok sorgu başı arasında paylaştırarak sayılarını azaltmaktır. Bu (1) modelin parametre sayısını düşürür ve (2) çıkarım sırasında KV önbelleğinde daha az anahtar ve değer saklanıp okunması gerektiği için anahtar ve değer tensörlerine ait bellek bant genişliği kullanımını azaltır.

GQA esas olarak MHA için bir hesaplama verimliliği çözümü olsa da, ablasyon çalışmaları ([orijinal GQA makalesi](https://arxiv.org/abs/2305.13245) ve [Llama 2 makalesi](https://arxiv.org/abs/2307.09288) gibi) LLM modelleme performansı açısından standart MHA ile karşılaştırılabilir sonuçlar verdiğini gösterir.

Ancak bu, anahtar-değer grubu sayısının dikkatlice seçildiğini varsayar. Tüm dikkat başlarının tek bir anahtar-değer grubunu paylaştığı ve çok sorgulu dikkat (multi-query attention) olarak bilinen uç durumda, bellek kullanımı çok daha keskin biçimde düşer ancak modelleme performansı zarar görebilir. (Diğer uçta ise, anahtar-değer grubu sayısını sorgu başı sayısına eşitlersek standart çok başlı dikkate geri dönmüş oluruz.)

&nbsp;
## GQA Bellek Tasarrufu

Bellek tasarrufu esas olarak KV depolamasına yansır. KV depolama boyutunu şu formülle hesaplayabiliriz:

bayt ≈ batch_size × seqlen × (embed_dim / n_heads) × n_layers × 2 (K,V) × eleman_başına_bayt × n_kv_heads

MHA yerine GQA kullanarak ne kadar bellek tasarrufu sağlayabileceğinizi görmek üzere bunu farklı model yapılandırmalarına uygulamak için bu klasördeki [memory_estimator_gqa.py](memory_estimator_gqa.py) betiğini kullanabilirsiniz:

```bash
➜ uv run memory_estimator_gqa.py \
  --emb_dim 4096 --n_heads 32 --n_layers 32 \
  --context_length 32768 --n_kv_groups 4 \
  --batch_size 1 --dtype bf16
==== Config ====
context_length   : 32768
emb_dim          : 4096
n_heads          : 32
n_layers         : 32
n_kv_groups      : 4
batch_size       : 1
dtype            : bf16 (2 Bytes/elem)
head_dim         : 128
GQA n_kv_heads   : 8

==== KV-cache totals across all layers ====
MHA total KV cache  : 17.18 GB
GQA total KV cache  : 4.29 GB
Ratio (MHA / GQA)   : 4.00x
Savings (GQA vs MHA): 75.00%
```

MHA yerine GQA kullanıldığındaki tasarruf, aşağıdaki grafikte farklı anahtar-değer grup boyutları için bağlam uzunluğunun bir fonksiyonu olarak ayrıca gösterilmiştir:

&nbsp;

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/gqa-memory/3.webp?4" alt="GQA" width="500px" />

&nbsp;

Grafiği `uv run plot_memory_estimates_gqa.py` komutuyla yeniden üretebilirsiniz.

&nbsp;
## GQA Kod Örnekleri

Bu klasördeki [gpt_with_kv_mha.py](gpt_with_kv_mha.py) ve [gpt_with_kv_gqa.py](gpt_with_kv_gqa.py) betikleri, bir GPT modeli uygulaması bağlamında MHA ve GQA bellek kullanımını karşılaştırmak için uygulamalı örnekler sunar.

GQA'nın [Llama 3](../../ch05/07_gpt_to_llama), [Gemma 3](../../ch05/12_gemma3) ve [Qwen3](../../ch05/11_qwen3) bonus materyallerinde de kullanıldığını belirtelim. Ancak basitlik adına, bu klasördeki kod betikleri geleneksel olarak GQA kullanmayan GPT mimarisini değiştirir.

Modelin eğitilmediğini ve dolayısıyla anlamsız metin ürettiğini unutmayın. Yine de 5-7. bölümlerdeki standart GPT modelinin yerine doğrudan kullanabilir ve eğitebilirsiniz.

Ayrıca bu uygulama, [başka bir bonus bölümde](../03_kv-cache) açıklanan KV önbelleğini kullanır; böylece bellek tasarrufu daha belirgin hâle gelir.

```bash
uv run gpt_with_kv_mha.py \
--max_new_tokens 32768 \
--n_heads 24 \
--n_layers 12

...

Time: 453.81 sec
72 tokens/sec
Max memory allocated: 1.54 GB
```

```bash
uv run gpt_with_kv_gqa.py \
--max_new_tokens 32768 \
--n_heads 24 \
--n_layers 12 \
--n_kv_groups 4

...

Time: 516.33 sec
63 tokens/sec
Max memory allocated: 0.63 GB
```

Yukarıdaki grafiklerdeki kadar büyük bir tasarruf görmememizin iki nedeni var:

1. Modelin üretimi makul bir sürede bitirmesi için daha küçük bir yapılandırma kullanıyorum.
2. Daha da önemlisi, burada yalnızca dikkat mekanizmasına değil, modelin tamamına bakıyoruz; belleğin çoğunu modeldeki tam bağlantılı katmanlar kaplıyor (ancak bu ayrı bir analizin konusu).
