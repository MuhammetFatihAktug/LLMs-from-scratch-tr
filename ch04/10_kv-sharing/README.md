# Katmanlar Arası KV Paylaşımı (Cross-Layer KV Sharing)

> 🇹🇷 **Türkçe çeviri.** Orijinal İngilizce sürüm: [README.md](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch04/10_kv-sharing/README.md) · Komut ve çıktı blokları birebir korunmuştur.

Bu bonus materyal, katmanlar arası KV paylaşımının bir KV önbelleğiyle birlikte kullanılması durumunda elde edilen bellek tasarrufunu gösterir.

&nbsp;
## Giriş

[../04_gqa](../04_gqa) klasöründe, birkaç sorgu başının aynı anahtar ve değer başlarını paylaştığı Gruplanmış Sorgu Dikkati'ni (GQA) ele almıştık. Katmanlar arası KV paylaşımı, benzer bir fikri transformer katmanları arasında uygular.

Her katmanda yeni bir anahtar ve değer izdüşümü hesaplamak yerine, sonraki katmanlar daha önceki bir katmandan gelen K/V tensörlerini yeniden kullanır. Yine de kendi sorgularını hesaplarlar, böylece her katman kendi dikkat örüntüsünü oluşturabilir. Ana bellek tasarrufu, önbellekte daha az K/V tensörü saklanmasından gelir.

Bu fikir çapraz katman dikkati (cross-layer attention) olarak da adlandırılır. Brandon *ve ark.*, [Reducing Transformer Key-Value Cache Size with Cross-Layer Attention](https://arxiv.org/abs/2405.12981) çalışmasında açıklanmıştır. Gemma 4 E2B ve E4B, buna benzer paylaşılan bir KV önbelleği şeması kullanır; bu da onu bu bölümdeki GQA, MLA ve SWA örneklerine faydalı bir ek yapar.

&nbsp;

<img src="gemma4-kv-sharing.webp" alt="Cross-layer KV sharing" width="800px" />

&nbsp;

[Gemma 4](../../ch05/17_gemma4) modelinde KV paylaşımı, GQA veya MQA ve kayan pencere dikkatiyle birleştirilir. Bu klasördeki basitleştirilmiş GPT örneğinde yalnızca katmanlar arası KV paylaşımı kısmını uyguluyoruz; böylece kod ana mekanizmaya odaklı kalıyor.

Burada kullanılan basitleştirilmiş kural şudur:

1. Erken katmanlar kendi K/V tensörlerini hesaplar ve önbelleğe alır.
2. Sonraki katmanlar, daha önceki bir üretici katmandan gelen en güncel K/V tensörlerini yeniden kullanır.
3. Tüm katmanlar yine de kendi sorgu izdüşümlerini hesaplar.

Bu, bağlam uzunluğuyla birlikte büyüyen K/V önbelleklerinin sayısını azaltır. Ödünleşim ise, bazı katmanların artık kendi K/V izdüşümlerine sahip olmaması nedeniyle model kapasitesinin azalmasıdır.

&nbsp;
## KV Paylaşımı Bellek Tasarrufu

Olağan KV önbelleği belleği şöyle hesaplanır:

bayt = batch_size x seqlen x head_dim x n_kv_heads x n_layers x 2 (K,V) x eleman_başına_bayt

Katmanlar arası KV paylaşımıyla, `n_layers` yerine K/V üreten katman sayısını koyarız:

bayt = batch_size x seqlen x head_dim x n_kv_heads x n_kv_producing_layers x 2 (K,V) x eleman_başına_bayt

Bunu farklı model yapılandırmalarına uygulamak için bu klasördeki [memory_estimator_kv_sharing.py](memory_estimator_kv_sharing.py) betiğini kullanabilirsiniz:

```bash
# Gemma 4 E2B-like setup
uv run memory_estimator_kv_sharing.py \
  --context_length 131072 \
  --emb_dim 2048 \
  --n_heads 8 \
  --n_layers 35 \
  --n_kv_groups 8 \
  --n_kv_producing_layers 15 \
  --batch_size 1 \
  --dtype bf16

# Gemma 4 E4B-like setup
# uv run memory_estimator_kv_sharing.py \
#   --context_length 131072 \
#   --emb_dim 2560 \
#   --n_heads 8 \
#   --n_layers 42 \
#   --n_kv_groups 4 \
#   --n_kv_producing_layers 24 \
#   --batch_size 1 \
#   --dtype bf16

==== Config ====
context_length         : 131072
emb_dim                : 2048
n_heads                : 8
n_layers               : 35
n_kv_groups            : 8
n_kv_producing_layers  : 15
batch_size             : 1
dtype                  : bf16 (2 Bytes/elem)
head_dim               : 256
GQA n_kv_heads         : 1

==== KV-cache totals across all layers ====
MHA total KV cache        : 37.58 GB
GQA total KV cache        : 4.70 GB
MHA + KV sharing          : 16.11 GB
GQA + KV sharing          : 2.01 GB
Ratio (MHA / GQA+sharing) : 18.67x
Savings vs MHA            : 94.64%
```

Bu, Gemma 4 E2B benzeri bir kurulumdur. 35 katmanın 15'i K/V üreten katmandır; kalan katmanlar daha önceki K/V tensörlerini yeniden kullanır. E4B benzeri kurulum için karşılık gelen sayılar toplam 42 katman ve 24 K/V üreten katmandır.

Tasarruflar aşağıda E2B ve E4B benzeri kurulumlar için gösterilmiştir. Basitlik adına bu grafikler, kayan pencere dikkatinden gelen ek tasarrufları içermez.

&nbsp;

<img src="kv_memory_mha_gqa_kvsharing_gemma4_e2b.webp" alt="KV-sharing memory savings for Gemma 4 E2B-like setup" width="800px" />

&nbsp;

<img src="kv_memory_mha_gqa_kvsharing_gemma4_e4b.webp" alt="KV-sharing memory savings for Gemma 4 E4B-like setup" width="800px" />

&nbsp;

Benzer grafikleri şu komutlarla yeniden üretebilirsiniz:

```bash
uv run plot_memory_estimates_kv_sharing.py --preset gemma4_e2b
uv run plot_memory_estimates_kv_sharing.py --preset gemma4_e4b
```

&nbsp;
## KV Paylaşımı Kod Örnekleri

Bu klasördeki [gpt_with_kv_mha.py](gpt_with_kv_mha.py) ve [gpt_with_kv_sharing.py](gpt_with_kv_sharing.py) betikleri, klasik MHA ile katmanlar arası KV paylaşımı varyantını karşılaştırmak için uygulamalı örnekler sunar.

Uygulama ayrıntılarını görmenin en kolay yolu, [gpt_with_kv_mha.py](gpt_with_kv_mha.py) ile [gpt_with_kv_sharing.py](gpt_with_kv_sharing.py) arasındaki dosya farkını (diff) incelemektir. Yorumlar, farkın KV paylaşımı değişikliklerini öne çıkarması için bilinçli olarak benzer tutulmuştur.

Modelin eğitilmediğini ve dolayısıyla anlamsız metin ürettiğini unutmayın. Yine de 5-7. bölümlerdeki standart GPT modelinin yerine doğrudan kullanabilir ve eğitebilirsiniz.

Ayrıca bu uygulama, [başka bir bonus bölümde](../03_kv-cache) açıklanan KV önbelleğini kullanır; böylece bellek tasarrufu daha belirgin hâle gelir.

```bash
uv run gpt_with_kv_mha.py \
--max_new_tokens 32768 \
--n_heads 24 \
--n_layers 12 \
--emb_dim 768
```

```bash
uv run gpt_with_kv_sharing.py \
--max_new_tokens 32768 \
--n_heads 24 \
--n_layers 12 \
--emb_dim 768 \
--n_kv_producing_layers 6
```

Bu küçük GPT kurulumunda modelin tamamı yine aynı ileri beslemeli katmanları ve çıkış başını içerir. Ana bellek farkı, kaç dikkat katmanının önbellekte K/V tensörü sakladığındadır.
