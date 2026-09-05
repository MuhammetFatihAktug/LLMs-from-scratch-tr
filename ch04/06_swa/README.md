# Kayan Pencere Dikkati (Sliding Window Attention, SWA)

> 🇹🇷 **Türkçe çeviri.** Orijinal İngilizce sürüm: [README.md](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch04/06_swa/README.md) · Komut ve çıktı blokları birebir korunmuştur.

Bu bonus materyal, klasik Çok Başlı Dikkat (MHA) yerine Kayan Pencere Dikkati (SWA) kullanıldığında elde edilen bellek tasarrufunu gösterir.



&nbsp;
## Giriş

Kayan pencere dikkati (SWA) nedir? Klasik öz-dikkati (self-attention) *küresel* (global) bir dikkat mekanizması olarak düşünürsek (çünkü her dizi elemanı diğer tüm dizi elemanlarına erişebilir), SWA'yı *yerel* (local) dikkat olarak düşünebiliriz; çünkü burada bağlam boyutunu mevcut sorgu konumunun etrafıyla sınırlandırırız. Bu, aşağıdaki şekilde gösterilmiştir.

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/swa-memory/1.webp?2" alt="Sliding Window Attention" width="500px" />

Yukarıdaki şekilde görüldüğü gibi, her token önceki tüm token'lara dikkat etmek yerine yalnızca kendi konumunun etrafındaki sabit boyutlu yerel bir pencereye dikkat eder. Bu yerelleştirilmiş dikkat, KV önbelleğinin boyutunu önemli ölçüde düşürür.

Bu girişin geri kalanında SWA'yı, [../../ch05/12_gemma3](../../ch05/12_gemma3) klasöründe sıfırdan uygulanan [Gemma 3](https://arxiv.org/abs/2503.19786) bağlamında ele alacağız.

Kayan pencere dikkati ilk olarak [2020'deki LongFormer makalesinde](https://arxiv.org/abs/2004.05150) tanıtıldı, ancak Google'ın Gemma modellerine odaklanmamızın nedeni, bunların kayan pencere dikkatinin güncel ve yetenekli modellerde gerçekten uygulanabilir bir yaklaşım olduğunu gösteren çok iyi açık ağırlıklı (open-weight) modeller olmasıdır.

[Gemma 2](https://arxiv.org/abs/2408.00118), yerel (kayan pencere) ve küresel dikkat katmanlarını 1:1 oranında birleştiren hibrit bir yaklaşım kullandı. Her token 4 bin token'lık bir bağlam penceresine dikkat edebiliyordu. Bu 1:1 hibrit yaklaşımın nedeni, verimlilik ile küresel bağlam modellemesi arasında bir denge kurmasıdır; çünkü yalnızca yerel dikkat kullanan bir LLM fazla kısıtlayıcı olabilir.

[Gemma 3](https://arxiv.org/abs/2503.19786) ise tasarımı verimlilik yönünde daha da ileri taşıdı. Kayan pencere ve tam dikkat katmanları arasında 5:1 oranı kullandı; yani her beş yerel dikkat katmanına karşılık bir küresel katman var. Ayrıca kayan pencere boyutu Gemma 2'deki 4096 token'dan Gemma 3'te 1024 token'a düşürüldü.

İlginç biçimde, Gemma 3 teknik raporundaki ablasyon çalışmaları bu değişikliklerin genel model kalitesi üzerinde yalnızca küçük bir etkisi olduğunu gösteriyor. Başka bir deyişle, kayan pencere dikkatiyle elde edilen kayda değer bellek ve hesaplama tasarrufu, modelleme performansında çok az kayıpla geliyor.



&nbsp;
## Kayan Pencere Dikkati (SWA) Bellek Tasarrufu

Bellek tasarrufu esas olarak KV depolamasına yansır. KV depolama boyutunu şu formülle hesaplayabiliriz:

bayt ≈ batch_size × seqlen × (embed_dim / n_heads) × n_layers × 2 (K,V) × eleman_başına_bayt × n_kv_heads

SWA kullanırken, yukarıdaki dizi uzunluğunu (seqlen) pencere boyutu W ile değiştiririz. Yani kayan pencere dikkati kullanırken KV önbelleği boyutunu "W / seqlen" çarpanı kadar azaltmış oluruz. (Basitlik adına bunun, kayan pencere dikkatinin her katmanda kullanıldığını varsaydığını unutmayın.)


MHA yerine SWA kullanarak ne kadar bellek tasarrufu sağlayabileceğinizi görmek üzere bunu farklı model yapılandırmalarına uygulamak için bu klasördeki [memory_estimator_swa.py](memory_estimator_swa.py) betiğini kullanabilirsiniz:

```bash
➜ uv run memory_estimator_swa.py \
  --emb_dim 4096 --n_heads 32 --n_layers 32 \
  --context_length 32768 --n_kv_groups 4 \
  --batch_size 1 --dtype bf16 \
  --sliding_window_size 1024 --swa_ratio "5:1"
==== Config ====
context_length         : 32768
sliding_window_size    : 1024
emb_dim                : 4096
n_heads                : 32
n_layers               : 32
n_kv_groups            : 4
batch_size             : 1
dtype                  : bf16 (2 Bytes/elem)
head_dim               : 128
GQA n_kv_heads         : 8
Effective SWA window W : 1024
Layer ratio (SWA:Full) : 5:1
Distributed layers     : 27 SWA, 5 FULL

==== KV-cache totals across all layers ====
MHA KV total           : 17.18 GB
GQA KV total           : 4.29 GB
MHA + SWA (Ratio: 5:1) : 3.14 GB
GQA + SWA (Ratio: 5:1) : 0.78 GB
```

Gemma 3'ün SWA'yı GQA ile birlikte kullandığını unutmayın.

MHA yerine SWA kullanıldığındaki tasarruf, aşağıdaki grafikte farklı bağlam uzunlukları için ayrıca gösterilmiştir:

&nbsp;

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/swa-memory/4.webp?2" alt="SWA" width="800px" />

&nbsp;

Bu grafikleri şu komutla yeniden üretebilirsiniz:

```bash
uv run plot_memory_estimates_swa.py \
  --emb_dim 4096 --n_heads 48 --n_layers 36 \
  --batch_size 1 --dtype bf16 \
  --sliding_window_size 2048 --swa_ratio "5:1"
```


&nbsp;
## SWA Kod Örnekleri

Bu klasördeki [gpt_with_kv_mha.py](gpt_with_kv_mha.py) ve [gpt_with_kv_swa.py](gpt_with_kv_swa.py) betikleri, bir GPT modeli uygulaması bağlamında MHA ve SWA bellek kullanımını karşılaştırmak için uygulamalı örnekler sunar.

SWA'nın (daha önce belirtildiği gibi) MLA ve GQA ile birlikte de kullanılabileceğini, ancak basitlik adına burada bunun yapılmadığını unutmayın.

Modelin eğitilmediğini ve dolayısıyla anlamsız metin ürettiğini unutmayın. Yine de 5-7. bölümlerdeki standart GPT modelinin yerine doğrudan kullanabilir ve eğitebilirsiniz.

Ayrıca bu uygulama, [başka bir bonus bölümde](../03_kv-cache) açıklanan KV önbelleğini kullanır; böylece bellek tasarrufu daha belirgin hâle gelir.

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
uv run gpt_with_kv_swa.py \
--max_new_tokens 32768 \
--n_heads 24 \
--n_layers 12 \
--emb_dim 768 \
--sliding_window_size 1024 \
--sliding_window_stride 5   # like Gemma 3

...

Time: 514.38 sec
63 tokens/sec
Max memory allocated: 0.63 GB
```

Yukarıdaki grafiklerdeki kadar büyük bir tasarruf görmememizin iki nedeni var:

1. Modelin üretimi makul bir sürede bitirmesi için daha küçük bir yapılandırma kullanıyorum.
2. Daha da önemlisi, burada yalnızca dikkat mekanizmasına değil, modelin tamamına bakıyoruz; belleğin çoğunu modeldeki tam bağlantılı katmanlar kaplıyor (ancak bu ayrı bir analizin konusu).
