# Uzmanlar Karışımı (Mixture of Experts, MoE)

> 🇹🇷 **Türkçe çeviri.** Orijinal İngilizce sürüm: [README.md](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch04/07_moe/README.md) · Komut ve çıktı blokları birebir korunmuştur.

Bu bonus materyal, klasik ileri beslemeli (feed-forward, FFN) katmanlar yerine Uzmanlar Karışımı (MoE) katmanları kullanıldığında (token başına) elde edilen bellek tasarrufunu gösterir.



&nbsp;
## Giriş

MoE'deki temel fikir, bir transformer bloğundaki her ileri beslemeli modülü birden çok uzman katmanıyla değiştirmektir; bu uzman katmanlarının her biri de birer ileri beslemeli modüldür. Yani aşağıdaki şekilde gösterildiği gibi, tek bir ileri beslemeli bloğu birden çok ileri beslemeli blokla değiştiriyoruz.



&nbsp;

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/moe-memory/1.webp" alt="SWA" width="800px" />

Bir transformer bloğunun içindeki ileri beslemeli blok (yukarıdaki şekilde koyu gri blok olarak gösterilmiştir) tipik olarak modelin toplam parametrelerinin büyük bir kısmını barındırır. (Transformer bloğunun ve dolayısıyla ileri beslemeli bloğun bir LLM'de defalarca tekrarlandığını unutmayın; DeepSeek-V3 örneğinde 61 kez.)

Dolayısıyla, *tek bir* ileri beslemeli bloğu *birden çok* ileri beslemeli blokla değiştirmek (bir MoE kurulumunda yapıldığı gibi) modelin toplam parametre sayısını ciddi biçimde artırır. Ancak asıl numara, her token için tüm uzmanları kullanmamamızdır ("etkinleştirmememizdir"). Bunun yerine bir yönlendirici (router), token başına yalnızca küçük bir uzman alt kümesini seçer.

Aynı anda yalnızca birkaç uzman etkin olduğu için, MoE modülleri sıklıkla *seyrek* (sparse) olarak adlandırılır; buna karşılık her zaman parametre kümesinin tamamını kullanan modüller *yoğun* (dense) olarak anılır. Bununla birlikte, MoE üzerinden gelen yüksek toplam parametre sayısı LLM'in kapasitesini artırır; yani eğitim sırasında daha fazla bilgi barındırabilir. Yine de seyreklik, tüm parametreleri aynı anda kullanmadığımız için çıkarımı verimli tutar.

Örneğin DeepSeek-V3'te MoE modülü başına 256 uzman ve toplam 671 milyar parametre vardır. Yine de çıkarım sırasında aynı anda yalnızca 9 uzman etkindir (1 paylaşılan uzman artı yönlendiricinin seçtiği 8 uzman). Bu, her token çıkarım adımında 671 milyarın tamamı yerine yalnızca 37 milyar parametrenin kullanıldığı anlamına gelir.

DeepSeek-V3'ün MoE tasarımının dikkat çekici bir özelliği, paylaşılan bir uzman (shared expert) kullanmasıdır. Bu, her token için daima etkin olan bir uzmandır. Bu fikir yeni değildir; [2022 DeepSpeed-MoE](https://arxiv.org/abs/2201.05596) ve [2024 DeepSeek MoE](https://arxiv.org/abs/2401.06066) makalelerinde zaten tanıtılmıştı.

&nbsp;

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/moe-memory/3.webp?1" alt="MoE shared expert" width="500px" />

([DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models](https://arxiv.org/abs/2401.06066) makalesinden açıklamalı bir şekil.)

&nbsp;

Paylaşılan bir uzmana sahip olmanın faydası ilk olarak [DeepSpeed-MoE makalesinde](https://arxiv.org/abs/2201.05596) fark edildi; burada, paylaşılan uzman olmayan duruma kıyasla genel modelleme performansını artırdığı görüldü. Bunun nedeni muhtemelen, yaygın veya tekrar eden örüntülerin birden çok ayrı uzman tarafından öğrenilmek zorunda kalmaması ve böylece bu uzmanlara daha özelleşmiş örüntüleri öğrenmek için daha fazla alan kalmasıdır.

&nbsp;
## Uzmanlar Karışımı (MoE) Bellek Tasarrufu

MoE modellerindeki bellek tasarrufu esas olarak azalan aktivasyon depolaması ve hesaplamadan gelir. Klasik (yoğun) bir ileri beslemeli katmanda (FFN) her token, ara boyutun tamamını etkinleştirir.

Buna karşılık bir MoE katmanı, her token'ı yalnızca küçük bir uzman alt kümesi üzerinden yönlendirir (örneğin, `num_experts` içinden `top_k` kadarı).

Bir MoE katmanı kullanırken token başına yalnızca `top_k` uzman etkin olur; dolayısıyla etkin bellek (ve hesaplama), aynı toplam kapasiteye sahip yoğun bir FFN'e kıyasla kabaca `top_k / num_experts` çarpanıyla ölçeklenir.


FFN yerine MoE kullanarak ne kadar bellek tasarrufu sağlayabileceğinizi görmek üzere bunu farklı model yapılandırmalarına uygulamak için bu klasördeki [memory_estimator_moe.py](memory_estimator_moe.py) betiğini kullanabilirsiniz (bunun tek bir transformer bloğu için olduğunu, toplam tasarrufu bulmak için modelinizdeki transformer bloğu sayısıyla çarpmanız gerektiğini unutmayın):

```bash
uv run memory_estimator_moe.py --emb_dim 7168 --hidden_dim 14336 --ffn_type swiglu \
  --num_experts 8 --top_k 2 --match_dense 
==== Config ====
emb_dim                : 7168
hidden_size            : 14336
ffn_type               : swiglu
num_experts            : 8
top_k                  : 2
dtype                  : bf16 (2 Bytes/elem)
match_dense            : True

==== Model weights (parameters) ====
Dense FFN params       : 308,281,344 (0.62 GB)
Per-expert params      : 38,535,168 (0.08 GB)
Router params          : 57,344 (0.00 GB)
MoE TOTAL params       : 308,338,688 (0.62 GB)
MoE ACTIVE/Token       : 77,127,680 (0.15 GB)
moe_hidden_size        : 1792
```

Yukarıdaki sonuçlara dayanarak görebiliriz ki, girdi/çıktı boyutu (`emb_dim`) 7.168 ve ara boyutu (`hidden_dim`) 14.336 olan bir FFN'imiz varsa, bu katmanda yaklaşık 308 milyon parametre bulunur ve bu parametrelerin tamamı ileri geçişte etkindir.

Şimdi, kabaca aynı toplam parametre sayısına (~308M) sahip, 8 uzmanlı ve 2 uzmanın etkin olduğu bir MoE katmanı kullanırsak, her ileri geçişte yalnızca ~77M parametre etkin olur.

Dahası, sabit sayıda uzman için, ne kadar çok uzmanımız olursa etkin parametre sayısı o kadar düşer ve "tasarruf" o kadar artar:

&nbsp;

&nbsp;

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/moe-memory/2.webp" alt="SWA" width="500px" />



&nbsp;

Bu grafiği şu komutla yeniden üretebilirsiniz:

```bash
uv run plot_memory_estimates_moe.py \
    --emb_dim 7168 \
    --hidden_dim 28672 \
    --ffn_type swiglu \
    --top_k 8
```


&nbsp;
## MoE Kod Örnekleri

Bu klasördeki [gpt_with_kv_ffn.py](gpt_with_kv_ffn.py) ve [gpt_with_kv_moe.py](gpt_with_kv_moe.py) betikleri, bir GPT modeli uygulaması bağlamında klasik FFN ile MoE bellek kullanımını karşılaştırmak için uygulamalı örnekler sunar. Her iki betiğin de bu sayfanın ilk şeklinde gösterildiği gibi [SwiGLU](https://arxiv.org/abs/2002.05202) ileri beslemeli modüllerini kullandığını unutmayın (GPT-2 geleneksel olarak GELU kullanır).

**Not: Model eğitilmemiştir ve dolayısıyla anlamsız metin üretir. Eğitilmiş bir MoE'yi [../../ch05/11_qwen3/standalone-qwen3-moe-plus-kvcache.ipynb](../../ch05/11_qwen3/standalone-qwen3-moe-plus-kvcache.ipynb) bonus materyalinde bulabilirsiniz.**



Önce modeli klasik bir FFN ile çalıştıralım:


```bash
uv run gpt_with_kv_ffn.py \
--max_new_tokens 1024 \
--n_heads 16 \
--n_layers 12 \
--emb_dim 4096 \
--hidden_dim 32768

...
Avg FFN time/call: 0.759 ms
Avg FFN mem delta/call: 0.19 MB (max 0.75 MB)
...
Time: 25.13 sec
40 tokens/sec
Max memory allocated: 11.47 GB
```

MoE ile adil bir karşılaştırma için uzman boyutunu küçültmemiz gerekir. Örneğin 32 uzman kullanırsak `--hidden_dim 32768/32` ayarlamalıyız:


```bash
uv run gpt_with_kv_moe.py \
--max_new_tokens 1024 \
--n_heads 16 \
--n_layers 12 \
--emb_dim 4096 \
--hidden_dim 1024 \
--num_experts 32 \
--num_experts_per_tok 2

...
Avg MoE FF time/call: 1.555 ms
Avg MoE FF mem delta/call: 0.04 MB (max 0.11 MB)
...
Time: 35.11 sec
29 tokens/sec
Max memory allocated: 11.48 GB
```

Görebiliyoruz ki yoğun ileri beslemeli katman bir token'ı yaklaşık 0,76 ms'de işliyor ve kabaca 0,19 MB aktivasyon kullanıyor (tepe noktası 0,75 MB civarında).

Seyrek MoE katmanı ise yalnızca yaklaşık 0,04 MB bellek tutuyor (tepe noktası 0,11 MB). Ancak bu, kabaca iki katı hesaplama süresi pahasına geliyor. (Ek bir yönlendirme yükü var ve benim uygulamam en verimli uygulama olmayabilir.)

Genel üretim yine de her iki durumda da yaklaşık 11,5 GB GPU belleğinde zirve yapıyor; çünkü her iki sürüm de aynı sayıda ağırlık parametresi yüklüyor ve aynı KV önbelleği boyutuna sahip; burada baskın olan bunlar.

Her hâlükârda, buradaki ödünleşimi görebiliyoruz: MoE, FFN belleğini yaklaşık 4-5 kat azaltırken ileri besleme hesaplama süresini kabaca ikiye katlıyor.

Aynı anda daha fazla token işleseydik, örneğin 1'den büyük bir yığın boyutuyla (burada kod basitliği nedeniyle yığın kullanmıyoruz), tasarrufun daha belirgin olacağını unutmayın.
