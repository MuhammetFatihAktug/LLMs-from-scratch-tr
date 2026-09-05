# Bonus Materyal: KV Önbelleği (KV Cache)

> 🇹🇷 **Türkçe çeviri.** Orijinal İngilizce sürüm: [README.md](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch04/03_kv-cache/README.md) · Kod blokları, gerçek kaynak dosyalarla birebir eşleşmesi için çevrilmeden bırakılmıştır.



**Bu klasör, GPT modeline bir KV önbelleği eklenmesini uygular.**

&nbsp;
## Genel Bakış

Kısaca, bir KV önbelleği, çıkarım (inference) sırasında yeniden kullanılmak üzere ara anahtar (key, K) ve değer (value, V) hesaplamalarını saklar; bu da yanıt üretirken kayda değer bir hızlanma sağlar. Dezavantajı ise koda biraz karmaşıklık eklemesi, bellek kullanımını artırması ve eğitim sırasında kullanılamamasıdır. Yine de, LLM'leri dağıtıma alırken (deploy) elde edilen çıkarım hızlanmaları, kod karmaşıklığı ve bellek açısından yapılan bu takasa çoğunlukla fazlasıyla değer.

&nbsp;
## Nasıl çalışır

LLM'in bir metin ürettiğini düşünün. Somut olarak, LLM'e şu istemin (prompt) verildiğini varsayalım: "Time flies".

Aşağıdaki şekil, 3. bölümden alınan ve anahtar ile değer vektörlerinin vurgulandığı değiştirilmiş bir grafikle, arka plandaki dikkat skoru hesaplamasından bir kesit gösterir:

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/kv-cache/kv-cache-attn-1.png?3" width=800>

Şimdi, 2. ve 4. bölümlerde öğrendiğimiz gibi, LLM'ler her seferinde bir kelime (veya token) üretir. LLM'in "fast" kelimesini ürettiğini ve böylece bir sonraki turun isteminin "Time flies fast" hâline geldiğini varsayalım. Bu, aşağıdaki bir sonraki şekilde gösterilmiştir:

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/kv-cache/kv-cache-attn-2.png?3" width=800>

Önceki 2 şekli karşılaştırdığımızda görebileceğimiz gibi, ilk iki token'a ait anahtar ve değer vektörleri tamamen aynıdır; bunları her yeni token üretim turunda yeniden hesaplamak israf olurdu.

Dolayısıyla KV önbelleğinin fikri, daha önce üretilmiş anahtar ve değer vektörlerini yeniden kullanılmak üzere saklayan bir önbellekleme mekanizması uygulamaktır; bu da gereksiz yeniden hesaplamalardan kaçınmamızı sağlar.

&nbsp;

## KV önbelleği uygulaması

Bir KV önbelleğini uygulamanın birçok yolu vardır; ana fikir, her üretim adımında yalnızca yeni üretilen token'lar için anahtar ve değer tensörlerini hesaplamamızdır.

Ben kod okunabilirliğini ön plana çıkaran basit bir yaklaşım tercih ettim. Nasıl uygulandığını görmek için kod değişikliklerini baştan sona incelemenin en kolay yol olduğunu düşünüyorum.

Bu klasörde iki dosya bulunur:

1. [`gpt_ch04.py`](gpt_ch04.py): LLM'i uygulamak ve basit metin üretme fonksiyonunu çalıştırmak için 3. ve 4. bölümlerden alınmış, kendi kendine yeten (self-contained) kod
2. [`gpt_with_kv_cache.py`](gpt_with_kv_cache.py): Yukarıdakinin aynısı, ancak KV önbelleğini uygulamak için gerekli değişiklikler yapılmış hâli.

Şunlardan birini yapabilirsiniz:

a. [`gpt_with_kv_cache.py`](gpt_with_kv_cache.py) dosyasını açıp yeni değişiklikleri işaretleyen `# NEW` bölümlerine bakın:

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/kv-cache/new-sections.png?3" width=800>

b. Değişiklikleri karşılaştırmak için iki kod dosyasını tercih ettiğiniz bir dosya karşılaştırma (diff) aracıyla inceleyin:

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/kv-cache/file-diff.png?3" width=800>

Uygulama ayrıntılarını özetlemek için kısa bir gezinti:

&nbsp;

### 1. Önbellek buffer'larını kaydetmek

`MultiHeadAttention` yapıcısının (constructor) içine, adımlar boyunca birleştirilmiş anahtar ve değerleri tutacak olan `cache_k` ve `cache_v` adlı iki buffer ekliyoruz:

```python
self.register_buffer("cache_k", None)
self.register_buffer("cache_v", None)
```

&nbsp;

### 2. `use_cache` bayrağıyla ileri geçiş

Ardından, `MultiHeadAttention` sınıfının `forward` metodunu `use_cache` argümanını kabul edecek şekilde genişletiyoruz. Yeni token yığınını `keys_new`, `values_new` ve `queries` üzerine izdüşürdükten sonra ya kv önbelleğini başlatırız ya da önbelleğimize ekleme yaparız:

```python
def forward(self, x, use_cache=False):
    b, num_tokens, d_in = x.shape

    keys_new = self.W_key(x)  # Shape: (b, num_tokens, d_out)
    values_new = self.W_value(x)
    queries = self.W_query(x)
    #...

    if use_cache:
        if self.cache_k is None:
            self.cache_k, self.cache_v = keys_new, values_new
        else:
            self.cache_k = torch.cat([self.cache_k, keys_new], dim=1)
            self.cache_v = torch.cat([self.cache_v, values_new], dim=1)
        keys, values = self.cache_k, self.cache_v
    else:
        keys, values = keys_new, values_new
        
    # ...
    
    num_tokens_Q = queries.shape[-2]
    num_tokens_K = keys.shape[-2]
    if use_cache:
        mask_bool = self.mask.bool()[
            self.ptr_current_pos:self.ptr_current_pos + num_tokens_Q, :num_tokens_K
        ]
        self.ptr_current_pos += num_tokens_Q
    else:
        mask_bool = self.mask.bool()[:num_tokens_Q, :num_tokens_K]
```

&nbsp;


### 3. Önbelleği temizlemek

Metin üretirken, birbirinden bağımsız diziler arasında (örneğin ayrı metin üretme çağrıları arasında) her iki buffer'ı da sıfırlamamız gerekir; bu nedenle `MultiHeadAttention` sınıfına bir önbellek sıfırlama metodu da ekliyoruz:

```python
def reset_cache(self):
    self.cache_k, self.cache_v = None, None
    self.ptr_current_pos = 0
```

&nbsp;

### 4. `use_cache` bayrağını modelin tamamına yaymak

`MultiHeadAttention` sınıfındaki değişiklikler yerine oturduğuna göre, şimdi `GPTModel` sınıfını değiştiriyoruz. Önce, yapıcıya token indeksleri için bir konum takibi ekliyoruz:

```python
self.current_pos = 0
```

Ardından, tek satırlık blok çağrısını açık bir döngüyle değiştirip `use_cache` bayrağını her transformer bloğuna aktarıyoruz:

```python
def forward(self, in_idx, use_cache=False):
    # ...
 
    if use_cache:
        pos_ids = torch.arange(
            self.current_pos, self.current_pos + seq_len,            
            device=in_idx.device, dtype=torch.long
        )
        self.current_pos += seq_len
    else:
        pos_ids = torch.arange(
            0, seq_len, device=in_idx.device, dtype=torch.long
        )
    
    pos_embeds = self.pos_emb(pos_ids).unsqueeze(0)
    x = tok_embeds + pos_embeds
    # ...
    for blk in self.trf_blocks:
        x = blk(x, use_cache=use_cache)
```

Yukarıdaki değişiklik, `TransformerBlock` sınıfının da `use_cache` argümanını kabul etmesi için küçük bir değişiklik gerektirir:
```python
    def forward(self, x, use_cache=False):
        # ...
        self.att(x, use_cache=use_cache)
```

Son olarak, kolaylık olsun diye tüm blok önbelleklerini tek seferde temizleyen model düzeyinde bir sıfırlama metodunu `GPTModel` sınıfına ekliyoruz:

```python
def reset_kv_cache(self):
    for blk in self.trf_blocks:
        blk.att.reset_cache()
    self.current_pos = 0
```

&nbsp;

### 5. Önbelleği üretimde kullanmak

`GPTModel`, `TransformerBlock` ve `MultiHeadAttention` sınıflarındaki değişikliklerle birlikte, KV önbelleğini basit bir metin üretme fonksiyonunda şöyle kullanıyoruz:

```python
def generate_text_simple_cached(model, idx, max_new_tokens, 
                                context_size=None, use_cache=True):
    model.eval()
    ctx_len = context_size or model.pos_emb.num_embeddings

    with torch.no_grad():
        if use_cache:
            # Init cache with full prompt
            model.reset_kv_cache()
            logits = model(idx[:, -ctx_len:], use_cache=True)

            for _ in range(max_new_tokens):
                # a) pick the token with the highest log-probability (greedy sampling)
                next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                # b) append it to the running sequence
                idx = torch.cat([idx, next_idx], dim=1)
                # c) feed model only the new token
                logits = model(next_idx, use_cache=True)
        else:
            for _ in range(max_new_tokens):
                logits = model(idx[:, -ctx_len:], use_cache=False)
                next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                idx = torch.cat([idx, next_idx], dim=1)

    return idx
```

Dikkat edin: c) adımında `logits = model(next_idx, use_cache=True)` ile modele yalnızca yeni token'ı veriyoruz. Önbellekleme olmadan ise modele tüm girdiyi veririz (`logits = model(idx[:, -ctx_len:], use_cache=False)`), çünkü yeniden kullanabileceği saklanmış anahtar ve değerler yoktur.

&nbsp;

## Basit bir performans karşılaştırması

KV önbelleğini kavramsal düzeyde ele aldıktan sonra, akla gelen büyük soru şu: küçük bir örnekte pratikte gerçekten ne kadar iyi çalışıyor? Uygulamayı denemek için yukarıda bahsedilen iki kod dosyasını Python betiği olarak çalıştırabiliriz; bu, 124 milyon parametreli küçük LLM'i çalıştırarak 200 yeni token üretecektir (başlangıç olarak 4 token'lık "Hello, I am" istemi verilir):

```bash
pip install -r https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/refs/heads/main/requirements.txt

python gpt_ch04.py

python gpt_with_kv_cache.py
```

M4 çipli bir Mac Mini'de (CPU) sonuçlar şöyledir:

|                        | Token/saniye |
| ---------------------- | ---------- |
| `gpt_ch04.py`          | 27         |
| `gpt_with_kv_cache.py` | 144        |

Görüldüğü gibi, 124 milyon parametreli küçük bir model ve 200 token'lık kısa bir dizi uzunluğuyla bile yaklaşık 5 kat hızlanma elde ediyoruz. (Bu uygulamanın kod okunabilirliği için optimize edildiğini, CUDA veya MPS çalışma zamanı hızı için optimize edilmediğini unutmayın; bunun için tensörleri yeniden oluşturup birleştirmek yerine önceden ayırmak gerekirdi.)

**Not:** Model her iki durumda da "anlamsız" metin üretir, yani şuna benzer bir çıktı verir:

> Output text: Hello, I am Featureiman Byeswickattribute argue logger Normandy Compton analogous bore ITVEGIN ministriesysics Kle functional recountrictionchangingVirgin embarrassedgl ...

Bunun nedeni modeli henüz eğitmemiş olmamızdır. Sonraki bölüm modeli eğitir ve tutarlı metin üretmek için KV önbelleğini eğitilmiş model üzerinde kullanabilirsiniz (yine de KV önbelleği yalnızca çıkarım sırasında kullanılmak üzere tasarlanmıştır). Burada kodu (daha) basit tutmak için eğitilmemiş modeli kullanıyoruz.

Ancak daha önemlisi, `gpt_ch04.py` ve `gpt_with_kv_cache.py` uygulamalarının tamamen aynı metni üretmesidir. Bu bize KV önbelleğinin doğru uygulandığını söyler; birbirinden farklı sonuçlara yol açabilecek indeksleme hataları yapmak oldukça kolaydır.


&nbsp;

## KV önbelleğinin avantajları ve dezavantajları

Dizi uzunluğu arttıkça, bir KV önbelleğinin faydaları ve sakıncaları aşağıdaki şekillerde daha belirgin hâle gelir:

- [İyi] **Hesaplama verimliliği artar**: Önbellekleme olmadan, *t* adımındaki dikkat mekanizması yeni sorguyu *t* adet önceki anahtarla karşılaştırmak zorundadır; dolayısıyla toplam iş yükü karesel olarak, O(n²) büyür. Önbellekle birlikte her anahtar ve değer bir kez hesaplanıp sonra yeniden kullanılır ve adım başına toplam karmaşıklık doğrusala, O(n)'e iner.

- [Kötü] **Bellek kullanımı doğrusal olarak artar**: Her yeni token KV önbelleğine eklenir. Uzun diziler ve daha büyük LLM'ler için biriken KV önbelleği büyür ve ciddi, hatta karşılanamaz miktarda (GPU) bellek tüketebilir. Geçici çözüm olarak KV önbelleğini kırpabiliriz, ancak bu daha da fazla karmaşıklık ekler (yine de, LLM'leri dağıtıma alırken buna değebilir).



&nbsp;
## KV Önbelleği Uygulamasını Optimize Etmek

Yukarıdaki kavramsal KV önbelleği uygulamam anlaşılırlığa yardımcı olsa ve esas olarak kod okunabilirliği ile eğitim amaçlarına yönelik olsa da, gerçek dünya senaryolarında (özellikle daha büyük modeller ve daha uzun dizi uzunluklarıyla) dağıtıma almak daha dikkatli bir optimizasyon gerektirir.

&nbsp;
### Önbelleği ölçeklerken sık karşılaşılan tuzaklar

- **Bellek parçalanması ve tekrar eden tahsisler**: Daha önce gösterildiği gibi `torch.cat` ile sürekli tensör birleştirmek, sık bellek tahsisi ve yeniden tahsisi nedeniyle performans darboğazlarına yol açar.

- **Bellek kullanımında doğrusal büyüme**: Uygun şekilde ele alınmazsa, çok uzun diziler için KV önbelleği boyutu pratik olmaktan çıkar.

&nbsp;
#### İpucu 1: Belleği önceden tahsis edin

Tensörleri tekrar tekrar birleştirmek yerine, beklenen maksimum dizi uzunluğuna göre yeterince büyük bir tensörü önceden tahsis edebiliriz. Bu, tutarlı bellek kullanımı sağlar ve ek yükü azaltır. Sözde kod (pseudo-code) olarak şöyle görünebilir:

```python
# Example pre-allocation for keys and values
max_seq_len = 1024  # maximum expected sequence length
cache_k = torch.zeros((batch_size, num_heads, max_seq_len, head_dim), device=device)
cache_v = torch.zeros((batch_size, num_heads, max_seq_len, head_dim), device=device)
```

Çıkarım sırasında ise bu önceden tahsis edilmiş tensörlerin dilimlerine (slice) yazabiliriz.

&nbsp;
#### İpucu 2: Önbelleği kayan pencere ile kırpın

GPU belleğimizi patlatmamak için dinamik kırpmalı bir kayan pencere (sliding window) yaklaşımı uygulayabiliriz. Kayan pencere sayesinde önbellekte yalnızca son `window_size` kadar token'ı tutarız:


```python
# Sliding window cache implementation
window_size = 512
cache_k = cache_k[:, :, -window_size:, :]
cache_v = cache_v[:, :, -window_size:, :]
```

&nbsp;
#### Pratikte optimizasyonlar

Bu optimizasyonları [`gpt_with_kv_cache_optimized.py`](gpt_with_kv_cache_optimized.py) dosyasında bulabilirsiniz.


M4 çipli bir Mac Mini'de (CPU), 200 token üretimi ve bağlam uzunluğuna eşit bir pencere boyutuyla (aynı sonuçları garantilemek için) kodların çalışma süreleri şöyle karşılaştırılır:

|                                  | Token/saniye |
| -------------------------------- | ---------- |
| `gpt_ch04.py`                    | 27         |
| `gpt_with_kv_cache.py`           | 144        |
| `gpt_with_kv_cache_optimized.py` | 166        |

Ne yazık ki, bu çok küçük bir model olduğu için CUDA cihazlarında hız avantajları ortadan kalkar; bu küçük modelde cihaz aktarımı ve iletişim maliyeti, KV önbelleğinin faydalarına ağır basar.


&nbsp;
## Ek Kaynaklar

1. [Sıfırdan Qwen3 KV önbelleği ölçümleri](../../ch05/11_qwen3#pro-tip-2-speed-up-inference-with-compilation)
2. [Sıfırdan Llama 3 KV önbelleği ölçümleri](../../ch05/07_gpt_to_llama/README.md#pro-tip-3-speed-up-inference-with-compilation)
3. [Understanding and Coding the KV Cache in LLMs from Scratch](https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms) -- Bu README'nin daha ayrıntılı bir yazılı hâli
