# Doğrusal Dikkat İçin Gated DeltaNet

> 🇹🇷 **Türkçe çeviri.** Orijinal İngilizce sürüm: [README.md](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch04/08_deltanet/README.md) · Kod blokları, gerçek kaynak dosyalarla birebir eşleşmesi için çevrilmeden bırakılmıştır.

Yakın zamanda [Qwen3-Next](https://qwen.ai/blog?id=4074cca80393150c248e508aa62983f9cb7d27cd&from=research.latest-advancements-list) ve [Kimi Linear](https://arxiv.org/abs/2510.26692), dikkat mekanizmasına alternatif olarak bağlam uzunluğuna göre karesel yerine doğrusal ölçeklenen hibrit transformer'lar önerdi.

Hem Qwen3-Next hem de Kimi Linear 3:1 oranını kullanır; yani doğrusal Gated DeltaNet varyantını kullanan her üç transformer bloğuna karşılık, aşağıdaki şekilde gösterildiği gibi tam dikkat kullanan bir blok bulunur.

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/gated_deltanet/01.webp" alt="Qwen3-Next versus Kimi Linear">



&nbsp;

## Giriş ve Genel Bakış

Gated DeltaNet, [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464) makalesinden gelen bir kapı (gating) mekanizması dahil olmak üzere, yinelemeli sinir ağlarından (RNN) esinlenen bir doğrusal dikkat varyantıdır. Bir bakıma Gated DeltaNet, Mamba tarzı kapılara sahip bir DeltaNet'tir; DeltaNet ise doğrusal bir dikkat mekanizmasıdır.

Kimi Linear, Qwen3-Next'in doğrusal dikkat mekanizmasını, esasen Gated DeltaNet'in bir iyileştirmesi olan Kimi Delta Attention (KDA) mekanizmasıyla değiştirir. Qwen3-Next bellek sönümleme oranını kontrol etmek için skaler bir kapı (dikkat başı başına tek bir değer) uygularken, Kimi Linear bunu her öznitelik boyutu için kanal bazlı bir kapılama ile değiştirir. Yazarlara göre bu, bellek üzerinde daha fazla denetim sağlar ve bu da uzun bağlamlı akıl yürütmeyi iyileştirir.

Ayrıca, tam dikkat katmanları için Kimi Linear, Qwen3-Next'in kapılı dikkat katmanlarını (esasen çıkış kapılamalı standart çok başlı dikkat katmanları) Çok Başlı Gizil Dikkat (MLA) ile değiştirir. Bu, daha önce DeepSeek V3/R1 bölümünde ele aldığımız MLA mekanizmasının aynısıdır, ancak ek bir kapı içerir. (Hatırlatma: MLA, KV önbelleği boyutunu azaltmak için anahtar/değer uzayını sıkıştırır.)

Kimi Linear'daki MLA bu kapıyı kullanmaz; bu bilinçli bir tercihti, böylece yazarlar mimariyi standart MLA ile daha doğrudan karşılaştırabildiler. Ancak gelecekte kapıyı eklemeyi planladıklarını [belirttiler](https://x.com/yzhang_cs/status/1984631714464088563).

MLA'yı zaten [../05_mla](../05_mla) klasöründe uyguladığımız için, bu bonus materyal Gated DeltaNet tarafına odaklanmaktadır.


&nbsp;
## Kapılı Dikkat (Gated Attention)

Gated DeltaNet'in kendisine geçmeden önce kapıdan (gate) kısaca söz edelim. Önceki şekildeki Qwen3-Next mimarisinin üst kısmında görebileceğiniz gibi, Qwen3-Next "kapılı dikkat" kullanır. Bu, esasen ek bir sigmoid kapıya sahip klasik tam dikkattir.

Bu kapılama, aşağıda gösterim amacıyla 3. bölümdeki `MultiHeadAttention` koduna eklediğim basit bir değişikliktir:

```python
import torch
from torch import nn

class GatedMultiHeadAttention(nn.Module):
    def __init__(
        self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False
    ):
        super().__init__()
        assert d_out % num_heads == 0

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        ####################################################
        ### NEW: Add gate
        self.W_gate = nn.Linear(d_in, d_out, bias=qkv_bias)
        ####################################################
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1),
            persistent=False,
        )

    def forward(self, x):
        b, num_tokens, _ = x.shape
        queries = self.W_query(x)
        ####################################################
        ### NEW: Add gate
        gate = self.W_gate(x)
        ####################################################
        keys = self.W_key(x)
        values = self.W_value(x)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)

        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(
            mask_bool, torch.finfo(attn_scores.dtype).min
        )

        attn_weights = torch.softmax(
            attn_scores / (self.head_dim ** 0.5), dim=-1
        )
        attn_weights = self.dropout(attn_weights)

        context = (attn_weights @ values).transpose(1, 2)
        context = context.reshape(b, num_tokens, self.d_out)

        ####################################################
        ### NEW: Add gate
        context = context * torch.sigmoid(gate)
        ####################################################
        out = self.out_proj(context)
        return out
```



Görüldüğü gibi, dikkat her zamanki gibi hesaplandıktan sonra model aynı girdiden ayrı bir kapılama sinyali üretir, bunu 0 ile 1 arasında tutmak için bir sigmoid uygular ve dikkat çıktısıyla çarpar. Bu, modelin belirli öznitelikleri dinamik olarak büyütmesine veya küçültmesine olanak tanır. Qwen3-Next geliştiricileri bunun eğitim kararlılığına yardımcı olduğunu [belirtiyor](https://qwen.ai/blog?id=4074cca80393150c248e508aa62983f9cb7d27cd&from=research.latest-advancements-list):

> [...] dikkat çıkışı kapılama mekanizması, Attention Sink ve Massive Activation gibi sorunların giderilmesine yardımcı olur ve model genelinde sayısal kararlılığı güvence altına alır.


&nbsp;
## Gated DeltaNet

Peki Gated DeltaNet nedir? Gated DeltaNet (*Gated Delta Network*'ün kısaltması), Qwen3-Next'in doğrusal dikkat katmanıdır ve standart softmax dikkatine alternatif olarak tasarlanmıştır. Daha önce belirtildiği gibi [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464) makalesinden alınmıştır.

Gated DeltaNet, başlangıçta Mamba2'nin geliştirilmiş bir sürümü olarak önerildi; Mamba2'nin kapılı sönümleme mekanizmasını bir delta kuralıyla birleştirir.

Mamba bir durum-uzayı modelidir (transformer'lara bir alternatif); gelecekte ayrıca ele alınmayı hak eden geniş bir konudur.

Delta kuralı kısmı, bellek durumu olarak kullanılan bir gizli durumu (hidden state) güncellemek için yeni ve öngörülen değerler arasındaki farkın (delta, Δ) hesaplanmasını ifade eder (bu konuya birazdan döneceğiz).

(Yan not: Klasik makine öğrenmesi literatürüne aşina okuyucular bunu, biyolojiden esinlenen Hebbian öğrenmeye benzer düşünebilir: "Birlikte ateşlenen hücreler birbirine bağlanır." Bu esasen perceptron güncelleme kuralının ve gradyan inişine dayalı öğrenmenin bir öncülüdür, ancak denetim (supervision) içermez.)

Gated DeltaNet, daha önce ele alınan kapılı dikkattekine benzer bir kapıya sahiptir; tek fark, lojistik sigmoid yerine aşağıda gösterildiği gibi SiLU aktivasyonu kullanmasıdır. (SiLU tercihi muhtemelen standart sigmoid'e kıyasla gradyan akışını ve kararlılığı iyileştirmek içindir.)

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/gated_deltanet/02.webp" alt="Gated DeltaNet" width=500px>

Ancak yukarıdaki şekilde gösterildiği gibi, Gated DeltaNet'teki "gated" ifadesi birkaç ek kapıya da işaret eder:

- `α` (sönümleme kapısı) belleğin zaman içinde ne kadar hızlı söneceğini veya sıfırlanacağını denetler,
- `β` (güncelleme kapısı) yeni girdilerin durumu ne kadar güçlü değiştireceğini denetler.

Kod olarak, yukarıda tasvir edilen Gated DeltaNet'in basitleştirilmiş bir sürümü (evrişimsel karıştırma olmadan) şöyle uygulanabilir (kod, Qwen3 ekibinin [resmî uygulamasından](https://github.com/huggingface/transformers/blob/0ed6d51ae8ed3f4fafca67a983b8d75bc76cd51b/src/transformers/models/qwen3_next/modular_qwen3_next.py#L835) esinlenmiştir).

(Bazı uygulamaların sönümleme kapısını `gk` (k adımı için kapı) olarak adlandırdığını ve `exp(gk)` ifadesinin makaledeki $\alpha_t$ değerine karşılık geldiğini unutmayın. Bu ilişkiyi açık tutmak için aşağıdaki kod parçası, logaritmik uzaydaki `alpha_log` kapısını üstel sönümleme `alpha` değerinden ayırır.)


```python
import torch
from torch import nn
import torch.nn.functional as F

def l2norm(x, dim=-1, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)

class GatedDeltaNet(nn.Module):
    def __init__(
        self, d_in, d_out, dropout, num_heads, qkv_bias=False
    ):
        super().__init__()
        assert d_out % num_heads == 0

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        ####################################################
        ### NEW: Gates for delta rule and output gating
        self.W_gate = nn.Linear(d_in, d_out, bias=False)
        self.W_beta = nn.Linear(d_in, d_out, bias=False)

        # Note: The decay gate alpha corresponds to
        # A_log + W_alpha(x) + dt_bias
        self.W_alpha = nn.Linear(d_in, num_heads, bias=False)
        self.dt_bias = nn.Parameter(torch.ones(num_heads))
        A_init = torch.empty(num_heads).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A_init))
        # We could implement this as
        # W_alpha = nn.Linear(d_in, num_heads, bias=True)
        # but the bias is separate for interpretability and
        # to mimic the official implementation

        self.norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        ####################################################

        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, num_tokens, _ = x.shape
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)
        ####################################################
        ### NEW: Compute delta rule gates
        beta = torch.sigmoid(self.W_beta(x))
        alpha_log = -self.A_log.exp().view(1, 1, -1) * F.softplus(
            self.W_alpha(x) + self.dt_bias
        )
        alpha = alpha_log.exp()
        gate = self.W_gate(x)
        ####################################################

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        beta = beta.view(b, num_tokens, self.num_heads, self.head_dim)
        gate = gate.view(b, num_tokens, self.num_heads, self.head_dim)  # NEW

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)
        beta = beta.transpose(1, 2)

        ####################################################
        ### NEW: QKNorm-like normalization for delta rule
        queries = l2norm(queries, dim=-1) / (self.head_dim ** 0.5)
        keys = l2norm(keys, dim=-1)
        ####################################################

        S = x.new_zeros(b, self.num_heads, self.head_dim, self.head_dim)

        outs = []
        ####################################################
        ### NEW: Gated delta rule update
        for t in range(num_tokens):
            k_t = keys[:, :, t]
            q_t = queries[:, :, t]
            v_t = values[:, :, t]
            b_t = beta[:, :, t]
            a_t = alpha[:, t].unsqueeze(-1).unsqueeze(-1)

            S = S * a_t
            kv_mem = (S * k_t.unsqueeze(-1)).sum(dim=-2)
            delta = (v_t - kv_mem) * b_t
            S = S + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
            y_t = (S * q_t.unsqueeze(-1)).sum(dim=-2)
            ####################################################
            outs.append(y_t)

        context = torch.stack(outs, dim=2).transpose(1, 2).contiguous()
        context = context.view(b, num_tokens, self.num_heads, self.head_dim)

        ####################################################
        ### NEW: Apply RMSNorm and SiLU gate
        context = self.norm(context)
        context = context * F.silu(gate)
        ####################################################

        context = context.view(b, num_tokens, self.d_out)
        context = self.dropout(context)
        out = self.out_proj(context)
        return out
```

(Basitlik adına, kodu daha okunabilir tutmak ve yinelemeli (recurrent) yönlere odaklanmak için Qwen3-Next ve Kimi Linear'ın kullandığı evrişimsel karıştırmayı atladığımı unutmayın.)

Yukarıda görüldüğü gibi, standart (veya kapılı) dikkate kıyasla pek çok fark var.

Kapılı dikkatte model, tüm token'lar arasında normal dikkati hesaplar (her token diğer her token'a dikkat eder, yani bakar). Ardından, dikkat çıktısı elde edildikten sonra bir kapı (bir sigmoid), bu çıktının ne kadarının korunacağına karar verir. Buradaki çıkarım şu: bu hâlâ bağlam uzunluğuna göre karesel ölçeklenen klasik ölçeklenmiş nokta çarpımı dikkatidir.

Hatırlatma olarak, ölçeklenmiş nokta çarpımı dikkati softmax(QKᵀ)V şeklinde hesaplanır; burada Q ve K, *n*-çarpı-*d* boyutlu matrislerdir, *n* girdi token'larının sayısı ve *d* gömme boyutudur. Yani QKᵀ, *n*-çarpı-*n* boyutlu bir dikkat matrisi verir; bu da *n*-çarpı-*d* boyutlu değer matrisi V ile çarpılır:

```
attn_scores = queries @ keys.transpose(2, 3)

mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
attn_scores.masked_fill_(
    mask_bool, torch.finfo(attn_scores.dtype).min
)

attn_weights = torch.softmax(
    attn_scores / (self.head_dim ** 0.5), dim=-1
)

context = (attn_weights @ values).transpose(1, 2)
context = context.reshape(b, num_tokens, self.d_out)
```



<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/gated_deltanet/03.webp" alt="Quadratic attention" width=500px />

Gated DeltaNet'te ise *n*-çarpı-*n* boyutlu bir dikkat matrisi yoktur. Bunun yerine model token'ları tek tek işler. Her yeni token geldiğinde güncellenen, süregelen bir bellek (bir durum) tutar. Aşağıda uygulanan budur; burada `S`, her *t* zaman adımı için yinelemeli olarak güncellenen durumdur.

```python
S = x.new_zeros(b, self.num_heads, self.head_dim, self.head_dim)
outs = []

for t in range(num_tokens):
    k_t = keys[:, :, t]
    q_t = queries[:, :, t]
    v_t = values[:, :, t]
    b_t = beta[:, :, t]
    a_t = alpha[:, t].unsqueeze(-1).unsqueeze(-1)

    S = S * a_t
    kv_mem = (S * k_t.unsqueeze(-1)).sum(dim=-2)
    delta = (v_t - kv_mem) * b_t
    S = S + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
    y_t = (S * q_t.unsqueeze(-1)).sum(dim=-2)
```

Kapılar ise bu belleğin nasıl değiştiğini denetler:

- α (`alpha`) eski belleğin ne kadarının unutulacağını (sönümleme) düzenler.

- β (`beta`) *t* zaman adımındaki mevcut token'ın belleği ne kadar güncelleyeceğini düzenler.

(Ve yukarıdaki kod parçasında gösterilmeyen son çıkış kapısı, kapılı dikkattekine benzer; çıktının ne kadarının korunacağını denetler.)

Yani bir bakıma, Gated DeltaNet'teki bu durum güncellemesi, yinelemeli sinir ağlarının (RNN) çalışma şekline benzer. Avantajı, bağlam uzunluğuna göre karesel yerine (for döngüsü aracılığıyla) doğrusal ölçeklenmesidir.

Bu yinelemeli durum güncellemesinin dezavantajı ise, klasik (veya kapılı) dikkate kıyasla, tam ikili (pairwise) dikkatten gelen küresel bağlam modelleme yeteneğinden feragat etmesidir.

Gated DeltaNet bir ölçüde hâlâ bağlamı yakalayabilir, ancak bunu bellek (*S*) darboğazından geçirmek zorundadır. Bu bellek sabit boyutludur ve dolayısıyla daha verimlidir, ancak RNN'lere benzer şekilde geçmiş bağlamı tek bir gizli durumda sıkıştırır.

Bu nedenle Qwen3-Next ve Kimi Linear mimarileri tüm dikkat katmanlarını DeltaNet katmanlarıyla değiştirmez; bunun yerine daha önce bahsedilen 3:1 oranını kullanır.

&nbsp;
## DeltaNet Bellek Tasarrufu

Önceki bölümde, DeltaNet'in tam dikkate göre avantajını, bağlam uzunluğuna göre karesel yerine doğrusal hesaplama karmaşıklığı açısından ele almıştık.

Doğrusal hesaplama karmaşıklığının yanı sıra, DeltaNet'in bir diğer büyük avantajı bellek tasarrufudur; çünkü DeltaNet modülleri KV önbelleğini büyütmez. (KV önbellekleme hakkında daha fazla bilgi için bkz. [../03_kv-cache](../03_kv-cache)). Bunun yerine, daha önce belirtildiği gibi sabit boyutlu bir yinelemeli durum tutarlar; böylece bellek, bağlam uzunluğundan bağımsız olarak sabit kalır.

Klasik bir çok başlı dikkat (MHA) katmanı için KV önbelleği boyutunu şöyle hesaplayabiliriz:

```
KV_cache_MHA ≈ batch_size × n_tokens × n_heads × d_head × 2 × bytes
```

(Buradaki 2 çarpanı, önbellekte hem anahtarları hem de değerleri sakladığımız içindir.)

Yukarıda uygulanan basitleştirilmiş DeltaNet sürümü için ise:


```
KV_cache_DeltaNet = batch_size × n_heads × d_head × d_head × bytes
```

`KV_cache_DeltaNet` bellek boyutunun bağlam uzunluğuna (`n_tokens`) bağımlılığı olmadığını unutmayın. Ayrıca ayrı anahtar ve değerler yerine yalnızca S bellek durumunu sakladığımız için `2 × bytes` yerine sadece `bytes` geçerlidir. Ancak burada artık karesel bir `d_head × d_head` terimi olduğunu belirtelim. Bu, durumdan gelir:

```
S = x.new_zeros(b, self.num_heads, self.head_dim, self.head_dim)
```

Ama bu genellikle endişelenecek bir şey değildir; çünkü baş boyutu (head dimension) genelde nispeten küçüktür. Örneğin Qwen3-Next'te 128'dir.

Evrişimsel karıştırma içeren tam sürüm, çekirdek boyutu (kernel size) vb. dahil biraz daha karmaşıktır, ancak yukarıdaki formüller Gated DeltaNet'in arkasındaki ana eğilimi ve gerekçeyi göstermeye yeter.

Farklı bağlam uzunlukları için bellek tahminlerini ve tasarrufları aşağıdaki yardımcı betikle görselleştirebiliriz:

```bash
uv run plot_memory_estimates_gated_deltanet.py \
  --emb_dim 2048 \
  --n_heads 16 \
  --n_layers 48 \
  --dtype "bf16"
```

Yukarıdakinin `head_dim` değerini `emb_dim / n_heads` olarak hesapladığını unutmayın. Yani 2048 / 16 = 128.

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/gated_deltanet/plot.webp" alt="Gated DeltaNet scaling" width=500px>
