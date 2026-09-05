# DeepSeek Seyrek Dikkati (DeepSeek Sparse Attention, DSA)

> 🇹🇷 **Türkçe çeviri.** Orijinal İngilizce sürüm: [README.md](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch04/09_dsa/README.md) · Komut, formül ve kod blokları birebir korunmuştur.

Bu bonus materyal, [DeepSeek-V3.2](https://huggingface.co/deepseek-ai/DeepSeek-V3.2) ile tanıtılan ve ilk olarak deneysel [DeepSeek-V3.2-Exp](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp) sürümünde yayımlanan DeepSeek Seyrek Dikkat (DSA) mekanizmasını uygular.

Aşağıdaki genel bakış, [From DeepSeek V3 to V3.2: Architecture, Sparse Attention, and RL Updates](https://magazine.sebastianraschka.com/p/technical-deepseek) yazısındaki DSA tartışmasını izler.

&nbsp;
## Giriş

Standart nedensel (causal) öz-dikkat, her sorgu için önceki tüm token'lara dikkat eder; bu da L dizi uzunluğuyla O(L²) hesaplama ve O(L) KV önbelleği büyümesi anlamına gelir.

[Kayan Pencere Dikkati (SWA)](../06_swa) zaten dikkati sabit bir yerel pencereyle sınırlamanın bu maliyeti önemli ölçüde azalttığını gösterdi. SWA'da her sorgu token'ı yalnızca yakınındaki önceki token'lardan oluşan yerel bir aralığa dikkat eder.

&nbsp;

<img src="https://sebastianraschka.com/images/blog/2025/technical-deepseek/09.png" alt="Sliding window attention" width="800px" />

*Şekil 1. Kayan pencere dikkati, her sorgu token'ını sabit bir yerel bağlam penceresiyle sınırlar.*

&nbsp;

DSA, önceki token'ların yalnızca bir alt kümesine dikkat etme fikrini genel hatlarıyla aynı şekilde kullanır. Ancak sabit pencereyi öğrenilmiş bir seçim mekanizmasıyla değiştirir. Model, her sorgu token'ı için aday geçmiş token'lara puan verir ve yalnızca en ilgili olanları tutar.

&nbsp;

<img src="https://sebastianraschka.com/images/blog/2025/technical-deepseek/10.png" alt="DeepSeek Sparse Attention selected-token pattern" width="800px" />

*Şekil 2. DeepSeek Seyrek Dikkati, her sorgu token'ı için geçmiş token'lardan öğrenilmiş bir alt küme seçer.*

&nbsp;

### Mimariye genel bakış

DSA, standart dikkatin üzerine iki bileşen ekler.

**1. Lightning Indexer (Yıldırım İndeksleyici)**

Her $t$ sorgu token'ı ve her $s$ aday geçmiş token'ı için indeksleyici skaler bir ilgililik puanı hesaplar. Bu uygulama, referans koddaki ölçek çarpanlarını açıkça gösterir:

$$I_{t,s} = \sum_{j=1}^{H_I} \frac{w_{t,j}}{\sqrt{H_I}} \cdot \text{ReLU}\left(\frac{q_{t,j} \cdot k_s}{\sqrt{d_I}}\right)$$

Burada:
- $H_I$ hafif indeks başlarının sayısıdır,
- $q_{t,j}$, $t$ token'ı ve $j$ başı için indeksleyici sorgu vektörüdür,
- $k_s$, $s$ geçmiş token'ı için paylaşılan indeksleyici anahtar vektörüdür,
- $w_{t,j}$, $1 / \sqrt{H_I}$ ile ölçeklenen, baş başına öğrenilmiş bir kapıdır.

ReLU, negatif nokta çarpımı katkılarını sıfırlar; kapılı toplam ise indeks başları boyunca toplayarak geçmiş token başına tek bir ilgililik puanı üretir.

DeepSeek modelinin tamamında indeksleyici, Çok Başlı Gizil Dikkat'ten (MLA) gelen sıkıştırılmış token temsilleriyle çalışır. Bu klasör GPT uygulamasını daha basit tutar ve indeksleyici sorgularını ile anahtarlarını klasik gizli durumlardan (hidden state) hesaplar.

**2. Token Selector (Token Seçici)**

Tüm indeksleyici puanları hesaplandıktan sonra yalnızca en yüksek puanlı ilk K konum tutulur. Diğer tüm konumlar, standart softmax'tan *önce* −∞ ile maskelenir; böylece model etkin olarak yalnızca $k \ll L$ token'a dikkat eder.

Nihai seyrekliğin kaynağı indeksleyicideki ReLU değildir. Puanlar birden çok indeks başı üzerinden toplandığı için, nihai puanların çoğu yine de sıfırdan farklı olabilir. Seyrek örüntüyü oluşturan, yalnızca ilk K konumu tutan token seçicidir.

Kaynaşık (fused) bir üretim uygulamasında bu, dikkat hesaplamasını O(L²)'den O(L·k)'ye düşürebilir. Buradaki uygulama standart yoğun dikkat puanı matrisini korur ve DSA ile seçilen ilk K maskesini softmax'tan önce uygular. Bu, seçim mantığını incelemeyi kolaylaştırır, ancak kaynaşık çekirdek (fused kernel) hesaplama tasarrufunu sağlamaz.

Aşağıdaki şekil akışı özetler. Yıldırım indeksleyici aday token'lara puan verir, seçici ilk K konumu tutar ve ortaya çıkan maske olağan dikkat softmax'ını kısıtlar.

&nbsp;

<img src="https://sebastianraschka.com/images/blog/2025/technical-deepseek/11.png" alt="DeepSeek Sparse Attention flowchart" width="700px" />

*Şekil 3. DSA önce aday token'lara puan verir, ardından nihai dikkat maskesi için ilk K token'ı tutar.*

&nbsp;
## Uygulama

`gpt_with_kv_dsa.py` şunları sağlar:

| Sınıf | Açıklama |
|---|---|
| `LightningIndexer` | Geçmiş token ilgililiği için hafif, çok başlı puanlayıcı. |
| `MultiHeadAttentionWithDSA` | DSA seyrek maskeleme + isteğe bağlı KV önbelleği içeren standart MHA. |
| `GPTModel` | `MultiHeadAttentionWithDSA` kullanan GPT tarzı model. |

Uygulama, bu depodaki diğer bonus materyallerin tarzını izler ve bağımsız bir betik olarak çalıştırılabilir. Amacı, DSA mekanizmasını küçük bir GPT tarzı modelde incelenebilir kılmaktır. DeepSeek'in tam MLA yığınını, kaynaşık seyrek çekirdekleri veya dağıtıma özgü optimizasyonları uygulamaz.

&nbsp;
## Kullanım

```bash
uv run gpt_with_kv_dsa.py \
  --emb_dim 768 \
  --n_heads 12 \
  --n_layers 12 \
  --max_new_tokens 200 \
  --index_n_heads 4 \
  --index_head_dim 64 \
  --topk 64
```

Temel argümanlar:

| Argüman | Varsayılan | Açıklama |
|---|---|---|
| `--index_n_heads` | 4 | Hafif indeksleyici başlarının sayısı (H_I). |
| `--index_head_dim` | 64 | Her indeksleyici başının boyutu. |
| `--topk` | 64 | Her sorgunun dikkat ettiği token sayısı (k). Kısa diziler için dizi uzunluğuyla sınırlanır. |

&nbsp;
## DeepSeek V3.2 ile İlişkisi

Tam ölçekli DeepSeek-V3.2 modeli, DSA ile birlikte Çok Başlı Gizil Dikkat'i (MLA, bkz. [../05_mla](../05_mla)) kullanır ve indeksleyici sorguları ham girdi yerine paylaşılan sıkıştırılmış gizil temsilden türetilir. DeepSeek-V3.2, DSA'nın ilk kez tanıtıldığı ve test edildiği DeepSeek-V3.2-Exp ile aynı mimariyi kullanır.

Temel seçim fikri burada yeniden üretilmiştir: ucuz, öğrenilmiş bir nokta çarpımı puanlayıcısı, dikkat softmax'ından önce her sorguyu en ilgili token'larla sınırlar.

Aşağıda raporlanan çıkarım maliyeti karşılaştırması, DSA'nın uzun bağlamlı dağıtımlarda neden önemli olduğunu anlamak için faydalı bir bağlam sunar. Tasarruflar üretim çekirdeklerine ve sunum altyapısına bağlıdır; bu nedenle bu şekil, bu klasördeki eğitim amaçlı uygulama için bir kıyaslama (benchmark) olarak okunmamalıdır.

&nbsp;

<img src="https://sebastianraschka.com/images/blog/2025/technical-deepseek/19.png" alt="Inference cost comparison for DeepSeek Sparse Attention" width="800px" />

*Şekil 4. DeepSeek'in uzun bağlamlı sunumda DSA'dan raporladığı çıkarım maliyeti tasarrufu; kaynak: [DeepSeek V3.2 teknik raporu](https://huggingface.co/deepseek-ai/DeepSeek-V3.2/resolve/main/assets/paper.pdf).*

&nbsp;
## Kaynaklar

- DeepSeek V3.2 teknik raporu: https://huggingface.co/deepseek-ai/DeepSeek-V3.2/resolve/main/assets/paper.pdf
- DeepSeek V3.2-Exp model kartı ve referans kodu: https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp
- Sebastian Raschka, "From DeepSeek V3 to V3.2: Architecture, Sparse Attention, and RL Updates": https://magazine.sebastianraschka.com/p/technical-deepseek
