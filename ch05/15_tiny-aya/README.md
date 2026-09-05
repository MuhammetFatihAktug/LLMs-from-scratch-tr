# Sıfırdan Tiny Aya 3.35B

> 🇹🇷 **Türkçe çeviri.** Orijinal İngilizce sürüm: [README.md](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch05/15_tiny-aya/README.md) · Tiny Aya, Cohere tarafından geliştirilen yeni ve "küçük" bir LLM'dir; 3B parametre sınıfında "en yetenekli çok dilli açık ağırlıklı model" olduğu söylenmektedir. ([Duyuru gönderisine](https://cohere.com/blog/cohere-labs-tiny-aya) göre Tiny Aya, Qwen3-4B, Gemma 3 4B ve Ministral 3 3B modellerini geride bırakıyor.)

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/tiny-aya/01.webp">



Yerelde çalıştırıp denemek için harika bir model. Tek çekince, açık ağırlıklı bir model olmasına rağmen lisans koşullarının nispeten kısıtlı olması ve yalnızca ticari olmayan kullanıma izin vermesidir.

Bunun dışında Arya, kişisel ve (ticari olmayan) araştırma kullanımı için faydalı olan birkaç türde gelen 3,35 milyar parametreli bir modeldir:

  - [tiny-aya-base](https://huggingface.co/CohereLabs/tiny-aya-base) (temel model)
  - [tiny-aya-global](https://huggingface.co/CohereLabs/tiny-aya-global) (diller ve bölgeler arasında en iyi denge; not defterinin varsayılanı)
  - [tiny-aya-fire](https://huggingface.co/CohereLabs/tiny-aya-fire) (Güney Asya dilleri için optimize edilmiş)
  - [tiny-aya-water](https://huggingface.co/CohereLabs/tiny-aya-water) (Avrupa ve Asya Pasifik dilleri için optimize edilmiş)
  - [tiny-aya-earth](https://huggingface.co/CohereLabs/tiny-aya-earth) (Batı Asya ve Afrika dilleri için optimize edilmiş)



Daha ayrıntılı olarak, modellerin optimize edildiği dillerin listesi şöyledir:

| Bölge            | Diller                                                       | Optimize Edilmiş Model |
| ---------------- | ------------------------------------------------------------ | --------------- |
| **Asya Pasifik** | Geleneksel Çince, Kantonca, Vietnamca, Tagalogca, Cavaca, Khmerce, Tayca, Birmanca, Malayca, Korece, Laoca, Endonezce, Basitleştirilmiş Çince, Japonca | tiny-aya-water  |
| **Afrika**       | Zuluca, Amharca, Hausaca, İgboca, Svahilice, Xhosaca, Wolofça, Shonaca, Yorubaca, Nijerya Pidgin dili, Malgaşça | tiny-aya-earth  |
| **Güney Asya**   | Teluguca, Marathice, Bengalce, Tamilce, Hintçe, Pencapça, Guceratça, Urduca, Nepalce | tiny-aya-fire   |
| **Avrupa**       | Katalanca, Galiçyaca, Felemenkçe, Danca, Fince, Çekçe, Portekizce, Fransızca, Litvanca, Slovakça, Baskça, İngilizce, İsveççe, Lehçe, İspanyolca, Slovence, Ukraynaca, Yunanca, Bokmål, Rumence, Sırpça, Almanca, İtalyanca, Rusça, İrlandaca, Macarca, Bulgarca, Hırvatça, Estonca, Letonca, Galce | tiny-aya-water  |
| **Batı Asya**    | Arapça, Maltaca, Türkçe, İbranice, Farsça                    | tiny-aya-earth  |


Mimari açıdan Tiny Aya, birkaç dikkate değer değişiklik dışında (SwiGLU ve Gruplanmış Sorgu Dikkati gibi bariz olanların yanı sıra) klasik bir kod çözücü (decoder) tarzı transformer'dır:

1. **Paralel transformer blokları.** Paralel bir transformer bloğu, dikkat ve MLP'yi aynı normalize edilmiş girdiden hesaplar, ardından her ikisini tek adımda artık bağlantıya (residual) ekler. Bunun, hesaplama verimini artırmak için bir katman içindeki sıralı bağımlılıkları azaltmaya yönelik olduğunu tahmin ediyorum.

2. **Kayan pencere dikkati.** Özellikle, Arcee Trinity ve Olmo 3'e benzer şekilde 3:1 yerel:küresel oranı kullanır. Pencere boyutu da 4096'dır. Ayrıca Arcee'ye benzer şekilde, kayan pencere katmanları RoPE kullanırken tam dikkat katmanları NoPE kullanır.

3. **LayerNorm.** Çoğu mimari, hesaplama açısından biraz daha ucuz olduğu ve iyi performans gösterdiği için RMSNorm'a geçti. Tiny Aya ise LayerNorm'un değiştirilmiş bir sürümüyle daha klasik kalıyor (buradaki uygulama standart LayerNorm gibidir, ancak kaydırma (shift), yani bias parametresi yoktur).



&nbsp;
## Dosyalar

[standalone-tiny-aya.ipynb](standalone-tiny-aya.ipynb), Tiny Aya mimarisini uygulayan ve önceden eğitilmiş ağırlıkları yükleyen bağımsız bir Jupyter not defteridir.


Alternatif [standalone-tiny-aya-plus-kvcache.ipynb](standalone-tiny-aya-plus-kv-cache.ipynb) not defteri, daha iyi çalışma zamanı performansı için bir KV önbelleği ekler (ancak koda daha fazla karmaşıklık katar). KV önbellekleme hakkında daha fazla bilgi için [Understanding and Coding the KV Cache in LLMs from Scratch](https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms) yazıma bakın.


<br>

Mimari farkları hakkında daha fazla bilgi edinmek ve diğer mimarilerle karşılaştırmaları okumak için [The Big LLM Architecture Comparison: From DeepSeek-V3 to Kimi K2: A Look At Modern LLM Architecture Design](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison) yazıma bakın.
