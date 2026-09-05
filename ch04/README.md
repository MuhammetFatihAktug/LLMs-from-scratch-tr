# Bölüm 4: Metin Üretmek İçin Sıfırdan Bir GPT Modeli Uygulamak

> 🇹🇷 **Türkçe çeviri.** Orijinal İngilizce sürüm: [README.md](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch04/README.md)

&nbsp;
## Ana Bölüm Kodu

- [01_main-chapter-code](01_main-chapter-code) ana bölüm kodunu içerir.

&nbsp;
## Bonus Materyaller

- [02_performance-analysis](02_performance-analysis) ana bölümde uygulanan GPT model(ler)inin performansını analiz eden isteğe bağlı kod içerir
- [03_kv-cache](03_kv-cache) çıkarım (inference) sırasında metin üretimini hızlandırmak için bir KV önbelleği (KV cache) uygular
- [07_moe](07_moe) Uzmanlar Karışımı (Mixture-of-Experts, MoE) açıklaması ve uygulaması
- [ch05/07_gpt_to_llama](../ch05/07_gpt_to_llama) bir GPT mimarisi uygulamasını Llama 3.2'ye dönüştürmek için adım adım bir rehber içerir ve Meta AI'dan önceden eğitilmiş ağırlıkları yükler (4. bölümü tamamladıktan sonra alternatif mimarilere bakmak ilgi çekici olabilir, ancak bunu 5. bölümü okuduktan sonraya da bırakabilirsiniz)


&nbsp;
## Dikkat Mekanizmasına Alternatifler

&nbsp;

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/attention-alternatives/attention-alternatives.webp">

&nbsp;

- [04_gqa](04_gqa) çoğu modern LLM'in (Llama 4, gpt-oss, Qwen3, Gemma 3 ve daha birçoğu) klasik Çok Başlı Dikkat'e (Multi-Head Attention, MHA) alternatif olarak kullandığı Gruplanmış Sorgu Dikkati'ne (Grouped-Query Attention, GQA) bir giriş içerir
- [05_mla](05_mla) DeepSeek V3 tarafından klasik Çok Başlı Dikkat'e (MHA) alternatif olarak kullanılan Çok Başlı Gizil Dikkat'e (Multi-Head Latent Attention, MLA) bir giriş içerir
- [06_swa](06_swa) Gemma 3 ve diğerleri tarafından kullanılan Kayan Pencere Dikkati'ne (Sliding Window Attention, SWA) bir giriş içerir
- [08_deltanet](08_deltanet) popüler bir doğrusal dikkat varyantı olan Gated DeltaNet'in açıklaması (Qwen3-Next ve Kimi Linear'da kullanılır)
- [10_kv-sharing](10_kv-sharing) Gemma 4 E2B ve E4B tarafından KV önbelleği belleğini azaltmak için kullanılan katmanlar arası KV paylaşımına bir giriş içerir


&nbsp;
## Dahası

Aşağıdaki videoda, bölüm içeriğinin bir kısmını kapsayan, birlikte kod yazdığımız bir oturum sunuyorum (tamamlayıcı materyal).

<br>
<br>

[![Videoya bağlantı](https://img.youtube.com/vi/YSAkgEarBGE/0.jpg)](https://www.youtube.com/watch?v=YSAkgEarBGE)
