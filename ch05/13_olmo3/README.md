# Sıfırdan Olmo 3 7B ve 32B

> 🇹🇷 **Türkçe çeviri.** Orijinal İngilizce sürüm: [README.md](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch05/13_olmo3/README.md) · Bu klasördeki [standalone-olmo3.ipynb](standalone-olmo3.ipynb) Jupyter not defteri, Olmo 3 7B ve 32B'nin sıfırdan bir uygulamasını içerir ve çalıştırmak için yaklaşık 13 GB RAM gerektirir.

Alternatif [standalone-olmo3-plus-kvcache.ipynb](standalone-olmo3-plus-kv-cache.ipynb) not defteri, daha iyi çalışma zamanı performansı için bir KV önbelleği ekler (ancak koda daha fazla karmaşıklık katar). KV önbellekleme hakkında daha fazla bilgi için [Understanding and Coding the KV Cache in LLMs from Scratch](https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms) yazıma bakın.

Aşağıda, referans model olarak Qwen3 ile yan yana bir karşılaştırma yer alıyor; Qwen3 0.6B bağımsız not defteriyle ilgileniyorsanız [buradan](../11_qwen3) ulaşabilirsiniz.

<br>

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/olmo3/olmo3-7B.webp?1">

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/olmo3/olmo3-32B.webp?1">

Olmo 3, aşağıda gösterildiği gibi farklı türlerde de gelir (mimari aynıdır, yalnızca eğitim hattı farklıdır):

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/olmo3/olmo3-pipeline.webp?1">


&nbsp;
## Olmo 3, Qwen3 ile nasıl karşılaştırılır

Eğitim ayrıntılarına değil mimariye odaklanan bu bölüm, Qwen3 ile kısa bir karşılaştırma sunar.


7B model:

1. Yukarıdaki şekillerde görebileceğimiz gibi, Olmo 3 mimarisi Qwen3'e nispeten benzerdir. Ancak bunun esasen Qwen3'ten değil, selefi Olmo 2'den esinlenmiş olması muhtemeldir.

2) Olmo 2'ye benzer şekilde, Olmo 3 hâlâ pre-norm yerine post-norm türünü kullanır; çünkü Olmo 2 makalesinde bunun eğitimi kararlı hâle getirdiği bulunmuştu.

3) İlginç biçimde, 7B model Olmo 2'ye benzer şekilde hâlâ çok başlı dikkat kullanıyor.
Ancak işleri daha verimli kılmak ve KV önbelleği boyutunu azaltmak için artık kayan pencere dikkati kullanıyorlar (örneğin Gemma 3'e benzer şekilde).

Ardından 32B model:

4) Genel olarak mimari aynı, sadece ölçeklendirilmiş. Ayrıca oranlar (örneğin ileri beslemeli katmanda girdiden ara boyuta geçiş vb.) kabaca Qwen3'tekilerle eşleşiyor.

5) Tahminim, mimarinin daha küçük sözlük nedeniyle başlangıçta Qwen3'ten biraz daha küçük olduğu ve doğrudan karşılaştırma için 32B'lik bir model elde etmek amacıyla ara boyut genişletmesini Qwen3'teki 5 katından Olmo 3'te 5,4 katına çıkardıkları yönünde.

6) Ayrıca, 32B modelin (nihayet!) gruplanmış sorgu dikkati kullandığını belirtelim.





<br>

Mimari farkları hakkında daha fazla bilgi edinmek ve diğer mimarilerle karşılaştırmaları okumak için [The Big LLM Architecture Comparison: From DeepSeek-V3 to Kimi K2: A Look At Modern LLM Architecture Design](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison) yazıma bakın.
