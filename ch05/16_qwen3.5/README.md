# Sıfırdan Qwen3.5 0.8B

> 🇹🇷 **Türkçe çeviri.** Orijinal İngilizce sürüm: [README.md](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch05/16_qwen3.5/README.md) · Bu klasör, [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) modelinin sıfırdan tarzda bir uygulamasını içerir.

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/qwen3.5/03.webp">

Qwen3.5, [Beyond Standard LLMs](https://magazine.sebastianraschka.com/p/beyond-standard-llms) yazımın [2. (Linear) Attention Hybrids](https://magazine.sebastianraschka.com/i/177848019/2-linear-attention-hybrids) bölümünde daha ayrıntılı anlattığım Qwen3-Next mimarisine dayanır.

<a href="https://magazine.sebastianraschka.com/p/beyond-standard-llms"><img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/qwen3.5/02.webp" width="500px"></a>

Qwen3.5'in `linear_attention` ve `full_attention` katmanlarını dönüşümlü olarak kullandığını unutmayın.  
Not defterleri, doğrusal dikkat yapı taşlarını [qwen3_5_transformers.py](qwen3_5_transformers.py) dosyasından yeniden kullanırken modelin genel akışını okunabilir tutar; bu dosya, Hugging Face'ten alınan ve Apache 2.0 açık kaynak lisansı altında sunulan doğrusal dikkat kodunu içerir.

&nbsp;
## Dosyalar

- [qwen3.5.ipynb](qwen3.5.ipynb): Ana Qwen3.5 0.8B not defteri uygulaması.
- [qwen3.5-plus-kv-cache.ipynb](qwen3.5-plus-kv-cache.ipynb): Verimlilik için KV önbellekli kod çözme kullanan aynı model.
- [qwen3_5_transformers.py](qwen3_5_transformers.py): Qwen3.5 doğrusal dikkati için kullanılan, Hugging Face Transformers'tan bazı yardımcı bileşenler.
