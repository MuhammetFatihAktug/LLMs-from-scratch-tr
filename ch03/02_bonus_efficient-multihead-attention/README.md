# Daha Verimli Çok Başlı Dikkat (Multi-Head Attention) Uygulamaları

> 🇹🇷 **Türkçe çeviri.** Orijinal İngilizce sürüm: [README.md](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch03/02_bonus_efficient-multihead-attention/README.md)

- [mha-implementations.ipynb](mha-implementations.ipynb) çok başlı dikkatin farklı uygulamalarını içerir ve karşılaştırır



### Özet

Aşağıdaki şekiller performans ölçüm sonuçlarını özetler (düşük olan daha iyidir).


&nbsp;
#### Yalnızca ileri geçiş (forward pass)

<a href="mha-implementations.ipynb"><img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/mha-benchmark/1_forward-only.webp?1" width="500px"></a>

&nbsp;
#### İleri ve geri geçiş (forward and backward pass)

<a href="mha-implementations.ipynb"><img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/mha-benchmark/2_forward-and-backward.webp?1" width="500px"></a>

&nbsp;
#### Derlemeden (compilation) sonra ileri ve geri geçiş

<a href="mha-implementations.ipynb"><img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/mha-benchmark/3_forward-and-backward-compiled.webp?1" width="500px"></a>
