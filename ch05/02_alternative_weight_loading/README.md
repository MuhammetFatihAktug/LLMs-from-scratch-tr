# Önceden Eğitilmiş Ağırlıkları Yüklemek İçin Alternatif Yaklaşımlar

> 🇹🇷 **Türkçe çeviri.** Orijinal İngilizce sürüm: [README.md](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch05/02_alternative_weight_loading/README.md) · Bu klasör, ağırlıkların OpenAI üzerinden erişilemez hâle gelmesi ihtimaline karşı alternatif ağırlık yükleme stratejilerini içerir.

- [weight-loading-pytorch.ipynb](weight-loading-pytorch.ipynb): (Önerilen) orijinal TensorFlow ağırlıklarını dönüştürerek oluşturduğum PyTorch state dict'lerinden ağırlıkları yükleyen kodu içerir

- [weight-loading-hf-transformers.ipynb](weight-loading-hf-transformers.ipynb): ağırlıkları Hugging Face Model Hub'dan `transformers` kütüphanesi aracılığıyla yükleyen kodu içerir

- [weight-loading-hf-safetensors.ipynb](weight-loading-hf-safetensors.ipynb): ağırlıkları Hugging Face Model Hub'dan doğrudan `safetensors` kütüphanesi aracılığıyla yükleyen kodu içerir (bir Hugging Face transformer modeli örneklemeyi atlayarak)
