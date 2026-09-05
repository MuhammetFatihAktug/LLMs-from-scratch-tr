# Bölüm 5: Etiketlenmemiş Veri Üzerinde Ön Eğitim (Pretraining)

> 🇹🇷 **Türkçe çeviri.** Orijinal İngilizce sürüm: [README.md](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch05/README.md)

&nbsp;
## Ana Bölüm Kodu

- [01_main-chapter-code](01_main-chapter-code) ana bölüm kodunu içerir

&nbsp;
## Bonus Materyaller

- [02_alternative_weight_loading](02_alternative_weight_loading) model ağırlıklarının OpenAI üzerinden erişilemez hale gelmesi ihtimaline karşı GPT model ağırlıklarını alternatif kaynaklardan yüklemek için kod içerir
- [03_bonus_pretraining_on_gutenberg](03_bonus_pretraining_on_gutenberg) LLM'i Project Gutenberg'deki tüm kitap derlemi üzerinde daha uzun süre ön eğitmek için kod içerir
- [04_learning_rate_schedulers](04_learning_rate_schedulers) öğrenme oranı zamanlayıcıları (learning rate scheduler) ve gradyan kırpma (gradient clipping) dahil olmak üzere daha gelişmiş bir eğitim fonksiyonunu uygulayan kod içerir
- [05_bonus_hparam_tuning](05_bonus_hparam_tuning) isteğe bağlı bir hiperparametre ayarlama betiği içerir
- [06_user_interface](06_user_interface) önceden eğitilmiş LLM ile etkileşim kurmak için interaktif bir kullanıcı arayüzü uygular
- [08_memory_efficient_weight_loading](08_memory_efficient_weight_loading) model ağırlıklarının PyTorch'un `load_state_dict` metodu aracılığıyla nasıl daha verimli yükleneceğini gösteren bir bonus not defteri içerir
- [09_extending-tokenizers](09_extending-tokenizers) GPT-2 BPE tokenizer'ının sıfırdan bir uygulamasını içerir
- [10_llm-training-speed](10_llm-training-speed) LLM eğitim hızını artırmak için PyTorch performans ipuçlarını gösterir
- [18_muon](18_muon) Muon optimize edicisinin GPT model eğitim kurulumuyla nasıl kullanılacağını açıklar

&nbsp;
## Sıfırdan LLM Mimarileri

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/qwen/qwen-overview.webp">

&nbsp;


- [07_gpt_to_llama](07_gpt_to_llama) bir GPT mimarisi uygulamasını Llama 3.2'ye dönüştürmek için adım adım bir rehber içerir ve Meta AI'dan önceden eğitilmiş ağırlıkları yükler
- [11_qwen3](11_qwen3) Qwen3 0.6B ve Qwen3 30B-A3B'nin (Uzmanlar Karışımı) sıfırdan bir uygulaması; temel (base), akıl yürütme (reasoning) ve kodlama model varyantlarının önceden eğitilmiş ağırlıklarını yükleyen kodu da içerir
- [12_gemma3](12_gemma3) Gemma 3 270M'in sıfırdan bir uygulaması ve KV önbellekli alternatifi; önceden eğitilmiş ağırlıkları yükleyen kodu da içerir
- [13_olmo3](13_olmo3) Olmo 3 7B ve 32B'nin (Base, Instruct ve Think varyantları) sıfırdan bir uygulaması ve KV önbellekli alternatifi; önceden eğitilmiş ağırlıkları yükleyen kodu da içerir
- [17_gemma4](17_gemma4) Gemma 4'ün E2B ve E4B yoğun (dense) varyantlarının sıfırdan bir uygulaması

&nbsp;
## Bu Bölüm İçin Birlikte Kod Yazma Videosu

<br>
<br>

[![Videoya bağlantı](https://img.youtube.com/vi/Zar2TJv-sE0/0.jpg)](https://www.youtube.com/watch?v=Zar2TJv-sE0)
