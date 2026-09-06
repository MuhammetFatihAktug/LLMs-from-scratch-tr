# Build a Large Language Model (From Scratch) — Türkçe

> ### 🇹🇷 Bu, resmî olmayan bir Türkçe çeviridir
>
> Özgün eser **Sebastian Raschka**'ya aittir: [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) · Apache License 2.0 · *Copyright 2023-2026 Sebastian Raschka*
>
> **Bu depodaki dosyalar değiştirilmiştir** (Apache 2.0 §4(b) bildirimi): metinler Türkçeye çevrilmiştir. Çalıştırılabilir kod, hücre çıktıları ve ölçüm tabloları değiştirilmemiştir. Ayrıntı: [NOTICE](NOTICE)
>
> Özgün İngilizce sürüm bu deponun `main` dalında dokunulmadan durur. Bu çeviri `tr` dalındadır.
> Kitabın yazarı veya Manning tarafından incelenmemiş / onaylanmamıştır.
>
> 📖 [Okuma sırası ve depo haritası](OKUMA_SIRASI.md) · 📗 [Terim sözlüğü](SOZLUK.md) · 📊 [Türkçe örnek veriler](TURKCE_VERI.md)

Bu depo, GPT benzeri bir LLM'in geliştirilmesi, ön eğitimi ve ince ayarı için gerekli kodu içerir ve [Build a Large Language Model (From Scratch)](https://amzn.to/4fqvn0D) kitabının resmî kod deposudur.

<br>
<br>

<a href="https://amzn.to/4fqvn0D"><img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/cover.jpg?123" width="250px"></a>

<br>

[*Build a Large Language Model (From Scratch)*](http://mng.bz/orYv) kitabında, büyük dil modellerinin (LLM) nasıl çalıştığını sıfırdan, adım adım kodlayarak içeriden öğrenip anlayacaksınız. Bu kitapta size kendi LLM'inizi oluşturma sürecinde rehberlik ediyor, her aşamayı anlaşılır metin, şema ve örneklerle açıklıyorum.

Bu kitapta eğitim amaçlı, küçük ama işlevsel kendi modelinizi eğitmek ve geliştirmek için anlatılan yöntem, ChatGPT'nin arkasındakiler gibi büyük ölçekli temel modellerin (foundation model) oluşturulmasında kullanılan yaklaşımın bir yansımasıdır. Ayrıca bu kitap, daha büyük önceden eğitilmiş modellerin ağırlıklarını ince ayar için yükleyen kodu da içerir.

- [Kaynak kod deposunun resmî bağlantısı](https://github.com/rasbt/LLMs-from-scratch)
- [Kitabın Manning'deki (yayıncının web sitesi) bağlantısı](http://mng.bz/orYv)
- [Kitabın Amazon.com sayfası](https://www.amazon.com/gp/product/1633437167)
- ISBN 9781633437166

<a href="http://mng.bz/orYv#reviews"><img src="https://sebastianraschka.com//images/LLMs-from-scratch-images/other/reviews.png" width="220px"></a>


<br>
<br>

Bu deponun bir kopyasını indirmek için [Download ZIP](https://github.com/rasbt/LLMs-from-scratch/archive/refs/heads/main.zip) düğmesine tıklayın veya terminalinizde şu komutu çalıştırın:

```bash
git clone --depth 1 https://github.com/rasbt/LLMs-from-scratch.git
```

<br>

(Kod paketini Manning web sitesinden indirdiyseniz, en güncel değişiklikler için lütfen GitHub'daki resmî kod deposunu ziyaret etmeyi düşünün: [https://github.com/rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch))

<br>
<br>


# İçindekiler

Bu `README.md` dosyasının bir Markdown (`.md`) dosyası olduğunu lütfen unutmayın. Bu kod paketini Manning web sitesinden indirdiyseniz ve yerel bilgisayarınızda görüntülüyorsanız, düzgün görüntüleme için bir Markdown editörü veya önizleyicisi kullanmanızı öneririm. Henüz bir Markdown editörü kurmadıysanız, [Ghostwriter](https://ghostwriter.kde.org) iyi ve ücretsiz bir seçenektir.

Alternatif olarak bu ve diğer dosyaları GitHub'da tarayıcınızdan görüntüleyebilirsiniz ([https://github.com/rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)); GitHub Markdown'ı otomatik olarak işler.

<br>
<br>


> **İpucu:**
> Python ve Python paketlerini kurma ile kod ortamınızı ayarlama konusunda rehberlik arıyorsanız, [setup](setup) dizinindeki [README.md](setup/README.md) dosyasını okumanızı öneririm.

<br>
<br>

[![Code tests Linux](https://github.com/rasbt/LLMs-from-scratch/actions/workflows/basic-tests-linux-uv.yml/badge.svg)](https://github.com/rasbt/LLMs-from-scratch/actions/workflows/basic-tests-linux-uv.yml)
[![Code tests Windows](https://github.com/rasbt/LLMs-from-scratch/actions/workflows/basic-tests-windows-uv-pip.yml/badge.svg)](https://github.com/rasbt/LLMs-from-scratch/actions/workflows/basic-tests-windows-uv-pip.yml)
[![Code tests macOS](https://github.com/rasbt/LLMs-from-scratch/actions/workflows/basic-tests-macos-uv.yml/badge.svg)](https://github.com/rasbt/LLMs-from-scratch/actions/workflows/basic-tests-macos-uv.yml)

- [Sorun Giderme Rehberi](./troubleshooting.md)


| Bölüm Başlığı                                              | Ana Kod (Hızlı Erişim İçin)                                                                                                    | Tüm Kod + Ek Materyaller      |
|------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------|-------------------------------|
| [Kurulum önerileri](setup) <br/>[Bu kitap en iyi nasıl okunur](https://sebastianraschka.com/blog/2025/reading-books.html)                            | -                                                                                                                               | -                             |
| Böl. 1: Büyük Dil Modellerini Anlamak                  | Kod yok                                                                                                                         | -                             |
| Böl. 2: Metin Verileriyle Çalışmak                               | - [ch02.ipynb](ch02/01_main-chapter-code/ch02.ipynb)<br/>- [dataloader.ipynb](ch02/01_main-chapter-code/dataloader.ipynb) (özet)<br/>- [exercise-solutions.ipynb](ch02/01_main-chapter-code/exercise-solutions.ipynb)               | [./ch02](./ch02)            |
| Böl. 3: Dikkat Mekanizmalarını Kodlamak                          | - [ch03.ipynb](ch03/01_main-chapter-code/ch03.ipynb)<br/>- [multihead-attention.ipynb](ch03/01_main-chapter-code/multihead-attention.ipynb) (özet) <br/>- [exercise-solutions.ipynb](ch03/01_main-chapter-code/exercise-solutions.ipynb)| [./ch03](./ch03)             |
| Böl. 4: Sıfırdan Bir GPT Modeli Uygulamak                | - [ch04.ipynb](ch04/01_main-chapter-code/ch04.ipynb)<br/>- [gpt.py](ch04/01_main-chapter-code/gpt.py) (özet)<br/>- [exercise-solutions.ipynb](ch04/01_main-chapter-code/exercise-solutions.ipynb) | [./ch04](./ch04)           |
| Böl. 5: Etiketlenmemiş Veri Üzerinde Ön Eğitim                        | - [ch05.ipynb](ch05/01_main-chapter-code/ch05.ipynb)<br/>- [gpt_train.py](ch05/01_main-chapter-code/gpt_train.py) (özet) <br/>- [gpt_generate.py](ch05/01_main-chapter-code/gpt_generate.py) (özet) <br/>- [exercise-solutions.ipynb](ch05/01_main-chapter-code/exercise-solutions.ipynb) | [./ch05](./ch05)              |
| Böl. 6: Metin Sınıflandırma İçin İnce Ayar                   | - [ch06.ipynb](ch06/01_main-chapter-code/ch06.ipynb)  <br/>- [gpt_class_finetune.py](ch06/01_main-chapter-code/gpt_class_finetune.py)  <br/>- [exercise-solutions.ipynb](ch06/01_main-chapter-code/exercise-solutions.ipynb) | [./ch06](./ch06)              |
| Böl. 7: Talimatları İzlemek İçin İnce Ayar                    | - [ch07.ipynb](ch07/01_main-chapter-code/ch07.ipynb)<br/>- [gpt_instruction_finetuning.py](ch07/01_main-chapter-code/gpt_instruction_finetuning.py) (özet)<br/>- [ollama_evaluate.py](ch07/01_main-chapter-code/ollama_evaluate.py) (özet)<br/>- [exercise-solutions.ipynb](ch07/01_main-chapter-code/exercise-solutions.ipynb) | [./ch07](./ch07)  |
| Ek A: PyTorch'a Giriş                        | - [code-part1.ipynb](appendix-A/01_main-chapter-code/code-part1.ipynb)<br/>- [code-part2.ipynb](appendix-A/01_main-chapter-code/code-part2.ipynb)<br/>- [DDP-script.py](appendix-A/01_main-chapter-code/DDP-script.py)<br/>- [exercise-solutions.ipynb](appendix-A/01_main-chapter-code/exercise-solutions.ipynb) | [./appendix-A](./appendix-A) |
| Ek B: Kaynaklar ve İleri Okuma                 | Kod yok                                                                                                                         | [./appendix-B](./appendix-B) |
| Ek C: Alıştırma Çözümleri                             | - [alıştırma çözümlerinin listesi](appendix-C)                                                                 | [./appendix-C](./appendix-C) |
| Ek D: Eğitim Döngüsüne Ek Özellikler Eklemek | - [appendix-D.ipynb](appendix-D/01_main-chapter-code/appendix-D.ipynb)                                                          | [./appendix-D](./appendix-D)  |
| Ek E: LoRA ile Parametre Açısından Verimli İnce Ayar       | - [appendix-E.ipynb](appendix-E/01_main-chapter-code/appendix-E.ipynb)                                                          | [./appendix-E](./appendix-E) |

<br>
&nbsp;

Aşağıdaki zihinsel model, bu kitapta ele alınan içeriği özetler.

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/mental-model.jpg" width="650px">


<br>
&nbsp;

## Ön Koşullar

En önemli ön koşul, Python programlamada sağlam bir temeldir.
Bu bilgiyle, LLM'lerin büyüleyici dünyasını keşfetmeye ve bu kitapta sunulan kavramlar ile kod örneklerini anlamaya hazır olacaksınız.

Derin sinir ağlarıyla biraz deneyiminiz varsa, LLM'ler bu mimariler üzerine kurulduğu için bazı kavramları daha tanıdık bulabilirsiniz.

Bu kitap, kodu herhangi bir harici LLM kütüphanesi kullanmadan sıfırdan uygulamak için PyTorch'u kullanır. PyTorch'ta uzmanlık bir ön koşul olmasa da, PyTorch temellerine aşinalık kesinlikle faydalıdır. PyTorch'a yeniyseniz, Ek A kısa bir PyTorch girişi sunar. Alternatif olarak, temelleri öğrenmek için [PyTorch in One Hour: From Tensors to Training Neural Networks on Multiple GPUs](https://sebastianraschka.com/teaching/pytorch-1h/) adlı kitabımı faydalı bulabilirsiniz.



<br>
&nbsp;

## Donanım Gereksinimleri

Bu kitabın ana bölümlerindeki kod, sıradan dizüstü bilgisayarlarda makul bir sürede çalışacak şekilde tasarlanmıştır ve özel donanım gerektirmez. Bu yaklaşım, geniş bir kitlenin materyalle etkileşim kurabilmesini sağlar. Ayrıca kod, mevcutsa GPU'ları otomatik olarak kullanır. (Ek öneriler için lütfen [setup](https://github.com/rasbt/LLMs-from-scratch/blob/main/setup/README.md) belgesine bakın.)


&nbsp;
## Video Kursu

[17 saat 15 dakikalık eşlik eden bir video kursunda](https://www.manning.com/livevideo/master-and-build-large-language-models) kitabın her bölümünü kodlayarak anlatıyorum. Kurs, kitabın yapısını yansıtan bölümler ve kısımlar hâlinde düzenlenmiştir; böylece kitaba bağımsız bir alternatif olarak ya da tamamlayıcı bir kodlama kaynağı olarak kullanılabilir.

<a href="https://www.manning.com/livevideo/master-and-build-large-language-models"><img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/video-screenshot.webp?123" width="350px"></a>


&nbsp;


## Eşlik Eden Kitap / Devamı

[*Build A Reasoning Model (From Scratch)*](https://mng.bz/lZ5B) bağımsız bir kitap olmakla birlikte, *Build A Large Language Model (From Scratch)* kitabının devamı olarak değerlendirilebilir.

Önceden eğitilmiş bir modelle başlar ve modelin akıl yürütme yeteneklerini geliştirmek için çıkarım zamanı ölçeklendirme (inference-time scaling), pekiştirmeli öğrenme ve damıtma (distillation) dahil farklı akıl yürütme yaklaşımlarını uygular.

*Build A Large Language Model (From Scratch)* kitabına benzer şekilde, [*Build A Reasoning Model (From Scratch)*](https://mng.bz/lZ5B) de bu yöntemleri sıfırdan uygulayan uygulamalı bir yaklaşım benimser.

<a href="https://mng.bz/lZ5B"><img src="https://sebastianraschka.com/images/reasoning-from-scratch-images/cover.webp?123" width="120px"></a>

- [Amazon bağlantısı](https://amzn.to/4aAKiFY)
- [Manning bağlantısı](https://mng.bz/lZ5B)
- [GitHub deposu](https://github.com/rasbt/reasoning-from-scratch)

> ℹ️ Bu deponun içindeki [reasoning-from-scratch](./reasoning-from-scratch) klasörü, yukarıdaki deponun bir git submodule'üdür ve bu çalışma kopyasında sabitlenmiş sürüme (commit `0acaa28`) göre doldurulmuştur.

<br>

&nbsp;
## Alıştırmalar

Kitabın her bölümü birkaç alıştırma içerir. Çözümler Ek C'de özetlenmiştir ve ilgili kod not defterleri bu deponun ana bölüm klasörlerinde mevcuttur (örneğin, [./ch02/01_main-chapter-code/exercise-solutions.ipynb](./ch02/01_main-chapter-code/exercise-solutions.ipynb)).

Kod alıştırmalarına ek olarak, Manning web sitesinden [Test Yourself On Build a Large Language Model (From Scratch)](https://www.manning.com/books/test-yourself-on-build-a-large-language-model-from-scratch) başlıklı 170 sayfalık ücretsiz bir PDF indirebilirsiniz. Anlayışınızı test etmenize yardımcı olmak için bölüm başına yaklaşık 30 sınav sorusu ve çözümü içerir.

<a href="https://www.manning.com/books/test-yourself-on-build-a-large-language-model-from-scratch"><img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/test-yourself-cover.jpg?123" width="150px"></a>

&nbsp;
## Bonus Materyaller

Birkaç klasör, ilgilenen okuyucular için bonus niteliğinde isteğe bağlı materyaller içerir:
- **Kurulum**
  - [Python Kurulum İpuçları](setup/01_optional-python-setup-preferences)
  - [Bu Kitapta Kullanılan Python Paketlerini ve Kütüphanelerini Kurmak](setup/02_installing-python-libraries)
  - [Docker Ortamı Kurulum Rehberi](setup/03_optional-docker-environment)

- **Bölüm 2: Metin Verileriyle Çalışmak**
  - [Sıfırdan Byte Pair Encoding (BPE) Tokenizer](ch02/05_bpe-from-scratch/bpe-from-scratch-simple.ipynb)
  - [Çeşitli Byte Pair Encoding (BPE) Uygulamalarının Karşılaştırılması](ch02/02_bonus_bytepair-encoder)
  - [Gömme Katmanları ile Doğrusal Katmanlar Arasındaki Farkı Anlamak](ch02/03_bonus_embedding-vs-matmul)
  - [Basit Sayılarla Veri Yükleyici Sezgisi](ch02/04_bonus_dataloader-intuition)
  - [Sıfırdan BPE](ch02/05_bpe-from-scratch)
  - [SimpleTokenizerV3 varyantı](ch02/06_bonus_simple-tokenizer-v3)

- **Bölüm 3: Dikkat Mekanizmalarını Kodlamak**
  - [Verimli Çok Başlı Dikkat Uygulamalarının Karşılaştırılması](ch03/02_bonus_efficient-multihead-attention/mha-implementations.ipynb)
  - [PyTorch Buffer'larını Anlamak](ch03/03_understanding-buffers/understanding-buffers.ipynb)

- **Bölüm 4: Sıfırdan Bir GPT Modeli Uygulamak**
  - [FLOPs Analizi](ch04/02_performance-analysis/flops-analysis.ipynb)
  - [KV Önbelleği](ch04/03_kv-cache)
  - [Dikkat Mekanizmasına Alternatifler](ch04/#attention-alternatives)
    - [Gruplanmış Sorgu Dikkati (GQA)](ch04/04_gqa)
    - [Çok Başlı Gizil Dikkat (MLA)](ch04/05_mla)
    - [Kayan Pencere Dikkati (SWA)](ch04/06_swa)
    - [Gated DeltaNet](ch04/08_deltanet)
    - [DeepSeek Seyrek Dikkati (DSA)](ch04/09_dsa)
    - [Katmanlar Arası KV Paylaşımı](ch04/10_kv-sharing)
  - [Uzmanlar Karışımı (MoE)](ch04/07_moe)

- **Bölüm 5: Etiketlenmemiş Veri Üzerinde Ön Eğitim**
  - [Alternatif Ağırlık Yükleme Yöntemleri](ch05/02_alternative_weight_loading/)
  - [GPT'yi Project Gutenberg Veri Kümesi Üzerinde Ön Eğitmek](ch05/03_bonus_pretraining_on_gutenberg)
  - [Eğitim Döngüsüne Ek Özellikler Eklemek](ch05/04_learning_rate_schedulers)
  - [Ön Eğitim İçin Hiperparametreleri Optimize Etmek](ch05/05_bonus_hparam_tuning)
  - [Önceden Eğitilmiş LLM ile Etkileşim İçin Kullanıcı Arayüzü Oluşturmak](ch05/06_user_interface)
  - [GPT'yi Llama'ya Dönüştürmek](ch05/07_gpt_to_llama)
  - [Bellek Açısından Verimli Model Ağırlığı Yükleme](ch05/08_memory_efficient_weight_loading/memory-efficient-state-dict.ipynb)
  - [Tiktoken BPE Tokenizer'ını Yeni Token'larla Genişletmek](ch05/09_extending-tokenizers/extend-tiktoken.ipynb)
  - [Daha Hızlı LLM Eğitimi İçin PyTorch Performans İpuçları](ch05/10_llm-training-speed)
  - [LLM Mimarileri](ch05/#llm-architectures-from-scratch)
    - [Sıfırdan Llama 3.2](ch05/07_gpt_to_llama/standalone-llama32.ipynb)
    - [Sıfırdan Qwen3 Yoğun ve Uzmanlar Karışımı (MoE)](ch05/11_qwen3/)
    - [Sıfırdan Gemma 3](ch05/12_gemma3/)
    - [Sıfırdan Olmo 3](ch05/13_olmo3/)
    - [Sıfırdan Tiny Aya](ch05/15_tiny-aya/)
    - [Sıfırdan Qwen3.5](ch05/16_qwen3.5/)
    - [Sıfırdan Gemma 4 E2B ve E4B](ch05/17_gemma4/)
  - [Bölüm 5'i Diğer LLM'lerle Doğrudan Değiştirerek Çalıştırmak (ör. Llama 3, Qwen 3)](ch05/14_ch05_with_other_llms/)
- **Bölüm 6: Sınıflandırma için ince ayar**
  - [Farklı Katmanlara İnce Ayar ve Daha Büyük Modellerle Ek Deneyler](ch06/02_bonus_additional-experiments)
  - [50 Bin IMDb Film Yorumu Veri Kümesinde Farklı Modellere İnce Ayar](ch06/03_bonus_imdb-classification)
  - [GPT Tabanlı Spam Sınıflandırıcısıyla Etkileşim İçin Kullanıcı Arayüzü Oluşturmak](ch06/04_user_interface)
- **Bölüm 7: Talimatları izlemek için ince ayar**
  - [Yakın Kopyaları Bulmak ve Edilgen Çatılı Girdiler Oluşturmak İçin Veri Kümesi Araçları](ch07/02_dataset-utilities)
  - [Talimat Yanıtlarını OpenAI API ve Ollama ile Değerlendirmek](ch07/03_model-evaluation)
  - [Talimat İnce Ayarı İçin Veri Kümesi Üretmek](ch07/05_dataset-generation/llama3-ollama.ipynb)
  - [Talimat İnce Ayarı İçin Veri Kümesini İyileştirmek](ch07/05_dataset-generation/reflection-gpt4.ipynb)
  - [Llama 3.1 70B ve Ollama ile Tercih Veri Kümesi Üretmek](ch07/04_preference-tuning-with-dpo/create-preference-data-ollama.ipynb)
  - [LLM Hizalaması İçin Doğrudan Tercih Optimizasyonu (DPO)](ch07/04_preference-tuning-with-dpo/dpo-from-scratch.ipynb)
  - [Talimat İnce Ayarlı GPT Modeliyle Etkileşim İçin Kullanıcı Arayüzü Oluşturmak](ch07/06_user_interface)

[Reasoning From Scratch](https://github.com/rasbt/reasoning-from-scratch) deposundan daha fazla bonus materyal:

- **Qwen3 (Sıfırdan) Temelleri**
  - [Qwen3 Kaynak Kodu İncelemesi](https://github.com/rasbt/reasoning-from-scratch/blob/main/chC/01_main-chapter-code/chC_main.ipynb)
  - [Optimize Edilmiş Qwen3](https://github.com/rasbt/reasoning-from-scratch/tree/main/ch02/03_optimized-LLM)

- **Değerlendirme**
  - [Doğrulayıcı Temelli Değerlendirme (MATH-500)](https://github.com/rasbt/reasoning-from-scratch/tree/main/ch03)
  - [Çoktan Seçmeli Değerlendirme (MMLU)](https://github.com/rasbt/reasoning-from-scratch/blob/main/chF/02_mmlu)
  - [LLM Liderlik Tablosu Değerlendirmesi](https://github.com/rasbt/reasoning-from-scratch/blob/main/chF/03_leaderboards)
  - [Hakem Olarak LLM (LLM-as-a-Judge) Değerlendirmesi](https://github.com/rasbt/reasoning-from-scratch/blob/main/chF/04_llm-judge)
- **Çıkarım Ölçeklendirme**
  - [Öz Tutarlılık (Self-Consistency)](https://github.com/rasbt/reasoning-from-scratch/blob/main/ch04/01_main-chapter-code/ch04_main.ipynb)
  - [Öz İyileştirme (Self-Refinement)](https://github.com/rasbt/reasoning-from-scratch/blob/main/ch05/01_main-chapter-code/ch05_main.ipynb)

- **Pekiştirmeli Öğrenme** (RL)
  - [Sıfırdan GRPO ile RLVR](https://github.com/rasbt/reasoning-from-scratch/blob/main/ch06/01_main-chapter-code/ch06_main.ipynb)


<br>
&nbsp;

## Sorular, Geri Bildirim ve Bu Depoya Katkıda Bulunmak


Her türlü geri bildirimi memnuniyetle karşılıyorum; bunu en iyi [Manning Forum](https://livebook.manning.com/forum?product=raschka&page=1) veya [GitHub Discussions](https://github.com/rasbt/LLMs-from-scratch/discussions) üzerinden paylaşabilirsiniz. Aynı şekilde, sorularınız varsa ya da sadece başkalarıyla fikir alışverişi yapmak istiyorsanız, bunları da foruma göndermekten çekinmeyin.

Bu depo basılı bir kitaba karşılık gelen kodu içerdiğinden, şu anda ana bölüm kodunun içeriğini genişletecek katkıları kabul edemediğimi lütfen unutmayın; çünkü bu, basılı kitaptan sapmalara yol açardı. Tutarlılığı korumak herkes için sorunsuz bir deneyim sağlamaya yardımcı oluyor.


&nbsp;
## Atıf

Bu kitabı veya kodu araştırmanız için faydalı bulursanız, lütfen atıfta bulunmayı düşünün.

Chicago tarzı atıf:

> Raschka, Sebastian. *Build A Large Language Model (From Scratch)*. Manning, 2024. ISBN: 978-1633437166.

BibTeX girdisi:

```
@book{build-llms-from-scratch-book,
  author       = {Sebastian Raschka},
  title        = {Build A Large Language Model (From Scratch)},
  publisher    = {Manning},
  year         = {2024},
  isbn         = {978-1633437166},
  url          = {https://www.manning.com/books/build-a-large-language-model-from-scratch},
  github       = {https://github.com/rasbt/LLMs-from-scratch}
}
```
