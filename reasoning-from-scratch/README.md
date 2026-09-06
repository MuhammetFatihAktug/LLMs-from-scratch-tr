# Build A Reasoning Model (From Scratch) — Sıfırdan Akıl Yürütme Modeli Geliştirme

> 🇹🇷 **Türkçe çeviri.** Orijinal İngilizce sürüm: [README.md](https://github.com/rasbt/reasoning-from-scratch/blob/main/README.md) · Tüm bağlantılar orijinal hedeflerine işaret eder.
>
> ℹ️ Bu depo, `LLMs-from-scratch` deposunun içine bir git submodule olarak yerleştirilmiştir. Türkçe okuma sırası için üst dizindeki [OKUMA_SIRASI.md](../OKUMA_SIRASI.md) dosyasına bakabilirsiniz.

Bu depo, bir LLM akıl yürütme modeli geliştirmek için gerekli kodu içerir ve [*Build a Reasoning Model (From Scratch)*](https://mng.bz/lZ5B) kitabının resmî kod deposudur.


<br>
<br>

<a href="https://mng.bz/lZ5B"><img src="https://sebastianraschka.com/images/reasoning-from-scratch-images/cover.webp?123" width="250px"></a>

(Renkli basılmıştır.)

<br>

[*Build a Reasoning Model (From Scratch)*](https://mng.bz/lZ5B) kitabında, bir akıl yürütme büyük dil modelinin (LLM) nasıl çalıştığını öğrenip anlayacaksınız.

Akıl yürütme (reasoning), LLM'leri geliştirmedeki en heyecan verici ve önemli son gelişmelerden biridir; ancak yalnızca terimi duyup teoride okuduğunuzda en kolay yanlış anlaşılabilecek konulardan biri de odur. Bu kitabın uygulamalı bir yaklaşım benimsemesinin nedeni budur. Önceden eğitilmiş bir temel LLM ile başlayacak, ardından akıl yürütme yeteneklerini adım adım, kod yazarak kendimiz ekleyeceğiz; böylece nasıl çalıştığını tam olarak görebileceksiniz.

Bu kitapta anlatılan yöntemler, eğitim amaçlı olarak kendi küçük ama işlevsel akıl yürütme modelinizi geliştirme sürecinde size rehberlik eder. Bu, DeepSeek R1, GPT-5 Thinking ve diğerleri gibi büyük ölçekli akıl yürütme modellerinin oluşturulmasında kullanılan yaklaşımların bir yansımasıdır. Ayrıca bu kitap, mevcut önceden eğitilmiş modellerin ağırlıklarını yükleyen kodu da içerir.

- Resmî [kaynak kod deposunun](https://github.com/rasbt/reasoning-from-scratch) bağlantısı
- [Manning'deki kitabın](https://mng.bz/lZ5B) bağlantısı (yayıncının web sitesi)
- Kitabın Amazon.com sayfasının bağlantısı (henüz belirlenmedi)
- ISBN 9781633434677



<br>
<br>

Bu deponun bir kopyasını indirmek için [Download ZIP](https://github.com/rasbt/reasoning-from-scratch/archive/refs/heads/main.zip) düğmesine tıklayın veya terminalinizde şu komutu çalıştırın:

```bash
git clone --depth 1 https://github.com/rasbt/reasoning-from-scratch.git
```

<br>


> **İpucu:**
> 2. bölüm; Python kurulumu, Python paketlerinin yönetimi ve kodlama ortamınızın hazırlanması hakkında ek ipuçları sunar.

<br>
<br>

## İçindekiler (Devam Ediyor)

[![Code tests Linux](https://github.com/rasbt/reasoning-from-scratch/actions/workflows/tests-linux.yml/badge.svg)](https://github.com/rasbt/reasoning-from-scratch/actions/workflows/tests-linux.yml)
[![Code tests macOS](https://github.com/rasbt/reasoning-from-scratch/actions/workflows/tests-macos.yml/badge.svg)](https://github.com/rasbt/reasoning-from-scratch/actions/workflows/tests-macos.yml)
[![Code tests Windows](https://github.com/rasbt/reasoning-from-scratch/actions/workflows/tests-windows.yml/badge.svg)](https://github.com/rasbt/reasoning-from-scratch/actions/workflows/tests-windows.yml)

- [Sorun Giderme Rehberi](./troubleshooting.md)

| Bölüm Başlığı                                               | Ana Kod                                                      |
| ----------------------------------------------------------- | ------------------------------------------------------------ |
| Böl. 1: Akıl Yürütme Modellerini Anlamak                    | Kod yok                                                      |
| Böl. 2: Önceden Eğitilmiş Bir LLM ile Metin Üretmek         | - [ch02_main.ipynb](ch02/01_main-chapter-code/ch02_main.ipynb)<br/>- [ch02_exercise-solutions.ipynb](ch02/01_main-chapter-code/ch02_exercise-solutions.ipynb) |
| Böl. 3: Akıl Yürütme Modellerini Değerlendirmek             | - [ch03_main.ipynb](ch03/01_main-chapter-code/ch03_main.ipynb)<br/>- [ch03_exercise-solutions.ipynb](ch03/01_main-chapter-code/ch03_exercise-solutions.ipynb) |
| Böl. 4: Çıkarım Zamanı Ölçeklendirme ile Akıl Yürütmeyi İyileştirmek | - [ch04_main.ipynb](ch04/01_main-chapter-code/ch04_main.ipynb)<br/>- [ch04_exercise-solutions.ipynb](ch04/01_main-chapter-code/ch04_exercise-solutions.ipynb) |
| Böl. 5: Öz İyileştirme ile Çıkarım Zamanı Ölçeklendirme     | - [ch05_main.ipynb](ch05/01_main-chapter-code/ch05_main.ipynb)<br/>- [ch05_exercise-solutions.ipynb](ch05/01_main-chapter-code/ch05_exercise-solutions.ipynb) |
| Böl. 6: Pekiştirmeli Öğrenme ile Akıl Yürütme Modellerini Eğitmek | - [ch06_main.ipynb](ch06/01_main-chapter-code/ch06_main.ipynb)<br/>- [ch06_exercise-solutions.ipynb](ch06/01_main-chapter-code/ch06_exercise-solutions.ipynb) |
| Böl. 7: Pekiştirmeli Öğrenme İçin GRPO'yu İyileştirmek      | - [ch07_main.ipynb](ch07/01_main-chapter-code/ch07_main.ipynb)<br/>- [ch07_exercise-solutions.ipynb](ch07/01_main-chapter-code/ch07_exercise-solutions.ipynb) |
| Böl. 8: Verimli Akıl Yürütme İçin Modelleri Damıtmak        | - [ch08_main.ipynb](ch08/01_main-chapter-code/ch08_main.ipynb)<br/>- [ch08_exercise-solutions.ipynb](ch08/01_main-chapter-code/ch08_exercise-solutions.ipynb) |
| Ek A: Kaynaklar ve İleri Okuma                              | Kod yok                                                      |
| Ek B: Alıştırma Çözümleri                                   | Kod ve çözümler her bölümün alt klasöründedir                |
| Ek C: Qwen3 LLM Kaynak Kodu                                 | - [chC_main.ipynb](chC/01_main-chapter-code/chC_main.ipynb)  |
| Ek D: Daha büyük LLM'leri kullanmak                         | - [chD_main.ipynb](chD/chD_main.ipynb)                       |
| Ek E: Yığınlama ve verim odaklı çalıştırma                  | - [chE_main.ipynb](chE/chE_main.ipynb)                       |
| Ek F: LLM Değerlendirmesine Yaygın Yaklaşımlar              | - [chF_main.ipynb](chF/01_main-chapter-code/chF_main.ipynb)  |
| Ek G: Bir Sohbet Arayüzü Oluşturmak                         | - [chG](chG)                                                 |

<br>
&nbsp;

Aşağıdaki zihinsel model, bu kitapta ele alınan başlıca teknikleri özetler.

<img src="https://sebastianraschka.com/images/reasoning-from-scratch-images/mental-model.webp" width="650px">



<br>



&nbsp;
## Eşlik Eden Kitap

*Build A Reasoning Model (From Scratch)* kitabının, LLM akıl yürütmesini iyileştirme yöntemlerine odaklanan bağımsız bir kitap olduğunu lütfen unutmayın.

Bu kitapta, önceden eğitilmiş açık kaynaklı bir temel LLM (Qwen3) ile çalışıyor ve bunun üzerine akıl yürütme yöntemlerini sıfırdan kodluyoruz. Buna çıkarım zamanı ölçeklendirme, pekiştirmeli öğrenme ve damıtma dahildir.

Ancak klasik bir temel LLM'in nasıl uygulandığını anlamakla ilgileniyorsanız, önceki kitabım [*Build a Large Language Model (From Scratch)*](https://amzn.to/4fqvn0D) ilginizi çekebilir.

<a href="https://amzn.to/4fqvn0D"><img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/cover.jpg?123" width="120px"></a>

- [Amazon bağlantısı](https://amzn.to/4fqvn0D)
- [Manning bağlantısı](http://mng.bz/orYv)
- [GitHub deposu](https://github.com/rasbt/LLMs-from-scratch)


<br>
&nbsp;

## Donanım Gereksinimleri

Bu kitabın ana bölümlerindeki kod, çoğunlukla tüketici donanımında makul bir sürede çalışacak şekilde tasarlanmıştır ve özel sunucu donanımı gerektirmez. Bu yaklaşım, geniş bir kitlenin materyalle etkileşim kurabilmesini sağlar. Ayrıca kod, mevcutsa GPU'ları otomatik olarak kullanır. Bununla birlikte, 2-4. bölümler hem CPU'larda hem GPU'larda iyi çalışacaktır. 5. ve 6. bölümler için, bölümdeki sonuçları yeniden üretmek istiyorsanız bir GPU kullanmanız önerilir.


(Ek öneriler için lütfen [setup_tips](ch02/02_setup-tips/python-instructions.md) belgesine bakın.)

&nbsp;
## Alıştırmalar

Kitabın her bölümü birkaç alıştırma içerir. Çözümler Ek B'de özetlenmiştir ve ilgili kod not defterleri bu deponun ana bölüm klasörlerinde mevcuttur (örneğin, [`ch02/01_main-chapter-code/ch02_exercise-solutions.ipynb`](ch02/01_main-chapter-code/ch02_exercise-solutions.ipynb)).


&nbsp;
## Bonus Materyal

Birkaç klasör, ilgilenen okuyucular için bonus niteliğinde isteğe bağlı materyaller içerir:

- **Bölüm 2: Önceden Eğitilmiş Bir LLM ile Metin Üretmek**
  - [İsteğe Bağlı Python Kurulumu ve Bulut GPU Önerileri](ch02/02_setup-tips)
  - [LLM'in GPU için optimize edilmiş sürümünü kullanmak](ch02/03_optimized-LLM)
  - [Windows'ta `torch.compile()` kullanmak](ch02/04_torch-compile-windows)
  - [Modelle çıkarım çalıştırmak ve sohbet etmek](ch02/05_use_model)
- **Bölüm 3: LLM'leri Değerlendirmek**
  - [MATH-500 Doğrulayıcı Betikleri](ch03/02_math500-verifier-scripts)
  - [Gelişmiş Ayrıştırıcı](ch03/03_advanced-parser) (hibrit LaTeX ayrıştırıcısı)
- **Bölüm 4: Çıkarım Zamanı Ölçeklendirme ile Akıl Yürütmeyi İyileştirmek**
  - [MATH-500 Üzerinde Çıkarım Ölçeklendirme](ch04/02_math500-inference-scaling-scripts) (CoT istemi, öz tutarlılık)
- **Bölüm 5: Öz İyileştirme ile Çıkarım Zamanı Ölçeklendirme**
  - [MATH-500 Üzerinde Daha Fazla Çıkarım Ölçeklendirme](ch05/02_math500-more-inference-scaling-scripts) (Best-of-N, öz iyileştirme)
- **Bölüm 6: Pekiştirmeli Öğrenme ile Akıl Yürütme Modellerini Eğitmek**
  - Yığınlanmış moda sahip [GRPO betikleri](ch06/02_rlvr_grpo_scripts_intro)
- **Bölüm 7: Pekiştirmeli Öğrenme İçin GRPO'yu İyileştirmek**
  - [Gelişmiş GRPO betikleri](ch07/03_rlvr_grpo_scripts_advanced) (DeepSeek-V3.2, Olmo3 ve GDPO tarzı eğitim dahil)
  - [Eğitim kontrol noktalarını indirmek](ch07/04_download_trainining_checkpoints) (6. ve 7. bölümün GRPO kontrol noktalarının nasıl indirileceği ve kullanılacağı)
- **Bölüm 8: Verimli Akıl Yürütme İçin Akıl Yürütme Modellerini Damıtmak**
  - [Damıtma verisi üretmek](ch08/02_generate_distillation_data) (Ollama veya OpenRouter ile öğretmen çıktısı üretimi)
  - [Damıtma ile eğitmek](ch08/04_train_with_distillation) (tek örnekli ve yığınlanmış damıtma betikleri dahil)
  - [Eğitim kontrol noktalarını indirmek](ch08/05_download_training_checkpoints) (8. bölümün damıtma kontrol noktalarının nasıl indirileceği ve kullanılacağı)
  - [Qwen3'ü Hugging Face ile kullanmak](ch08/06_use_via_huggingface) (temel modelin ve 6-8. bölüm kontrol noktalarının `transformers` ile nasıl kullanılacağı)
- **Ek F: LLM Değerlendirmesine Yaygın Yaklaşımlar**
  - [MMLU Değerlendirme Yöntemleri](chF/02_mmlu)
  - [LLM liderlik tabloları](chF/03_leaderboards)
  - [Hakem olarak LLM (LLM-as-a-judge)](chF/04_llm-judge)
- **Ek G: Bir Sohbet Arayüzü Oluşturmak**
  - [Sohbet arayüzü kodu](chG/01_main-chapter-code)


&nbsp;
## Sorular, Geri Bildirim ve Bu Depoya Katkıda Bulunmak

Yaygın sorunlar için lütfen [Sorun Giderme Rehberi](./troubleshooting.md) belgesine bakın.

Her türlü geri bildirimi memnuniyetle karşılıyorum; bunu en iyi [Manning Discussion Forum](https://livebook.manning.com/forum?product=raschka2&page=1) veya [GitHub Discussions](https://github.com/rasbt/reasoning-from-scratch/discussions) üzerinden paylaşabilirsiniz. Aynı şekilde, sorularınız varsa ya da sadece başkalarıyla fikir alışverişi yapmak istiyorsanız, bunları da foruma göndermekten çekinmeyin.

Bu depo basılı bir kitaba karşılık gelen kodu içerdiğinden, şu anda ana bölüm kodunun içeriğini genişletecek katkıları kabul edemediğimi lütfen unutmayın; çünkü bu, basılı kitaptan sapmalara yol açardı. Tutarlılığı korumak herkes için sorunsuz bir deneyim sağlamaya yardımcı oluyor.

&nbsp;
## Atıf

Bu kitabı veya kodu araştırmanız için faydalı bulursanız, lütfen atıfta bulunmayı düşünün.

Chicago tarzı atıf:

> Raschka, Sebastian. *Build A Reasoning Model (From Scratch)*. Manning, 2025. ISBN: 9781633434677.

BibTeX girdisi:

```
@book{build-llms-from-scratch-book,
  author       = {Sebastian Raschka},
  title        = {Build A Reasoning Model (From Scratch)},
  publisher    = {Manning},
  year         = {2025},
  isbn         = {9781633434677},
  url          = {https://mng.bz/lZ5B},
  github       = {https://github.com/rasbt/reasoning-from-scratch}
}
```
