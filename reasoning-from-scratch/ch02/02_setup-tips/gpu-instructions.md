
# GPU Bulut Kaynakları

> 🇹🇷 **Türkçe çeviri.** Orijinal İngilizce sürüm: [gpu-instructions.md](https://github.com/rasbt/reasoning-from-scratch/blob/main/ch02/02_setup-tips/gpu-instructions.md) · Komutlar birebir korunmuştur.

Bu bölüm, kitapta sunulan kodu çalıştırmak için bulut alternatiflerini açıklar.

Kod, özel bir GPU olmadan sıradan dizüstü ve masaüstü bilgisayarlarda çalışabilse de, NVIDIA GPU'lu bulut platformları kodun çalışma süresini özellikle 5 ila 7. bölümlerde kayda değer ölçüde iyileştirebilir.

&nbsp;

## Lightning Studio Kullanmak

Bulutta sorunsuz bir geliştirme deneyimi için, kullanıcıların kalıcı bir ortam kurmasına ve bulut CPU'ları ile GPU'larında hem VSCode hem de Jupyter Lab kullanmasına olanak tanıyan [Lightning AI Studio](https://lightning.ai/) platformunu öneririm.

Yeni bir Studio başlattıktan sonra terminali açıp depoyu klonlamak ve bağımlılıkları kurmak için şu kurulum adımlarını çalıştırabilirsiniz:

```bash
git clone https://github.com/rasbt/reasoning-from-scratch.git
cd reasoning-from-scratch
pip install -r requirements.txt
```

(Google Colab'ın aksine, Lightning AI Studio ortamları CPU ve GPU makineleri arasında geçiş yapsanız bile kalıcı olduğu için bunların yalnızca bir kez çalıştırılması gerekir.)

Ardından, çalıştırmak istediğiniz Python betiğine veya Jupyter Notebook'a gidin. İsteğe bağlı olarak, örneğin 5. bölümde LLM'i ön eğitirken ya da 6. ve 7. bölümlerde ince ayar yaparken kodun çalışma süresini hızlandırmak için kolayca bir GPU da bağlayabilirsiniz.

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/README/studio.webp" alt="1" width="700">

&nbsp;

## Google Colab Kullanmak

Bulutta bir Google Colab ortamı kullanmak için [https://colab.research.google.com/](https://colab.research.google.com/) adresine gidin ve ilgili bölüm not defterini GitHub menüsünden açın ya da aşağıdaki şekilde gösterildiği gibi not defterini *Upload* alanına sürükleyin.

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/README/colab_1.webp" alt="1" width="700">


Ayrıca, aşağıda gösterildiği gibi ilgili dosyaları (veri kümesi dosyaları ve not defterinin içe aktardığı .py dosyaları) da Colab ortamına yüklediğinizden emin olun.

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/README/colab_2.webp" alt="2" width="700">


İsteğe bağlı olarak, aşağıdaki şekilde gösterildiği gibi *Runtime* ayarını değiştirerek kodu bir GPU üzerinde çalıştırabilirsiniz.

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/README/colab_3.webp" alt="3" width="700">


&nbsp;
## Sorularınız mı var?

Herhangi bir sorunuz varsa, lütfen bu GitHub deposundaki [Discussions](https://github.com/rasbt/reasoning-from-scratch/discussions) forumu üzerinden bize ulaşmaktan çekinmeyin.
