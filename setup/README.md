# İsteğe Bağlı Kurulum Talimatları

> 🇹🇷 **Türkçe çeviri.** Orijinal İngilizce sürüm: [README.md](https://github.com/rasbt/LLMs-from-scratch/blob/main/setup/README.md) · Komutlar birebir korunmuştur.


Bu belge, makinenizi kurmak ve bu depodaki kodu kullanmak için farklı yaklaşımları listeler. Farklı bölümlere yukarıdan aşağıya göz atıp ardından hangi yaklaşımın ihtiyaçlarınıza en uygun olduğuna karar vermenizi öneririm.

&nbsp;

## Hızlı Başlangıç

Makinenizde zaten bir Python kurulumu varsa, başlamanın en hızlı yolu, bu kod deposunun kök dizininde aşağıdaki pip kurulum komutunu çalıştırarak [../requirements.txt](../requirements.txt) dosyasındaki paket gereksinimlerini kurmaktır:

```bash
pip install -r requirements.txt
```

<br>

> **Not:** Not defterlerinden herhangi birini Google Colab'da çalıştırıyor ve bağımlılıkları kurmak istiyorsanız, not defterinin en üstüne yeni bir hücre ekleyip şu kodu çalıştırmanız yeterlidir:
> `pip install uv && uv pip install --system -r https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/refs/heads/main/requirements.txt`
> İsteğe bağlı olarak, depoyu klonladıktan sonra proje kökünde `uv pip install --group bonus` komutuyla tüm bonus materyallerin bağımlılıklarını kurabilirsiniz. Bu, isteğe bağlı bonus materyallere daha sonra bakarken bunları ayrı ayrı kurmak istemiyorsanız faydalıdır.



Aşağıdaki videoda, bilgisayarımda bir Python ortamı kurarken izlediğim kişisel yaklaşımı paylaşıyorum:

<br>
<br>

[![Videoya bağlantı](https://img.youtube.com/vi/yAcWnfsZhzo/0.jpg)](https://www.youtube.com/watch?v=yAcWnfsZhzo)


&nbsp;
# Yerel Kurulum

Bu bölüm, bu kitaptaki kodu yerelde çalıştırmaya dair öneriler sunar. Kitabın ana bölümlerindeki kodun, sıradan dizüstü bilgisayarlarda makul bir sürede çalışacak şekilde tasarlandığını ve özel donanım gerektirmediğini unutmayın. Tüm ana bölümleri bir M3 MacBook Air dizüstü bilgisayarda test ettim. Ayrıca, dizüstü veya masaüstü bilgisayarınızda bir NVIDIA GPU varsa kod bundan otomatik olarak yararlanacaktır.

&nbsp;
## Python'u kurmak

Makinenizde Python henüz kurulu değilse, kişisel Python kurulum tercihlerimi şu dizinlerde yazdım:

- [01_optional-python-setup-preferences](./01_optional-python-setup-preferences)
- [02_installing-python-libraries](./02_installing-python-libraries)

Aşağıdaki *DevContainer Kullanmak* bölümü, proje bağımlılıklarını makinenize kurmak için alternatif bir yaklaşımı özetler.

&nbsp;

## Docker DevContainer Kullanmak

Yukarıdaki *Python'u kurmak* bölümüne alternatif olarak, bir projenin bağımlılıklarını ve yapılandırmalarını yalıtan bir geliştirme kurulumunu tercih ediyorsanız Docker kullanmak son derece etkili bir çözümdür. Bu yaklaşım, yazılım paketlerini ve kütüphaneleri elle kurma ihtiyacını ortadan kaldırır ve tutarlı bir geliştirme ortamı sağlar. Docker kurulumuna ve DevContainer kullanımına dair daha fazla talimatı burada bulabilirsiniz:

- [03_optional-docker-environment](03_optional-docker-environment)

&nbsp;

## Visual Studio Code Editörü

Kod editörleri için pek çok iyi seçenek var. Benim tercihim, birçok faydalı eklenti ve uzantıyla kolayca zenginleştirilebilen popüler açık kaynaklı [Visual Studio Code (VSCode)](https://code.visualstudio.com) editörü (daha fazla bilgi için aşağıdaki *VSCode Uzantıları* bölümüne bakın). macOS, Linux ve Windows için indirme talimatları [VSCode ana sitesinde](https://code.visualstudio.com) bulunabilir.

&nbsp;

## VSCode Uzantıları

Ana kod editörünüz olarak Visual Studio Code (VSCode) kullanıyorsanız, önerilen uzantıları `.vscode` alt klasöründe bulabilirsiniz. Bu uzantılar, bu depo için faydalı olan gelişmiş işlevsellik ve araçlar sunar.

Bunları kurmak için bu "setup" klasörünü VSCode'da açın (File -> Open Folder...) ve ardından sağ altta beliren açılır menüdeki "Install" düğmesine tıklayın.

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/README/vs-code-extensions.webp?1" alt="1" width="700">

Alternatif olarak, `.vscode` uzantı klasörünü bu GitHub deposunun kök dizinine taşıyabilirsiniz:

```bash
mv setup/.vscode ./
```

Böylece VSCode, `LLMs-from-scratch` ana klasörünü her açtığınızda önerilen uzantıların sisteminizde kurulu olup olmadığını otomatik olarak kontrol eder.

&nbsp;

# Bulut Kaynakları

Bu bölüm, kitapta sunulan kodu çalıştırmak için bulut alternatiflerini açıklar.

Kod, özel bir GPU olmadan sıradan dizüstü ve masaüstü bilgisayarlarda çalışabilse de, NVIDIA GPU'lu bulut platformları kodun çalışma süresini özellikle 5 ila 7. bölümlerde kayda değer ölçüde iyileştirebilir.

&nbsp;

## Lightning Studio Kullanmak

Bulutta sorunsuz bir geliştirme deneyimi için, kullanıcıların kalıcı bir ortam kurmasına ve bulut CPU'ları ile GPU'larında hem VSCode hem de Jupyter Lab kullanmasına olanak tanıyan [Lightning AI Studio](https://lightning.ai/) platformunu öneririm.

Yeni bir Studio başlattıktan sonra terminali açıp depoyu klonlamak ve bağımlılıkları kurmak için şu kurulum adımlarını çalıştırabilirsiniz:

```bash
git clone https://github.com/rasbt/LLMs-from-scratch.git
cd LLMs-from-scratch
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

# Sorularınız mı var?

Herhangi bir sorunuz varsa, lütfen bu GitHub deposundaki [Discussions](https://github.com/rasbt/LLMs-from-scratch/discussions) forumu üzerinden bize ulaşmaktan çekinmeyin.
