# Python Kurulum İpuçları

> 🇹🇷 **Türkçe çeviri.** Orijinal İngilizce sürüm: [README.md](https://github.com/rasbt/LLMs-from-scratch/blob/main/setup/01_optional-python-setup-preferences/README.md) · Komutlar birebir korunmuştur.



Python'u kurmanın ve bilgi işlem ortamınızı ayarlamanın birkaç yolu vardır. Burada kişisel tercihlerimi paylaşıyorum.

<br>

> **Not:**
> Not defterlerinden herhangi birini Google Colab'da çalıştırıyor ve bağımlılıkları kurmak istiyorsanız, not defterinin en üstüne yeni bir hücre ekleyip şu kodu çalıştırmanız ve bu öğreticinin geri kalanını atlamanız yeterlidir:
> `pip install uv && uv pip install --system -r https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/refs/heads/main/requirements.txt`

Aşağıdaki bölümler, Python ortamınızı ve paketlerinizi yerel makinenizde nasıl yönetebileceğinizi açıklar.

Uzun süredir [Conda](https://anaconda.org/anaconda/conda) ve [pip](https://pypi.org/project/pip/) kullanıcısıyım; ancak son zamanlarda [uv](https://github.com/astral-sh/uv) paketi, paketleri kurmanın ve bağımlılıkları çözmenin daha hızlı ve verimli bir yolunu sunduğu için kayda değer bir ilgi görüyor.

2025'te daha modern yaklaşım olduğu için *Seçenek 1: uv Kullanmak* ile başlamanızı öneririm. *Seçenek 1* ile ilgili sorun yaşarsanız *Seçenek 2: Conda Kullanmak* seçeneğini değerlendirin.

Bu öğreticide macOS çalıştıran bir bilgisayar kullanıyorum, ancak bu iş akışı Linux makineler için de benzerdir ve diğer işletim sistemlerinde de çalışabilir.


&nbsp;
# Seçenek 1: uv Kullanmak

Bu bölüm, `uv` aracını `uv pip` arayüzü üzerinden kullanarak Python kurulumu ve paket yükleme sürecinde size rehberlik eder. `uv pip` arayüzü, daha önce pip kullanmış çoğu Python kullanıcısına yerel `uv` komutlarından daha tanıdık gelebilir.

&nbsp;
> **Not:**
> Python'u kurmanın ve `uv` kullanmanın alternatif yolları vardır. Örneğin, Python'u doğrudan `uv` ile kurabilir ve daha da hızlı paket yönetimi için `uv pip install` yerine `uv add` kullanabilirsiniz.
>
> macOS veya Linux kullanıcısıysanız ve yerel `uv` komutlarını tercih ediyorsanız [./native-uv.md öğreticisine](./native-uv.md) bakın. Ayrıca resmî [`uv` belgelerini](https://docs.astral.sh/uv/) incelemenizi öneririm.
>
> `uv add` söz dizimi Windows kullanıcıları için de geçerlidir. Ancak `pyproject.toml` içindeki bazı bağımlılıkların Windows'ta sorun çıkardığını gördüm. Bu nedenle Windows kullanıcılarına, `uv add` ile benzer bir `pixi add` iş akışına sahip olan `pixi` aracını öneriyorum. Daha fazla bilgi için [./native-pixi.md öğreticisine](./native-pixi.md) bakın.
>
> `uv add` ve `pixi add` ek hız avantajları sunsa da, `uv pip` bence biraz daha kullanıcı dostu; bu da onu yeni başlayanlar için iyi bir başlangıç noktası yapıyor. Yine de Python paket yönetimine yeniyseniz, yerel `uv` arayüzü baştan öğrenmek için harika bir fırsat. Ben de `uv` aracını artık böyle kullanıyorum, ancak `pip` ve `conda` alışkanlığıyla geliyorsanız giriş engelinin biraz daha yüksek olduğunun farkındayım.




&nbsp;
## 1. Python'u kurun (kurulu değilse)

Sisteminize daha önce elle Python kurmadıysanız, bunu yapmanızı şiddetle öneririm. Bu, işletim sisteminizin yerleşik Python kurulumuyla olası çakışmaları ve bunların yol açabileceği sorunları önlemeye yardımcı olur.

Ancak sisteminize daha önce Python kurmuş olsanız bile, terminalde aşağıdaki kodu çalıştırarak güncel bir Python sürümünüz olup olmadığını kontrol edin (3.10 veya daha yenisini öneririm):

```bash
python --version
```
3.10 veya daha yenisini döndürüyorsa başka bir işlem yapmanıza gerek yoktur.

&nbsp;
> **Not:**
> `python --version` komutu hiçbir Python sürümünün kurulu olmadığını gösteriyorsa, sisteminiz `python3` komutunu kullanacak şekilde yapılandırılmış olabileceğinden `python3 --version` komutunu da kontrol etmek isteyebilirsiniz.

&nbsp;
> **Not:**
> PyTorch uyumluluğunu güvence altına almak için, en güncel sürümden en az 2 sürüm eski bir Python sürümü kurmanızı öneririm. Örneğin, en güncel sürüm Python 3.13 ise 3.10 veya 3.11 sürümünü kurmanızı öneririm.

Aksi hâlde, Python kurulu değilse veya eski bir sürümse, aşağıda açıklandığı gibi işletim sisteminize kurabilirsiniz.

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/uv-setup/python-not-found.png" width="500" height="auto" alt="No Python Found">

<br>

**Linux (Ubuntu/Debian)**

```bash
sudo apt update
sudo apt install python3.10 python3.10-venv python3.10-dev
```

<br>

**macOS**

Homebrew kullanıyorsanız Python'u şu komutla kurun:

```bash
brew install python@3.10
```

Alternatif olarak, kurulum dosyasını resmî web sitesinden indirip çalıştırın: [https://www.python.org/downloads/](https://www.python.org/downloads/).


<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/uv-setup/python-version.png" width="700" height="auto" alt="Python version">

<br>

**Windows**

Kurulum dosyasını resmî web sitesinden indirip çalıştırın: [https://www.python.org/downloads/](https://www.python.org/downloads/).


&nbsp;

## 2. Sanal ortam oluşturun

İşletim sisteminizin bağımlı olabileceği sistem genelindeki paketleri değiştirmemek için Python paketlerini ayrı bir sanal ortama kurmanızı şiddetle öneririm. Mevcut klasörde bir sanal ortam oluşturmak için aşağıdaki üç adımı izleyin.

<br>

**1. uv'yi kurun**

```bash
pip install uv
```

<br>

**2. Sanal ortamı oluşturun**

```bash
uv venv --python=python3.10
```

<br>

**3. Sanal ortamı etkinleştirin**

```bash
source .venv/bin/activate
```

&nbsp;
> **Not:**
> Windows kullanıyorsanız, yukarıdaki komutu `source .venv/Scripts/activate` veya `.venv/Scripts/activate` ile değiştirmeniz gerekebilir.



Yeni bir terminal oturumu başlattığınız her seferde sanal ortamı etkinleştirmeniz gerektiğini unutmayın. Örneğin, terminalinizi veya bilgisayarınızı yeniden başlattıysanız ve ertesi gün projeye devam etmek istiyorsanız, sanal ortamınızı yeniden etkinleştirmek için proje klasöründe `source .venv/bin/activate` komutunu çalıştırmanız yeterlidir.

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/uv-setup/venv-activate-1.png" width="600" height="auto" alt="Venv activated">

İsteğe bağlı olarak, `deactivate` komutunu çalıştırarak ortamı devre dışı bırakabilirsiniz.

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/uv-setup/venv-activate-2.png" width="800" height="auto" alt="Venv deactivated">

&nbsp;
## 3. Paketleri kurun

Sanal ortamınızı etkinleştirdikten sonra Python paketlerini `uv` ile kurabilirsiniz. Örneğin:

```bash
uv pip install packaging
```

Bir `requirements.txt` dosyasındaki (örneğin bu GitHub deposunun en üst düzeyinde bulunan) gerekli tüm paketleri kurmak için, dosyanın terminal oturumunuzla aynı dizinde olduğunu varsayarak şu komutu çalıştırın:

```bash
uv pip install -r requirements.txt
```


Alternatif olarak, en güncel bağımlılıkları doğrudan depodan kurun:

```bash
uv pip install -r https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/refs/heads/main/requirements.txt
```


<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/uv-setup/uv-install.png" width="700" height="auto" alt="Uv install">

&nbsp;

> **Not:**
> Yukarıdaki komutlarda bazı bağımlılıklar nedeniyle sorun yaşarsanız (örneğin Windows kullanıyorsanız), her zaman klasik pip'e geri dönebilirsiniz:
> `pip install -r requirements.txt`
> veya
> `pip install -U -r https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/refs/heads/main/requirements.txt`

&nbsp;

> **Bonus materyaller için isteğe bağlı bağımlılıklar:**
> Bonus materyaller boyunca kullanılan isteğe bağlı bağımlılıkları dahil etmek için proje kökünden `bonus` bağımlılık grubunu kurun:
>  `uv pip install --group bonus`
> Bu, isteğe bağlı bonus materyallere daha sonra bakarken bunları ayrı ayrı kurmak istemiyorsanız faydalıdır.

<br>

**Kurulumu tamamlamak**

Hepsi bu kadar! Ortamınız artık depodaki kodu çalıştırmaya hazır olmalı.

İsteğe bağlı olarak, bu depodaki `python_environment_check.py` betiğini çalıştırarak bir ortam kontrolü yapabilirsiniz:

```bash
python setup/02_installing-python-libraries/python_environment_check.py
```

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/uv-setup/env-check.png" width="700" height="auto" alt="Environment check">

Belirli paketlerle ilgili sorun yaşarsanız, bunları şu komutla yeniden kurmayı deneyin:

```bash
uv pip install packagename
```

(Buradaki `packagename`, sorun yaşadığınız paketin adıyla değiştirilmesi gereken bir yer tutucudur.)

Sorunlar devam ederse, GitHub'da [bir tartışma açmayı](https://github.com/rasbt/LLMs-from-scratch/discussions) ya da aşağıdaki *Seçenek 2: Conda Kullanmak* bölümünü uygulamayı değerlendirin.

<br>

**Kodla çalışmaya başlamak**

Her şey hazır olduğunda kod dosyalarıyla çalışmaya başlayabilirsiniz. Örneğin, [JupyterLab](https://jupyterlab.readthedocs.io/en/latest/) uygulamasını şu komutla başlatın:

```bash
jupyter lab
```

&nbsp;
> **Not:**
> jupyter lab komutuyla ilgili sorun yaşarsanız, sanal ortamınızın içindeki tam yolu kullanarak da başlatabilirsiniz. Örneğin, Linux/macOS'ta `.venv/bin/jupyter lab`, Windows'ta ise `.venv\Scripts\jupyter-lab` kullanın.

&nbsp;

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/uv-setup/jupyter.png" width="900" height="auto" alt="Uv install">

&nbsp;
<br>
<br>
&nbsp;

# Seçenek 2: Conda Kullanmak



Bu bölüm, [miniforge](https://github.com/conda-forge/miniforge) aracılığıyla [`conda`](https://www.google.com/search?client=safari&rls=en&q=conda&ie=UTF-8&oe=UTF-8) kullanarak Python kurulumu ve paket yükleme sürecinde size rehberlik eder.

Bu öğreticide macOS çalıştıran bir bilgisayar kullanıyorum, ancak bu iş akışı Linux makineler için de benzerdir ve diğer işletim sistemlerinde de çalışabilir.


&nbsp;
## 1. Miniforge'u indirin ve kurun

Miniforge'u [buradaki](https://github.com/conda-forge/miniforge) GitHub deposundan indirin.

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/01_optional-python-setup-preferences/download.png" alt="download" width="600px">

İşletim sisteminize bağlı olarak bu, ya bir `.sh` (macOS, Linux) ya da bir `.exe` (Windows) dosyası indirmelidir.

`.sh` dosyası için komut satırı terminalinizi açın ve şu komutu çalıştırın:

```bash
sh ~/Desktop/Miniforge3-MacOSX-arm64.sh
```

Burada `Desktop/`, Miniforge kurulum dosyasının indirildiği klasördür. Kendi bilgisayarınızda bunu `Downloads/` ile değiştirmeniz gerekebilir.

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/01_optional-python-setup-preferences/miniforge-install.png" alt="miniforge-install" width="600px">

Ardından, "Enter" ile onaylayarak indirme talimatlarını adım adım izleyin.


&nbsp;
## 2. Yeni bir sanal ortam oluşturun

Kurulum başarıyla tamamlandıktan sonra `LLMs` adında yeni bir sanal ortam oluşturmanızı öneririm; bunu şu komutla yapabilirsiniz:

```bash
conda create -n LLMs python=3.10
```

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/01_optional-python-setup-preferences/new-env.png" alt="new-env" width="600px">

> Birçok bilimsel hesaplama kütüphanesi Python'un en yeni sürümünü hemen desteklemez. Bu nedenle PyTorch kurarken bir veya iki sürüm daha eski bir Python sürümü kullanmanız tavsiye edilir. Örneğin, Python'un en son sürümü 3.13 ise Python 3.10 veya 3.11 kullanmanız önerilir.

Ardından yeni sanal ortamınızı etkinleştirin (yeni bir terminal penceresi veya sekmesi açtığınız her seferde bunu yapmanız gerekir):

```bash
conda activate LLMs
```

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/01_optional-python-setup-preferences/activate-env.png" alt="activate-env" width="600px">


&nbsp;
## İsteğe bağlı: terminalinizi biçimlendirmek

Hangi sanal ortamın etkin olduğunu görebilmek için terminalinizi benimkine benzer şekilde biçimlendirmek isterseniz, [Oh My Zsh](https://github.com/ohmyzsh/ohmyzsh) projesine göz atın.

&nbsp;
## 3. Yeni Python kütüphaneleri kurun



Yeni Python kütüphaneleri kurmak için artık `conda` paket yükleyicisini kullanabilirsiniz. Örneğin, [JupyterLab](https://jupyter.org/install) ve [watermark](https://github.com/rasbt/watermark) paketlerini şöyle kurabilirsiniz:

```bash
conda install jupyterlab watermark
```

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/01_optional-python-setup-preferences/conda-install.png" alt="conda-install" width="600px">



Kütüphane kurmak için `pip` kullanmaya da devam edebilirsiniz. Varsayılan olarak `pip`, yeni `LLms` conda ortamınıza bağlı olmalıdır:

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/01_optional-python-setup-preferences/check-pip.png" alt="check-pip" width="600px">

&nbsp;
## 4. PyTorch'u kurun

PyTorch, diğer Python kütüphaneleri veya paketleri gibi pip ile kurulabilir. Örneğin:

```bash
pip install torch
```

Ancak PyTorch, CPU ve GPU uyumlu kodlar içeren kapsamlı bir kütüphane olduğundan kurulumu ek ayarlar ve açıklama gerektirebilir (daha fazla bilgi için kitaptaki *A.1.3 Installing PyTorch* kısmına bakın).

Ayrıca [https://pytorch.org](https://pytorch.org) adresindeki resmî PyTorch web sitesinde yer alan kurulum kılavuzu menüsüne başvurmanız şiddetle önerilir.

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/01_optional-python-setup-preferences/pytorch-installer.jpg" width="600px">

&nbsp;
## 5. Bu kitapta kullanılan Python paketlerini ve kütüphanelerini kurmak

Gerekli kütüphaneleri kurma talimatları için lütfen [Bu kitapta kullanılan Python paketlerini ve kütüphanelerini kurmak](../02_installing-python-libraries/README.md) belgesine bakın.

<br>

---




Sorularınız mı var? Lütfen [Tartışma Forumu](https://github.com/rasbt/LLMs-from-scratch/discussions) üzerinden bize ulaşmaktan çekinmeyin.
