# Yerel (native) uv ile Python ve paket yönetimi

> 🇹🇷 **Türkçe çeviri.** Orijinal İngilizce sürüm: [native-uv.md](https://github.com/rasbt/LLMs-from-scratch/blob/main/setup/01_optional-python-setup-preferences/native-uv.md) · Komutlar birebir korunmuştur.

Bu öğretici, `uv pip` arayüzü yerine `uv` aracının yerel komutlarını tercih edenler için [README.md](./README.md) belgesindeki *Seçenek 1: uv Kullanmak* bölümüne bir alternatiftir. `uv pip`, saf `pip`'ten daha hızlı olsa da, `uv` aracının yerel arayüzü `uv pip`'ten bile daha hızlıdır; çünkü daha az ek yük taşır ve PyPy paket bağımlılığı yönetimi için eski (legacy) desteği ele almak zorunda değildir.

Aşağıdaki tablo, farklı bağımlılık ve paket yönetimi yaklaşımlarının hızlarını karşılaştırır. Hız karşılaştırması özellikle kurulum sırasındaki paket bağımlılığı çözümlemesine işaret eder; kurulan paketlerin çalışma zamanı performansına değil. Bu proje için paket kurulumunun tek seferlik bir işlem olduğunu unutmayın; dolayısıyla tercih edilen yaklaşımı yalnızca kurulum hızına göre değil, genel kullanım kolaylığına göre seçmek makuldür.


| Komut                 | Hız Karşılaştırması |
|-----------------------|-----------------|
| `conda install <pkg>` | En yavaş (Temel çizgi) |
| `pip install <pkg>`   | Yukarıdakinden 2-10× daha hızlı |
| `uv pip install <pkg>`| Yukarıdakinden 5-10× daha hızlı |
| `uv add <pkg>`        | Yukarıdakinden 2-5× daha hızlı |

Bu öğretici `uv add` üzerine odaklanır.


Bunun dışında, [README.md](./README.md) belgesindeki *Seçenek 1: uv Kullanmak* bölümüne benzer şekilde, bu öğretici de `uv` kullanarak Python kurulumu ve paket yükleme sürecinde size rehberlik eder.

Bu öğreticide macOS çalıştıran bir bilgisayar kullanıyorum, ancak bu iş akışı Linux makineler için de benzerdir ve diğer işletim sistemlerinde de çalışabilir.


&nbsp;
## 1. uv'yi kurun

Uv, işletim sisteminize bağlı olarak şu şekilde kurulabilir.

<br>

**macOS ve Linux**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

veya

```bash
wget -qO- https://astral.sh/uv/install.sh | sh
```

<br>

**Windows**

```bash
powershell -c "irm https://astral.sh/uv/install.ps1 | more"
```

&nbsp;

> **Not:**
> Daha fazla kurulum seçeneği için lütfen resmî [uv belgelerine](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer) bakın.

&nbsp;
## 2. Python paketlerini ve bağımlılıkları kurun

Bir `pyproject.toml` dosyasındaki (örneğin bu GitHub deposunun en üst düzeyinde bulunan) gerekli tüm paketleri kurmak için, dosyanın terminal oturumunuzla aynı dizinde olduğunu varsayarak şu komutu çalıştırın:

```bash
uv sync --dev --python 3.11
```

> **Not:**
> Sisteminizde Python 3.11 mevcut değilse, uv sizin için indirip kuracaktır.
> PyTorch uyumluluğunu güvence altına almak için, en güncel sürümden en az 1-3 sürüm eski bir Python sürümü kullanmanızı öneririm. Örneğin, en güncel sürüm Python 3.13 ise 3.10, 3.11 veya 3.12 sürümünü kullanmanızı öneririm. En güncel Python sürümünü [python.org](https://www.python.org/downloads/) adresini ziyaret ederek öğrenebilirsiniz.

> **Not:**
> Yukarıdaki komutlarda bazı bağımlılıklar nedeniyle sorun yaşarsanız (örneğin Windows kullanıyorsanız), her zaman klasik pip'e geri dönebilirsiniz:
> `uv add pip`
> `uv run python -m pip install -U -r requirements.txt`


Yukarıdaki `uv sync` komutunun `.venv` alt klasörü aracılığıyla ayrı bir sanal ortam oluşturacağını unutmayın. (Sıfırdan başlamak için sanal ortamınızı silmek isterseniz, `.venv` klasörünü silmeniz yeterlidir.)

`pyproject.toml` dosyasında belirtilmeyen yeni paketleri `uv add` ile kurabilirsiniz, örneğin:

```bash
uv add packaging
```

Ve paketleri `uv remove` ile kaldırabilirsiniz, örneğin:

```bash
uv remove packaging
```



&nbsp;
## 3. Python kodunu çalıştırın

<br>

Ortamınız artık depodaki kodu çalıştırmaya hazır olmalı.

İsteğe bağlı olarak, bu depodaki `python_environment_check.py` betiğini çalıştırarak bir ortam kontrolü yapabilirsiniz:

```bash
uv run python setup/02_installing-python-libraries/python_environment_check.py
```



<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/uv-setup/uv-run-check.png?1" width="700" height="auto" alt="Uv install">


<br>

**JupyterLab'ı başlatmak**

Bir JupyterLab örneğini şu komutla başlatabilirsiniz:

```bash
uv run jupyter lab
```

**`uv run` komutunu atlamak**

Her seferinde `uv run` yazmayı zahmetli buluyorsanız, sanal ortamı aşağıda açıklandığı gibi elle etkinleştirebilirsiniz.

macOS/Linux'ta:

```bash
source .venv/bin/activate
```

Windows'ta (PowerShell):

```bash
.venv\Scripts\activate
```

Ardından betikleri şu şekilde çalıştırabilirsiniz:

```bash
python script.py
```

ve JupyterLab'ı şu komutla başlatabilirsiniz:

```bash
jupyter lab
```

&nbsp;
> **Not:**
> jupyter lab komutuyla ilgili sorun yaşarsanız, sanal ortamınızın içindeki tam yolu kullanarak da başlatabilirsiniz. Örneğin, Linux/macOS'ta `.venv/bin/jupyter lab`, Windows'ta ise `.venv\Scripts\jupyter-lab` kullanın.

&nbsp;


&nbsp;

## İsteğe bağlı: Sanal ortamları elle yönetmek

Alternatif olarak, bağımlılıkları doğrudan depodan `uv pip install` ile de kurabilirsiniz. Ancak bunun, `uv add` gibi bağımlılıkları bir `uv.lock` dosyasına kaydetmediğini unutmayın. Ayrıca sanal ortamın elle oluşturulmasını ve etkinleştirilmesini gerektirir:

<br>

**1. Yeni bir sanal ortam oluşturun**

Yeni bir `.venv` alt klasörü aracılığıyla kaydedilecek yeni bir sanal ortamı elle oluşturmak için şu komutu çalıştırın:

```bash
uv venv --python=python3.10
```

<br>

**2. Sanal ortamı etkinleştirin**

Ardından bu yeni sanal ortamı etkinleştirmemiz gerekir.

macOS/Linux'ta:

```bash
source .venv/bin/activate
```

Windows'ta (PowerShell):

```bash
.venv\Scripts\activate
```

<br>

**3. Bağımlılıkları kurun**

Son olarak, `uv pip` arayüzünü kullanarak bağımlılıkları uzak bir konumdan kurabiliriz:

```bash
uv pip install -U -r https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/refs/heads/main/requirements.txt
```



---

Sorularınız mı var? Lütfen [Tartışma Forumu](https://github.com/rasbt/LLMs-from-scratch/discussions) üzerinden bize ulaşmaktan çekinmeyin.
