# Yerel (native) pixi ile Python ve paket yönetimi

> 🇹🇷 **Türkçe çeviri.** Orijinal İngilizce sürüm: [native-pixi.md](https://github.com/rasbt/LLMs-from-scratch/blob/main/setup/01_optional-python-setup-preferences/native-pixi.md) · Komutlar birebir korunmuştur.

Bu öğretici, `conda` ve `pip` gibi geleneksel ortam ve paket yöneticileri yerine `pixi` aracının yerel komutlarını tercih edenler için [`./native-uv.md`](native-uv.md) belgesine bir alternatiftir.

[`./native-uv.md`](native-uv.md) belgesinde açıklandığı gibi, pixi'nin arka planda `uv add` kullandığını unutmayın.

Pixi ve uv, Python için modern paket ve ortam yönetim araçlarıdır; ancak pixi yalnızca Python'u değil, başka dilleri de yönetmek üzere tasarlanmış çok dilli (polyglot) bir paket yöneticisiyken (conda'ya benzer şekilde), uv son derece hızlı bağımlılık çözümlemesi ve paket kurulumu için optimize edilmiş, Python'a özgü bir araçtır.

Birden çok dili (yalnızca Python'u değil) destekleyen çok dilli bir paket yöneticisine ihtiyaç duyan ya da conda'ya benzer bildirimsel (declarative) bir ortam yönetimi yaklaşımını tercih eden biri, uv yerine pixi'yi seçebilir. Daha fazla bilgi için lütfen resmî [pixi belgelerini](https://pixi.sh/latest/) ziyaret edin.

Bu öğreticide macOS çalıştıran bir bilgisayar kullanıyorum, ancak bu iş akışı Linux makineler için de benzerdir ve diğer işletim sistemlerinde de çalışabilir.

&nbsp;
## 1. pixi'yi kurun

Pixi, işletim sisteminize bağlı olarak şu şekilde kurulabilir.

<br>

**macOS ve Linux**

```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

veya

```bash
wget -qO- https://pixi.sh/install.sh | sh
```

<br>

**Windows**

Kurulum dosyasını resmî [belgelerden](https://pixi.sh/latest/installation/#__tabbed_1_2) indirin veya orada listelenen PowerShell komutunu çalıştırın.



> **Not:**
> Daha fazla kurulum seçeneği için lütfen resmî [pixi belgelerine](https://pixi.sh/latest/) bakın.


&nbsp;
## 1. Python'u kurun

Python'u pixi ile kurabilirsiniz:

```bash
pixi add python=3.10
```

> **Not:**
> PyTorch uyumluluğunu güvence altına almak için, en güncel sürümden en az 2 sürüm eski bir Python sürümü kurmanızı öneririm. Örneğin, en güncel sürüm Python 3.13 ise 3.10 veya 3.11 sürümünü kurmanızı öneririm. En güncel Python sürümünü [python.org](https://www.python.org) adresini ziyaret ederek öğrenebilirsiniz.

&nbsp;
## 3. Python paketlerini ve bağımlılıkları kurun

Bir `pixi.toml` dosyasındaki (örneğin bu GitHub deposunun en üst düzeyinde bulunan) gerekli tüm paketleri kurmak için, dosyanın terminal oturumunuzla aynı dizinde olduğunu varsayarak şu komutu çalıştırın:

```bash
pixi install
```

> **Not:**
> Bağımlılıklarla ilgili sorun yaşarsanız (örneğin Windows kullanıyorsanız), her zaman pip'e geri dönebilirsiniz: `pixi run pip install -U -r requirements.txt`

Varsayılan olarak `pixi install`, projeye özgü ayrı bir sanal ortam oluşturur.

`pixi.toml` dosyasında belirtilmeyen yeni paketleri `pixi add` ile kurabilirsiniz, örneğin:

```bash
pixi add packaging
```

Ve paketleri `pixi remove` ile kaldırabilirsiniz, örneğin:

```bash
pixi remove packaging
```

&nbsp;
## 4. Python kodunu çalıştırın

Ortamınız artık depodaki kodu çalıştırmaya hazır olmalı.

İsteğe bağlı olarak, bu depodaki `python_environment_check.py` betiğini çalıştırarak bir ortam kontrolü yapabilirsiniz:

```bash
pixi run python setup/02_installing-python-libraries/python_environment_check.py
```

<br>

**JupyterLab'ı başlatmak**

Bir JupyterLab örneğini şu komutla başlatabilirsiniz:

```bash
pixi run jupyter lab
```


---

Sorularınız mı var? Lütfen [Tartışma Forumu](https://github.com/rasbt/LLMs-from-scratch/discussions) üzerinden bize ulaşmaktan çekinmeyin.
