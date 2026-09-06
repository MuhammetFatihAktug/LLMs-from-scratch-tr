# Python Kurulum Önerileri

> 🇹🇷 **Türkçe çeviri.** Orijinal İngilizce sürüm: [python-instructions.md](https://github.com/rasbt/reasoning-from-scratch/blob/main/ch02/02_setup-tips/python-instructions.md) · Komutlar birebir korunmuştur.

Bu kitaptaki kod büyük ölçüde kendi kendine yeterlidir ve harici bağımlılıkları en aza indirmeye çalıştım. Ancak kitabı erişilebilir, okunabilir ve 2000 sayfanın epey altında tutmak için birkaç Python paketi gereklidir.

Bu bölüm, kod örneklerini çalıştırabilmeniz için gerekli paketleri kurmanın yeni başlayanlara uygun iki yöntemini tanıtır.

Elbette Python paketlerini kurmanın ve yönetmenin başka pek çok yolu da vardır. Deneyimli bir Python kullanıcısıysanız ve kendi kurulumunuz ya da tercihleriniz varsa, bu bölümü atlamakta özgürsünüz.

Aşağıdaki iki seçenek de sizde çalışmazsa, lütfen örneğin bir [Discussion](https://github.com/rasbt/reasoning-from-scratch/discussions) açarak bize ulaşmaktan çekinmeyin.

&nbsp;
## Seçenek 1: `pip` kullanmak (yerleşik, her yerde çalışır)

Zaten güncel bir Python sürümü kullanıyorsanız, paketleri yerleşik `pip` yükleyicisiyle kurabilirsiniz.

Bu kitap için Python 3.12 kullandım. Ancak Python 3.13 ve 3.14 gibi daha yeni sürümler ile 3.11 ve 3.10 gibi daha eski sürümler de PyTorch tarafından desteklendikleri sürece sorunsuz çalışacaktır. Python sürümünüzü şu komutla kontrol edebilirsiniz:

```bash
python --version
```

Python 3.9 veya daha eskisini kullanıyorsanız, [python.org](https://www.python.org/downloads/) adresinden en güncel sürümü kurmayı ya da sürümleri yönetmek için [`pyenv`](https://github.com/pyenv/pyenv) gibi bir araç kullanmayı değerlendirin. Ancak yeni bir Python sürümü kuruyorsanız, lütfen [resmî PyTorch web sitesindeki](https://pytorch.org/get-started/locally/) öneriyi kontrol ederek bu sürümün PyTorch tarafından desteklendiğinden emin olun. PyTorch genellikle en güncel Python sürümünün birkaç ay gerisinden gelir; bu nedenle yeni çıkan Python sürümleri hemen desteklenmez veya önerilmez.

Gerektikçe yeni paketler kurmak için (örneğin PyTorch ve Jupyter Lab) şunu çalıştırın:

```bash
pip install torch jupyterlab
```

Alternatif olarak, bu kitapta kullanılan gerekli tüm Python paketlerini [`requirements.txt`](https://github.com/rasbt/reasoning-from-scratch/blob/main/requirements.txt) dosyası aracılığıyla tek seferde kurabilirsiniz:

```bash
pip install -r https://raw.githubusercontent.com/rasbt/reasoning-from-scratch/refs/heads/main/requirements.txt
```


&nbsp;
## Seçenek 2: `uv` kullanmak (daha hızlı ve yaygın olarak önerilen)

`pip`, Python paketlerini kurmanın klasik ve resmî yolu olmaya devam etse de, [`uv`](https://github.com/astral-sh/uv) modern ve yaygın olarak önerilen bir Python paket yöneticisidir ve otomatik olarak şunları yapar:

- Bir sanal ortam oluşturur ve yönetir
- Paketleri hızlıca kurar
- Yeniden üretilebilir kurulumlar için bir kilit dosyası (lockfile) tutar
- `pip` benzeri komutları destekler

&nbsp;
### `uv` ve Python paketlerini kurmak

`uv` kurmak için aşağıdaki komutları kullanabilirsiniz (en güncel öneriler için ayrıca resmî [Kurulum](https://docs.astral.sh/uv/getting-started/installation/) sayfasına bakın).

&nbsp;
**macOS / Linux:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

&nbsp;
**Windows (PowerShell):**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Kurulduktan sonra, önceki bölümde `pip` ile anlatıldığı gibi yeni Python paketleri kurabilirsiniz; tek fark `pip` yerine `uv pip` yazmanızdır. Örneğin:

```bash
uv pip install torch jupyterlab
```

Ancak `uv` kullanıyorsanız (ki bunu öneriyorum ve kendim de kullanıyorum), `uv pip` yerine aşağıda açıklandığı gibi yerel `uv` söz dizimini kullanmak daha da iyidir.

&nbsp;
### Önerilen `uv` iş akışı

`uv pip` kullanmak yerine, yerel `uv` iş akışını öneriyorum ve kendim de kullanıyorum.

Önce GitHub deposunu yerel makinenize klonlayın:



```bash
git clone https://github.com/rasbt/reasoning-from-scratch.git
```

Ardından bu klasöre girin, örneğin Linux ve macOS'ta:

```bash
cd reasoning-from-scratch
```

Bu klasör bir `pyproject.toml` ve bir `.python-version` dosyası içerdiğinden, artık hazırsınız: `uv`, bir betiği ilk kez çalıştırdığınızda veya Jupyter Lab'ı açtığınızda bu `reasoning-from-scratch` projesi için (varsayılan olarak görünmeyen) bir sanal ortam klasörünü (`.venv`) otomatik olarak oluşturur ve tüm bağımlılıkları oraya kurar.

`.python-version` dosyası şu anda yerel `uv` ortamı için Python 3.13 sürümünü sabitler. Bu, yanlışlıkla bu projeyle test edilen PyTorch sürümlerinden daha yeni bir Python sürümü seçilmesini önler. `uv` farklı bir Python sürümü kullanıyorsa, yerel sabitlemeyi şu komutla sıfırlayabilirsiniz:

```bash
uv python pin 3.13
uv sync
```

Muhtemelen ihtiyacınız olmayacak, ancak genel olarak `pyproject.toml` içinde listelenen gereksinimlerin parçası olmayan ek paketleri `uv add` ile kurabilirsiniz:


```bash
uv add llms_from_scratch
```

Yukarıdaki komut, paketi sanal ortama ve `pyproject.toml` dosyasına ekleyecektir.

&nbsp;
### `uv` ile kod çalıştırmak

Bu bölüm, Jupyter Lab ve Python betiklerini çalıştırmak için `uv` komutlarını açıklar.

Jupyter Lab'ı açmak için şunu çalıştırın:

```bash
uv run jupyter lab
```

Python betikleri şu şekilde çalıştırılabilir:

```bash
uv run python script.py
```




> **İleri düzey kullanım:** Bu bölüm, `uv` aracını `pip` kullanıcılarına tanıdık gelecek basit bir şekilde kullanmayı anlatır. Daha ileri düzey kullanımla ilgileniyorsanız, `uv` içinde sanal ortam yönetimine dair daha ayrıntılı talimatlar için lütfen [bu belgeye](https://github.com/rasbt/LLMs-from-scratch/tree/main/setup/01_optional-python-setup-preferences) bakın.
> macOS veya Linux kullanıcısıysanız ve yerel uv komutlarını tercih ediyorsanız, lütfen [bu öğreticiye](https://github.com/rasbt/LLMs-from-scratch/blob/main/setup/01_optional-python-setup-preferences/native-uv.md) bakın. Ek bilgi için [resmî uv belgelerini](https://docs.astral.sh/uv/) incelemenizi de öneririm.



&nbsp;
### JupyterLab ipuçları

Not defteri kodunu VSCode yerine JupyterLab'da görüntülüyorsanız, JupyterLab'ın (varsayılan ayarıyla) son sürümlerde kaydırma (scrolling) hataları yaşadığını unutmayın. Önerim, Settings -> Settings Editor yolunu izleyip "Windowing mode" ayarını (aşağıda gösterildiği gibi) "none" olarak değiştirmenizdir; bu, sorunu gidermektedir.


![Jupyter Glitch 1](https://sebastianraschka.com/images/reasoning-from-scratch-images/bonus/setup/jupyter_glitching_1.webp)

<br>

![Jupyter Glitch 2](https://sebastianraschka.com/images/reasoning-from-scratch-images/bonus/setup/jupyter_glitching_2.webp)

&nbsp;
## Sorularınız mı var?

Herhangi bir sorunuz varsa, lütfen bu GitHub deposundaki [Discussions](https://github.com/rasbt/reasoning-from-scratch/discussions) forumu üzerinden bize ulaşmaktan çekinmeyin.
