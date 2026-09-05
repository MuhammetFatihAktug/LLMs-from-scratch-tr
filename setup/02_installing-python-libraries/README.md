# Bu Kitapta Kullanılan Python Paketlerini ve Kütüphanelerini Kurmak

> 🇹🇷 **Türkçe çeviri.** Orijinal İngilizce sürüm: [README.md](https://github.com/rasbt/LLMs-from-scratch/blob/main/setup/02_installing-python-libraries/README.md) · Komutlar birebir korunmuştur.

Bu belge, kurulu Python sürümünüzü ve paketlerinizi bir kez daha doğrulamanız için daha fazla bilgi sunar. (Python ve Python paketlerini kurma hakkında daha fazla bilgi için lütfen [../01_optional-python-setup-preferences](../01_optional-python-setup-preferences) klasörüne bakın.)

Bu kitap için [buradaki](https://github.com/rasbt/LLMs-from-scratch/blob/main/requirements.txt) listede yer alan kütüphaneleri kullandım. Bu kütüphanelerin daha yeni sürümleri de büyük olasılıkla uyumludur. Ancak kodla ilgili herhangi bir sorun yaşarsanız, geri dönüş seçeneği olarak bu kütüphane sürümlerini deneyebilirsiniz.



> **Not:**
> [Seçenek 1: uv Kullanmak](../01_optional-python-setup-preferences/README.md) bölümünde açıklandığı gibi `uv` kullanıyorsanız, aşağıdaki komutlarda `pip` yerine `uv pip` yazabilirsiniz. Örneğin, `pip install -r requirements.txt` komutu `uv pip install -r requirements.txt` hâline gelir



Bu gereksinimleri en kolay şekilde kurmak için bu kod deposunun kök dizinindeki `requirements.txt` dosyasını kullanabilir ve şu komutu çalıştırabilirsiniz:

```bash
pip install -r requirements.txt
```

Alternatif olarak, GitHub URL'si üzerinden şu şekilde kurabilirsiniz:

```bash
pip install -r https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/requirements.txt
```


Ardından, kurulum tamamlandıktan sonra lütfen tüm paketlerin kurulu ve güncel olup olmadığını şu komutla kontrol edin:

```bash
python python_environment_check.py
```

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/02_installing-python-libraries/check_1.jpg" width="600px">

Ayrıca bu dizindeki `python_environment_check.ipynb` dosyasını çalıştırarak sürümleri JupyterLab'da kontrol etmeniz de önerilir; bu, ideal olarak yukarıdakiyle aynı sonuçları vermelidir.

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/02_installing-python-libraries/check_2.jpg" width="500px">

Aşağıdaki sorunları görüyorsanız, JupyterLab örneğiniz büyük olasılıkla yanlış conda ortamına bağlıdır:

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/02_installing-python-libraries/jupyter-issues.jpg" width="450px">

Bu durumda, JupyterLab örneğini doğru conda ortamında açıp açmadığınızı `--conda` bayrağıyla `watermark` kullanarak kontrol etmek isteyebilirsiniz:

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/02_installing-python-libraries/watermark.jpg" width="350px">


&nbsp;
## PyTorch'u Kurmak

PyTorch, diğer Python kütüphaneleri veya paketleri gibi pip ile kurulabilir. Örneğin:

```bash
pip install torch
```

Ancak PyTorch, CPU ve GPU uyumlu kodlar içeren kapsamlı bir kütüphane olduğundan kurulumu ek ayarlar ve açıklama gerektirebilir (daha fazla bilgi için kitaptaki *A.1.3 Installing PyTorch* kısmına bakın).

Ayrıca [https://pytorch.org](https://pytorch.org) adresindeki resmî PyTorch web sitesinde yer alan kurulum kılavuzu menüsüne başvurmanız şiddetle önerilir.

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/02_installing-python-libraries/pytorch-installer.jpg" width="600px">

<br>



&nbsp;
## JupyterLab ipuçları

Not defteri kodunu VSCode yerine JupyterLab'da görüntülüyorsanız, JupyterLab'ın (varsayılan ayarıyla) son sürümlerde kaydırma (scrolling) hataları yaşadığını unutmayın. Önerim, Settings -> Settings Editor yolunu izleyip "Windowing mode" ayarını (aşağıda gösterildiği gibi) "none" olarak değiştirmenizdir; bu, sorunu gidermektedir.


![Jupyter Glitch 1](https://sebastianraschka.com/images/reasoning-from-scratch-images/bonus/setup/jupyter_glitching_1.webp)

<br>

![Jupyter Glitch 2](https://sebastianraschka.com/images/reasoning-from-scratch-images/bonus/setup/jupyter_glitching_2.webp)

<br>

---




Sorularınız mı var? Lütfen [Tartışma Forumu](https://github.com/rasbt/LLMs-from-scratch/discussions) üzerinden bize ulaşmaktan çekinmeyin.
