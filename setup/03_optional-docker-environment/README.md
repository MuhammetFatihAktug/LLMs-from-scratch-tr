# Docker Ortamı Kurulum Rehberi

> 🇹🇷 **Türkçe çeviri.** Orijinal İngilizce sürüm: [README.md](https://github.com/rasbt/LLMs-from-scratch/blob/main/setup/03_optional-docker-environment/README.md) · Komut ve yapılandırma blokları birebir korunmuştur.

Bir projenin bağımlılıklarını ve yapılandırmalarını yalıtan bir geliştirme kurulumunu tercih ediyorsanız, Docker kullanmak son derece etkili bir çözümdür. Bu yaklaşım, yazılım paketlerini ve kütüphaneleri elle kurma ihtiyacını ortadan kaldırır ve tutarlı bir geliştirme ortamı sağlar.

Bu rehber, [../01_optional-python-setup-preferences](../01_optional-python-setup-preferences) ve [../02_installing-python-libraries](../02_installing-python-libraries) bölümlerinde açıklanan conda yaklaşımı yerine Docker'ı tercih ediyorsanız, bu kitap için isteğe bağlı bir Docker ortamı kurma sürecinde size rehberlik edecektir.

<br>

## Docker'ı indirmek ve kurmak

Docker'a başlamanın en kolay yolu, ilgili platformunuz için [Docker Desktop](https://docs.docker.com/desktop/) uygulamasını kurmaktır.

Linux (Ubuntu) kullanıcıları bunun yerine [Docker Engine](https://docs.docker.com/engine/install/ubuntu/) kurmayı ve [kurulum sonrası](https://docs.docker.com/engine/install/linux-postinstall/) adımları izlemeyi tercih edebilir.

<br>

## Visual Studio Code'da Docker DevContainer Kullanmak

Docker DevContainer (Geliştirme Konteyneri), geliştiricilerin Docker konteynerlerini tam teşekküllü bir geliştirme ortamı olarak kullanmasına olanak tanıyan bir araçtır. Bu yaklaşım, kullanıcıların yerel makine kurulumlarından bağımsız olarak tutarlı bir geliştirme ortamıyla hızlıca çalışmaya başlamasını sağlar.

DevContainer'lar başka IDE'lerle de çalışsa da, DevContainer'larla çalışmak için yaygın olarak kullanılan bir IDE/editör Visual Studio Code'dur (VS Code). Aşağıdaki rehber, bu kitabın DevContainer'ının VS Code bağlamında nasıl kullanılacağını açıklar; benzer bir süreç PyCharm için de geçerli olmalıdır. Kurulu değilse ve kullanmak istiyorsanız [buradan kurun](https://code.visualstudio.com/download).

1. Bu GitHub deposunu klonlayın ve `cd` ile proje kök dizinine girin.

```bash
git clone https://github.com/rasbt/LLMs-from-scratch.git
cd LLMs-from-scratch
```

2. `.devcontainer` klasörünü `setup/03_optional-docker-environment/` konumundan mevcut dizine (proje köküne) taşıyın.

```bash
mv setup/03_optional-docker-environment/.devcontainer ./
```

3. Docker Desktop'ta **_desktop-linux_ builder** bileşeninin çalıştığından ve Docker konteynerini derlemek için kullanılacağından emin olun (bkz. _Docker Desktop_ -> _Change settings_ -> _Builders_ -> _desktop-linux_ -> _..._ -> _Use_)

4. [CUDA destekli bir GPU'nuz](https://developer.nvidia.com/cuda-gpus) varsa, eğitimi ve çıkarımı hızlandırabilirsiniz:

    4.1 **NVIDIA Container Toolkit** aracını [burada](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installing-with-apt) açıklandığı gibi kurun. NVIDIA Container Toolkit desteği [burada](https://docs.nvidia.com/cuda/wsl-user-guide/index.html#nvidia-compute-software-support-on-wsl-2) anlatıldığı şekilde sağlanmaktadır.

    4.2 Docker Engine daemon yapılandırmasına çalışma zamanı (runtime) olarak _nvidia_ ekleyin (bkz. _Docker Desktop_ -> _Change settings_ -> _Docker Engine_). Yapılandırmanıza şu satırları ekleyin:

    ```json
    "runtimes": {
        "nvidia": {
        "path": "nvidia-container-runtime",
        "runtimeArgs": []
    ```

    Örneğin, Docker Engine daemon yapılandırmasının tam json kodu şöyle görünmelidir:

    ```json
    {
      "builder": {
        "gc": {
          "defaultKeepStorage": "20GB",
          "enabled": true
        }
      },
      "experimental": false,
      "runtimes": {
        "nvidia": {
          "path": "nvidia-container-runtime",
          "runtimeArgs": []
        }
      }
    }
    ```

    ve Docker Desktop'ı yeniden başlatın.

5. Projeyi VS Code'da açmak için terminale `code .` yazın. Alternatif olarak, VS Code'u başlatıp arayüzden açılacak projeyi seçebilirsiniz.

6. Sol taraftaki VS Code _Extensions_ menüsünden **Remote Development** uzantısını kurun.

7. DevContainer'ı açın.

`.devcontainer` klasörü ana `LLMs-from-scratch` dizininde bulunduğundan (ayarlarınıza bağlı olarak `.` ile başlayan klasörler işletim sisteminizde görünmeyebilir), VS Code bunu otomatik olarak algılamalı ve projeyi bir devcontainer içinde açmak isteyip istemediğinizi sormalıdır. Sormazsa, komut paletini açmak için `Ctrl + Shift + P` tuşlarına basıp `dev containers` yazmaya başlayarak tüm DevContainer'a özgü seçeneklerin listesini görebilirsiniz.


&nbsp;
> ⚠️ **root olarak çalıştırma hakkında not**
>
> Varsayılan olarak DevContainer *root kullanıcısı* olarak çalışır. Bu, güvenlik nedeniyle genel olarak önerilmez; ancak bu kitabın kurulumunda basitlik adına, gerekli tüm paketlerin konteyner içinde sorunsuz kurulabilmesi için root yapılandırması kullanılmıştır.
>
> Konteyner içinde Jupyter Lab'ı elle başlatmayı denerseniz şu hatayı görebilirsiniz:
>
>   ```bash
>   Running as root is not recommended. Use --allow-root to bypass.
>   ```
>
>   Bu durumda şunu çalıştırabilirsiniz:
>
>   ```bash
>   uv run jupyter lab --allow-root
>   ```
>
> - VS Code'u Jupyter uzantısıyla kullanırken genellikle Jupyter Lab'ı elle başlatmanız gerekmez. Not defterlerini uzantı üzerinden açmak doğrudan çalışmalıdır.
> - Daha sıkı güvenlik isteyen ileri düzey kullanıcılar, root olmayan bir kullanıcı ayarlamak için `.devcontainer.json` dosyasını değiştirebilir; ancak bu ek yapılandırma gerektirir ve çoğu kullanım senaryosu için gerekli değildir.



8. **Reopen in Container** seçeneğini seçin.

Docker artık, daha önce derlenmemişse `.devcontainer` yapılandırmasında belirtilen Docker imajını derlemeye başlayacak ya da bir kayıt defterinde (registry) mevcutsa imajı çekecektir.

Tüm süreç otomatiktir ve sisteminize ve internet hızınıza bağlı olarak birkaç dakika sürebilir. İsteğe bağlı olarak, mevcut derleme ilerlemesini görmek için VS Code'un sağ alt köşesindeki "Starting Dev Container (show log)" bağlantısına tıklayabilirsiniz.

Tamamlandığında VS Code otomatik olarak konteynere bağlanacak ve projeyi yeni oluşturulan Docker geliştirme ortamında yeniden açacaktır. Kodu yerel makinenizde çalışıyormuş gibi yazabilir, çalıştırabilir ve hata ayıklayabilirsiniz; üstelik Docker'ın yalıtım ve tutarlılık avantajlarıyla birlikte.

&nbsp;
> **Uyarı:**
> Derleme sürecinde bir hatayla karşılaşıyorsanız, bunun nedeni büyük olasılıkla makinenizde uyumlu bir GPU olmadığı için NVIDIA container toolkit desteğinin bulunmamasıdır. Bu durumda `devcontainer.json` dosyasını düzenleyip `"runArgs": ["--runtime=nvidia", "--gpus=all"],` satırını kaldırın ve "Reopen Dev Container" işlemini tekrar çalıştırın.

9. Tamamlandı.

İmaj çekilip derlendikten sonra, projeniz tüm paketler kurulu hâlde konteynerin içine bağlanmış (mount) ve geliştirmeye hazır olmalıdır.

<br>

## Docker İmajını Kaldırmak

Aşağıda, artık kullanmayı planlamıyorsanız bir Docker konteynerini ve imajını kaldırma talimatları yer alıyor. Bu işlem Docker'ın kendisini sisteminizden kaldırmaz; yalnızca projeye özgü Docker bileşenlerini temizler.

1. DevContainer'ınızla ilişkili olanı bulmak için tüm Docker imajlarını listeleyin:

```bash
docker image ls
```

2. Docker imajını, imaj kimliğini veya adını kullanarak kaldırın:

```bash
docker image rm [IMAGE_ID_OR_NAME]
```

<br>

## Docker'ı Kaldırmak

Docker'ın size uygun olmadığına karar verir ve kaldırmak isterseniz, işletim sisteminize özgü adımları özetleyen resmî belgelere [buradan](https://docs.docker.com/desktop/uninstall/) bakın.
