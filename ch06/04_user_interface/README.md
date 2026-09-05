# GPT Tabanlı Spam Sınıflandırıcısıyla Etkileşim İçin Kullanıcı Arayüzü Oluşturmak

> 🇹🇷 **Türkçe çeviri.** Orijinal İngilizce sürüm: [README.md](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch06/04_user_interface/README.md) · Komutlar birebir korunmuştur.



Bu bonus klasörü, aşağıda gösterildiği gibi 6. bölümdeki ince ayarlı GPT tabanlı spam sınıflandırıcısıyla etkileşim kurmak için ChatGPT benzeri bir kullanıcı arayüzü çalıştıran kodu içerir.



![Chainlit UI example](https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/chainlit/chainlit-spam.webp)



Bu kullanıcı arayüzünü uygulamak için açık kaynaklı [Chainlit Python paketini](https://github.com/Chainlit/chainlit) kullanıyoruz.

&nbsp;
## Adım 1: Bağımlılıkları kurun

Önce `chainlit` paketini şu komutla kuruyoruz:

```bash
pip install chainlit
```

(Alternatif olarak `pip install -r requirements-extra.txt` komutunu çalıştırın.)

&nbsp;
## Adım 2: `app` kodunu çalıştırın

[`app.py`](app.py) dosyası arayüz kodunu içerir. Daha fazlasını öğrenmek için bu dosyaları açıp inceleyin.

Bu dosya, 6. bölümde ürettiğimiz GPT-2 sınıflandırıcı ağırlıklarını yükler ve kullanır. Bunun için önce [`../01_main-chapter-code/ch06.ipynb`](../01_main-chapter-code/ch06.ipynb) dosyasını çalıştırmanız gerekir.

Arayüz sunucusunu başlatmak için terminalde şu komutu çalıştırın:

```bash
chainlit run app.py
```

Yukarıdaki komutları çalıştırmak, modelle etkileşim kurabileceğiniz yeni bir tarayıcı sekmesi açmalıdır. Tarayıcı sekmesi otomatik olarak açılmazsa, terminal çıktısını inceleyip yerel adresi tarayıcınızın adres çubuğuna kopyalayın (adres genellikle `http://localhost:8000` şeklindedir).
