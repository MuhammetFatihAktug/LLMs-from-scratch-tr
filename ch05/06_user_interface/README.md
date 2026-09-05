# Önceden Eğitilmiş LLM ile Etkileşim İçin Kullanıcı Arayüzü Oluşturmak

> 🇹🇷 **Türkçe çeviri.** Orijinal İngilizce sürüm: [README.md](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch05/06_user_interface/README.md) · Komutlar birebir korunmuştur.



Bu bonus klasörü, aşağıda gösterildiği gibi 5. bölümdeki önceden eğitilmiş LLM'lerle etkileşim kurmak için ChatGPT benzeri bir kullanıcı arayüzü çalıştıran kodu içerir.



![Chainlit UI example](https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/chainlit/chainlit-orig.webp)



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

Bu klasör 2 dosya içerir:

1. [`app_orig.py`](app_orig.py): Bu dosya OpenAI'ın orijinal GPT-2 ağırlıklarını yükler ve kullanır.
2. [`app_own.py`](app_own.py): Bu dosya 5. bölümde ürettiğimiz GPT-2 ağırlıklarını yükler ve kullanır. Bunun için önce [`../01_main-chapter-code/ch05.ipynb`](../01_main-chapter-code/ch05.ipynb) dosyasını çalıştırmanız gerekir.

(Daha fazlasını öğrenmek için bu dosyaları açıp inceleyin.)

Arayüz sunucusunu başlatmak için terminalde aşağıdaki komutlardan birini çalıştırın:

```bash
chainlit run app_orig.py
```

veya

```bash
chainlit run app_own.py
```

Yukarıdaki komutlardan birini çalıştırmak, modelle etkileşim kurabileceğiniz yeni bir tarayıcı sekmesi açmalıdır. Tarayıcı sekmesi otomatik olarak açılmazsa, terminal çıktısını inceleyip yerel adresi tarayıcınızın adres çubuğuna kopyalayın (adres genellikle `http://localhost:8000` şeklindedir).
