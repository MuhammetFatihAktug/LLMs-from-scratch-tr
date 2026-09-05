# Sohbet Arayüzüyle Sıfırdan Qwen3

> 🇹🇷 **Türkçe çeviri.** Orijinal İngilizce sürüm: [README.md](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch05/11_qwen3/qwen3-chat-interface/README.md) · Komutlar birebir korunmuştur.



Bu bonus klasörü, önceden eğitilmiş Qwen3 modeliyle etkileşim kurmak için ChatGPT benzeri bir kullanıcı arayüzü çalıştıran kodu içerir.



![Chainlit UI example](https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/qwen/qwen3-chainlit.gif)



Bu kullanıcı arayüzünü uygulamak için açık kaynaklı [Chainlit Python paketini](https://github.com/Chainlit/chainlit) kullanıyoruz.

&nbsp;
## Adım 1: Bağımlılıkları kurun

Önce `chainlit` paketini ve [requirements-extra.txt](requirements-extra.txt) listesindeki bağımlılıkları şu komutla kuruyoruz:

```bash
pip install -r requirements-extra.txt
```

Veya `uv` kullanıyorsanız:

```bash
uv pip install -r requirements-extra.txt
```



&nbsp;

## Adım 2: `app` kodunu çalıştırın

Bu klasör 2 dosya içerir:

1. [`qwen3-chat-interface.py`](qwen3-chat-interface.py): Bu dosya Qwen3 0.6B modelini düşünme (thinking) modunda yükler ve kullanır.
2. [`qwen3-chat-interface-multiturn.py`](qwen3-chat-interface-multiturn.py): Yukarıdakinin aynısı, ancak mesaj geçmişini hatırlayacak şekilde yapılandırılmış.

(Daha fazlasını öğrenmek için bu dosyaları açıp inceleyin.)

Arayüz sunucusunu başlatmak için terminalde aşağıdaki komutlardan birini çalıştırın:

```bash
chainlit run qwen3-chat-interface.py
```

veya `uv` kullanıyorsanız:

```bash
uv run chainlit run qwen3-chat-interface.py
```

Yukarıdaki komutlardan birini çalıştırmak, modelle etkileşim kurabileceğiniz yeni bir tarayıcı sekmesi açmalıdır. Tarayıcı sekmesi otomatik olarak açılmazsa, terminal çıktısını inceleyip yerel adresi tarayıcınızın adres çubuğuna kopyalayın (adres genellikle `http://localhost:8000` şeklindedir).
