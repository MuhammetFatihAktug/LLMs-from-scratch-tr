# Ek G: Bir Sohbet Arayüzü Oluşturmak

> 🇹🇷 **Türkçe çeviri.** Orijinal İngilizce sürüm: [README.md](https://github.com/rasbt/reasoning-from-scratch/blob/main/chG/01_main-chapter-code/README.md) · Komutlar birebir korunmuştur.



Bu klasör, aşağıda gösterildiği gibi bu kitapta kullanılan ve/veya geliştirilen LLM'lerle etkileşim kurmak için ChatGPT benzeri bir kullanıcı arayüzü çalıştıran kodu içerir.



![Chainlit UI example](https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/qwen/qwen3-chainlit.gif)



Bu kullanıcı arayüzünü uygulamak için açık kaynaklı [Chainlit Python paketini](https://github.com/Chainlit/chainlit) kullanıyoruz.

&nbsp;
## Adım 1: Bağımlılıkları kurun

Önce `chainlit` paketini ve bağımlılığını kuruyoruz:

```bash
pip install chainlit
```

Veya `uv` kullanıyorsanız:

```bash
uv add chainlit
```



&nbsp;

## Adım 2: `app` kodunu çalıştırın

Bu klasör 2 dosya içerir:

1. [`qwen3_chat_interface.py`](qwen3_chat_interface.py): Bu dosya Qwen3 0.6B modelini düşünme (thinking) modunda yükler ve kullanır.
2. [`qwen3_chat_interface_multiturn.py`](qwen3_chat_interface_multiturn.py): Yukarıdakinin aynısı, ancak mesaj geçmişini hatırlayacak şekilde yapılandırılmış.

(Daha fazlasını öğrenmek için bu dosyaları açıp inceleyin.)

Arayüz sunucusunu başlatmak için terminalde aşağıdaki komutlardan birini çalıştırın:

```bash
chainlit run qwen3_chat_interface.py
```

veya `uv` kullanıyorsanız:

```bash
uv run chainlit run qwen3_chat_interface.py
```

Yukarıdaki komutlardan birini çalıştırmak, modelle etkileşim kurabileceğiniz yeni bir tarayıcı sekmesi açmalıdır. Tarayıcı sekmesi otomatik olarak açılmazsa, terminal çıktısını inceleyip yerel adresi tarayıcınızın adres çubuğuna kopyalayın (adres genellikle `http://localhost:8000` şeklindedir).

## Özel bir kontrol noktası (checkpoint) kullanmak

`chainlit run ...` komut satırı argümanlarını kendisi kullandığı için, bu betikler özel bir kontrol noktası yolunu `argparse` yerine `CHECKPOINT_PATH` ortam değişkeninden okur.

Terminal örneği:

```bash
CHECKPOINT_PATH=/absolute/path/to/qwen3-0.6B-distill-step06682-epoch1.pth \
uv run chainlit run qwen3_chat_interface.py
```

Notlar:

- Betikteki `WHICH_MODEL` değerini, kontrol noktasının beklediği tokenizer ile uyumlu tutun.
- [`ch08/05_download_training_checkpoints`](../../ch08/05_download_training_checkpoints) klasöründeki 8. bölüm kontrol noktaları akıl yürütme (reasoning) tokenizer'ını kullanır; bu nedenle `WHICH_MODEL = "reasoning"` kullanın.
- `CHECKPOINT_PATH` ayarlandığında betik yalnızca tokenizer'ı `LOCAL_DIR` içine indirir; varsayılan model ağırlıklarını yeniden indirmez.
