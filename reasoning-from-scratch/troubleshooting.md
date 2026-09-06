# Sorun Giderme Rehberi

> 🇹🇷 **Türkçe çeviri.** Orijinal İngilizce sürüm: [troubleshooting.md](https://github.com/rasbt/reasoning-from-scratch/blob/main/troubleshooting.md) · Kod, hata mesajı ve komut blokları birebir korunmuştur.

Bu sayfa, kitap boyunca ilerlerken karşılaşılan yaygın sorunları ve kurulum ipuçlarını bir araya toplar.

&nbsp;
## JupyterLab kaydırma (scrolling) hatası

Not defteri kodunu VSCode yerine JupyterLab'da görüntülüyorsanız, JupyterLab'ın (varsayılan ayarıyla) son sürümlerde kaydırma hataları yaşadığını unutmayın. Önerim, Settings -> Settings Editor yolunu izleyip "Windowing mode" ayarını (aşağıda gösterildiği gibi) "none" olarak değiştirmenizdir; bu, sorunu gidermektedir.


![Jupyter Glitch 1](https://sebastianraschka.com/images/reasoning-from-scratch-images/bonus/setup/jupyter_glitching_1.webp)

<br>

![Jupyter Glitch 2](https://sebastianraschka.com/images/reasoning-from-scratch-images/bonus/setup/jupyter_glitching_2.webp)


&nbsp;
## Bölüm 2

&nbsp;
### Dosya İndirme Sorunları

Dosya indirmeleriyle ilgili herhangi bir sorununuz varsa lütfen [bu tartışma sayfasını](https://github.com/rasbt/reasoning-from-scratch/discussions/145) kullanın.

Kod, aşağıdaki Hugging Face konumlarından indirme yapar; makinenizin veya ağınızın bunları engelleyip engellemediğini kontrol etmek için bu adresleri tarayıcıda elle de açabilirsiniz:

- Bölüm 2 model ve tokenizer dosyaları: [rasbt/qwen3-from-scratch](https://huggingface.co/rasbt/qwen3-from-scratch/tree/main)
- Temel model dosyası: [qwen3-0.6B-base.pth](https://huggingface.co/rasbt/qwen3-from-scratch/resolve/main/qwen3-0.6B-base.pth)
- Temel tokenizer dosyası: [tokenizer-base.json](https://huggingface.co/rasbt/qwen3-from-scratch/resolve/main/tokenizer-base.json)
- Akıl yürütme model dosyası: [qwen3-0.6B-reasoning.pth](https://huggingface.co/rasbt/qwen3-from-scratch/resolve/main/qwen3-0.6B-reasoning.pth)
- Akıl yürütme tokenizer dosyası: [tokenizer-reasoning.json](https://huggingface.co/rasbt/qwen3-from-scratch/resolve/main/tokenizer-reasoning.json)
- Bölüm 7 GRPO kontrol noktaları: [rasbt/qwen3-from-scratch-grpo-checkpoints](https://huggingface.co/rasbt/qwen3-from-scratch-grpo-checkpoints/tree/main)
- Bölüm 8 damıtma kontrol noktaları: [rasbt/qwen3-from-scratch-distill-checkpoints](https://huggingface.co/rasbt/qwen3-from-scratch-distill-checkpoints/tree/main)

&nbsp;
#### SSL / proxy / sertifika hataları

Bir model indirmesi `SSL`, `CERTIFICATE_VERIFY_FAILED` veya `ProxyError` ifadelerini içeren hatalarla başarısız olursa, sorun genellikle eksik bir dosyadan değil ortamdan kaynaklanır.

Bu genel olarak yaygın değildir, ancak bir VPN, proxy, güvenlik duvarı veya antivirüs ürününün HTTPS trafiğini araya girerek incelediği iş veya okul makinelerinde olabilir. Bu durumda şunları deneyin:

- Yukarıda listelenen ilgili Hugging Face URL'sinin tarayıcınızda açılıp açılmadığını kontrol edin.
- Tokenizer iniyor ancak `.pth` model dosyası inmiyorsa, proxy daha büyük dosyaları veya `.pth` uzantısını engelliyor olabilir.
- BT ekibinizden indirmeye izin vermesini veya proxy sertifikasının Python tarafından güvenilir sayılmasını sağlamasını isteyin.
- Bazı yönetilen makinelerde okurlar, Python'un işletim sistemi sertifika deposunu kullanmasını sağlayan `pip install pip-system-certs` ile başarı elde ettiklerini bildirdi.

&nbsp;
### `InductorError: CppCompileError`
Linux kullanıcısıysanız ve `torch.compile` çalıştırırken aşağıdaki satırları içeren bir `InductorError: CppCompileError: C++ compile error` hatası görüyorsanız:

```python
Python.h: No such file or directory
81 | #include <Python.h>
| ^~~~~~~~~~
compilation terminated.
```

bu, Python çalışma zamanınızda modeli CPU kullanımı için derlemek üzere gereken bazı C++ başlık dosyalarının eksik olabileceğine işaret eder.

Örneğin dosyanın var olup olmadığını kontrol edebilirsiniz: `ls -l /usr/include/python3.12/Python.h`.

Yoksa, farklı bir Python çalışma zamanı kurmayı deneyebilirsiniz:

```bash
sudo apt-get install -y python3.12-dev build-essential
```

Veya `torch.compile` çağrısından önce PyTorch'taki C++ gereksinimlerini devre dışı bırakabilirsiniz:

```python
import torch
import torch._inductor.config as inductor_config

inductor_config.cpp_wrapper = False

compiled_model = torch.compile(model)
```

Daha fazla bağlam için ayrıca bkz. [#192](https://github.com/rasbt/reasoning-from-scratch/issues/192).


&nbsp;
### Windows CPU: `algorithm` veya `omp.h` ile `fatal error C1083`

Windows kullanıyorsanız ve `torch.compile()` şu hatayla başarısız oluyorsa:

```text
fatal error C1083: Cannot open include file: 'algorithm': No such file or directory
```

veya

```text
fatal error C1083: Cannot open include file: 'omp.h': No such file or directory
```

sorun genellikle (bu kitaptaki/depodaki koddan değil) TorchInductor tarafından kullanılan yerel Windows derleyici / OpenMP kurulumundan kaynaklanır.

Bir okur, yalnızca CPU'lu bir Intel sisteminde [forumda](https://livebook.manning.com/forum?product=raschka2&comment=583365) şu ipuçlarını paylaştı:

- PyTorch'u yükseltmek eksik `algorithm` başlığı sorununu çözdü.
- Ancak eksik `omp.h` başlığı sorunu devam etti.
- `"eager"` veya `"aot_eager"` gibi bir yedek (fallback) arka uç kullanmak kodun çalışmasını sağladı.

Örneğin:

```python
compiled_model = torch.compile(model, backend="eager")

# or

compiled_model = torch.compile(model, backend="aot_eager")
```

Bunun tam bir çözüm değil, geçici bir çözüm (workaround) olduğunu unutmayın. Yardımcı olabilir, ancak TorchInductor derleme yolunun tamamını kullanmaz; dolayısıyla hızlanmalar tam çalışan bir `torch.compile()` ile elde edilenden daha küçük olabilir.

**Ancak lütfen şunu da unutmayın: torch.compile bu kitap için zorunlu değildir ve bu bölümü tamamen atlamakta özgürsünüz.**

Yine de çalıştırmaya çalışıyorsanız, hata ayıklamaya çok zaman harcamadan önce asgari bir sağlık kontrolü (sanity check) yapmak faydalı olabilir:

```python
import torch

device = "cpu"  # or "xpu"

def foo(x, y):
    a = torch.sin(x)
    b = torch.cos(y)
    return a + b


opt_foo = torch.compile(foo)
out = opt_foo(torch.randn(10, 10).to(device), torch.randn(10, 10).to(device))
print(out.shape)
```

Bu küçük örnek bile başarısız oluyorsa, sorun büyük olasılıkla bu kitaptaki/depodaki model kodundan değil, PyTorch / derleyici kurulumunuzdan kaynaklanıyordur.

Ek kurulum ipuçları için ayrıca bkz. [Windows'ta `torch.compile()` Kullanmak](ch02/04_torch-compile-windows/README.md) ve PyTorch [Windows CPU/XPU rehberi](https://docs.pytorch.org/tutorials/unstable/inductor_windows.html). Ve daha önce belirtildiği gibi, `torch.compile()` sisteminizde kararsız kalmaya devam ederse, kitap örnekleri için atlamanız sorun değildir.



&nbsp;
## Bölüm 6

&nbsp;
### Bozuk Kontrol Noktaları

`train_rlvr_grpo` içinde (Bölüm 6), `Ctrl+C` tuşları `KeyboardInterrupt` işleyicisini tetikleyerek bir `-interrupt` kontrol noktası kaydeder. Kaydetme tamamlanmadan `Ctrl+C` tuşlarına ikinci kez basarsanız, bu `torch.save` işlemini yazma sırasında kesebilir ve kırpılmış bir `.pth` dosyası bırakabilir. Çıkmadan önce `-interrupt` kontrol noktası mesajını bekleyin.

Bozuk model kontrol noktaları genellikle yükleme hataları verir veya değerlendirme sırasında başarısız olur; bir diğer belirti de beklenen ~1,5 GB boyuttan çok daha küçük olmalarıdır.

&nbsp;
## Diğer Sorunlar

Diğer sorunlar için lütfen GitHub'da yeni bir [Issue](https://github.com/rasbt/reasoning-from-scratch/issues) açmaktan çekinmeyin.
