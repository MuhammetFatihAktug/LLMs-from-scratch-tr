# Windows'ta `torch.compile()` Kullanmak

> 🇹🇷 **Türkçe çeviri.** Orijinal İngilizce sürüm: [README.md](https://github.com/rasbt/reasoning-from-scratch/blob/main/ch02/04_torch-compile-windows/README.md) · Komutlar birebir korunmuştur.

`torch.compile()`, çekirdekleri JIT ile derleyen ve çalışan bir C/C++ derleyici araç zinciri gerektiren *TorchInductor* üzerine kuruludur.

Bu nedenle Windows'ta `torch.compile` işlevini çalıştırmak için gereken kurulum, PyTorch kurmak dışında genellikle ek adım gerektirmeyen Linux veya macOS'a kıyasla biraz daha karmaşık olabilir.

Windows kullanıcısıysanız ve `torch.compile` kullanmak size fazla zahmetli veya karmaşık geliyorsa endişelenmeyin; bu depodaki tüm kod örnekleri derleme olmadan da sorunsuz çalışır.

Aşağıda, [Daniel Kleine](https://github.com/d-kleine) tarafından yapılan önerilere ve şu [PyTorch rehberine](https://docs.pytorch.org/tutorials/unstable/inductor_windows.html) dayanarak derlediğim bazı ipuçları yer alıyor.

&nbsp;
## 1 Temel Kurulum (CPU veya CUDA)

&nbsp;
### 1.1 Visual Studio 2022'yi kurun

- **"Desktop development with C++"** iş yükünü (workload) seçin.
- **İngilizce dil paketini** eklemeyi unutmayın (bu olmadan UTF-8 kodlama hatalarıyla karşılaşabilirsiniz).

&nbsp;
### 1.2 Doğru komut istemini açın


Python'u şuradan başlatın:

**"x64 Native Tools Command Prompt for VS 2022"**

veya şuradan:

**"Visual Studio 2022 Developer Command Prompt"**.

Alternatif olarak, ortamı şu komutu çalıştırarak elle başlatabilirsiniz:

```bash
"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
```

&nbsp;
### 1.3 Derleyicinin çalıştığını doğrulayın

Şunu çalıştırın:

   ```bash
   cl.exe
   ```

Sürüm bilgisinin yazdırıldığını görüyorsanız derleyici hazırdır.

&nbsp;
## 2 Yaygın Hataların Giderilmesi

&nbsp;
### 2.1 Hata: `cl not found`

"C++ build tools" iş yüküyle birlikte **Visual Studio Build Tools** kurun ve Python'u bir geliştirici komut isteminden çalıştırın. (Ayrıntılar için bu Microsoft [rehberine](https://learn.microsoft.com/en-us/cpp/build/vscpp-step-0-installation?view=msvc-170) bakın.)

&nbsp;
### 2.2 Hata: `triton not found` (CUDA kullanırken)

Triton'un Windows sürümünü elle kurun:

```bash
pip install "triton-windows<3.4"
```

veya `uv` kullanıyorsanız:

```bash
uv pip install "triton-windows<3.4"
```

(Daha önce belirtildiği gibi, triton CUDA çekirdeği derlemesi için TorchInductor tarafından gereklidir.)



&nbsp;
## 3 Ek Notlar

Windows'ta `cl.exe` derleyicisine yalnızca Visual Studio Developer ortamı içinden erişilebilir. Bu, Jupyter gibi not defterlerinde `torch.compile()` kullanmanın, not defteri bir Developer Command Prompt üzerinden başlatılmadıysa çalışmayabileceği anlamına gelir.

Bu yazının başında belirtildiği gibi, bazı kullanıcıların `torch.compile()` işlevini Windows CPU derlemelerinde çalıştırırken faydalı bulduğu bir [PyTorch rehberi](https://docs.pytorch.org/tutorials/unstable/inductor_windows.html) de var. Ancak bunun PyTorch'un kararsız (unstable) dalına atıfta bulunduğunu unutmayın; yalnızca referans olarak kullanın.

**Derleme sorun çıkarmaya devam ederse, lütfen çekinmeden atlayın. Güzel bir bonustur, ancak kitabı takip etmek için önemli değildir.**
