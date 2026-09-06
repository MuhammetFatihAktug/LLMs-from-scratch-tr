# Testler

> 🇹🇷 **Türkçe çeviri.** Orijinal İngilizce sürüm: [README.md](https://github.com/rasbt/reasoning-from-scratch/blob/main/tests/README.md) · Komutlar birebir korunmuştur.

Bu dizin, deponun Python test paketini içerir.

## Yerel çalıştırmalar

Önce geliştirme (dev) ortamını kurun:

```bash
uv sync --group dev
```

### 1. Pahalı testleri yok sayarak normal paket (önerilen)

Hızlı test ve yeni özellik geliştirme için önerilir.

```bash
SKIP_EXPENSIVE=1 RUN_REAL_DOWNLOAD_TESTS=0 uv run pytest tests
```

Tek bir test dosyasını çalıştırmak için:

```bash
SKIP_EXPENSIVE=1 RUN_REAL_DOWNLOAD_TESTS=0 uv run pytest tests/test_ch03.py
```


Bu, varsayılan GitHub test matrisinin yerel karşılığına en yakın seçenektir.

### 2. Normal paket artı pahalı testler

Çalıştırılması nispeten pahalı olduğu için varsayılan olarak yok sayılan bazı kodlar vardır. Temel hata ayıklamayı bitirdiyseniz bu testleri çalıştırmanızı öneririm.

```bash
SKIP_EXPENSIVE=0 RUN_REAL_DOWNLOAD_TESTS=0 uv run pytest tests
```

Bunun, test dosyalarında `SKIP_EXPENSIVE` ile korunan testleri çalıştırdığını, ancak büyük model kontrol noktaları indiren gerçek ağ/indirme entegrasyon testlerini hâlâ hariç tuttuğunu unutmayın.

### 3. Yalnızca indirme testleri

Model kontrol noktası dosyalarının indirilebilir olup olmadığını ve sunucuların (hâlâ) çalışıp çalışmadığını kontrol eden bazı testler vardır. Bu testleri yerelde veya düzenli olarak çalıştırmak gerekli değildir. Bu daha çok ara sıra yapılacak testler içindir.

Bu indirme testlerini çalıştırmak için şunu kullanın:

```bash
SKIP_EXPENSIVE=0 RUN_REAL_DOWNLOAD_TESTS=1 uv run pytest tests -k real_download
```

Bu nasıl çalışır:

- `pytest tests`, `tests/` dizinindeki testleri toplar
- `-k real_download`, yalnızca adı `real_download` içeren testleri tutar

Daha hedefli bir örnek olarak, örneğin ek D gerçek anlık görüntü (snapshot) testini doğrudan çalıştırmak için şunu kullanın:

```bash
SKIP_EXPENSIVE=0 RUN_REAL_DOWNLOAD_TESTS=1 uv run pytest tests/test_appendix_d.py -k real_download_1_7b
```

İsteğe bağlı (opt-in) gerçek indirme testleri şu anda şunları kapsar:

- `tests/test_ch03.py`: gerçek `math500_test.json` indirmesi ve tokenizer indirmeleri
- `tests/test_ch06.py`: gerçek matematik eğitim kümesi indirmesi
- `tests/test_ch07.py`: gerçek GitHub ham (raw) dosya indirmesi
- `tests/test_ch08.py`: gerçek damıtma veri kümesi ve tokenizer indirmeleri
- `tests/test_appendix_d.py`: gerçek `Qwen/Qwen3-1.7B-Base` anlık görüntü indirmesi
- `tests/test_qwen3.py`: gerçek `Qwen/Qwen3-0.6B` tokenizer karşılaştırması


### 4. Her şey (önerilmez)

Bu, test paketindeki her şeyi çalıştırır. Bunun hem hesaplama açısından pahalı testleri (3. bölüm) hem de pahalı indirme testlerini (4. bölüm) içerdiğini unutmayın.

```bash
SKIP_EXPENSIVE=0 RUN_REAL_DOWNLOAD_TESTS=1 uv run pytest tests
```

Kod değişikliği yaparken rutin testler için bu önerilmez; çünkü dosya indirmeleri çok maliyetlidir ve düzenli olarak çalıştırılması gereksizdir.


## GitHub CI'da neler çalışır

Varsayılan GitHub test matrisi normal paketi çalıştırır ve daha ağır testleri atlar:

- `.github/workflows/tests-linux.yml`
- `.github/workflows/tests-macos.yml`
- `.github/workflows/tests-windows.yml`
- `.github/workflows/basic-tests-pip.yml`

Bu iş akışları `SKIP_EXPENSIVE=1` ayarlar; dolayısıyla pahalı testler orada atlanır. Bunun nedeni, GitHub CI'ın pahalı testleri çalıştırmak için gereken hesaplama kaynaklarına (GPU gibi) sahip olmamasıdır.

Gerçek ağ/indirme entegrasyon testleri ayrı bir iş akışında çalışır:

- `.github/workflows/real-download-tests.yml`

Bu iş akışı `RUN_REAL_DOWNLOAD_TESTS=1` ayarlar ve yalnızca `-k real_download` ile seçilen testleri çalıştırır.
Varsayılan PR/push matrisinin bir parçası değildir. Haftalık bir zamanlamayla çalışır ve ayrıca `workflow_dispatch` ile elle de başlatılabilir.
