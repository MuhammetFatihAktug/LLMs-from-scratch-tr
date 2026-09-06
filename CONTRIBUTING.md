# Katkıda Bulunma Rehberi

Bu depo, [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) deposunun **Türkçe çevirisidir**. Katkıya açıktır; ama çeviri deposu olduğu için normal bir yazılım projesinden farklı kuralları vardır. Lütfen bir PR açmadan önce bu sayfayı okuyun — özellikle **Değişmezler** bölümünü.

&nbsp;

## Ne tür katkılar bekleniyor?

| ✅ Bekleniyor | ❌ Bu depoda yeri yok |
|---|---|
| Çeviri hatası düzeltme | Kodun davranışını değiştiren düzeltmeler |
| Anlatım/akıcılık iyileştirmesi | Yeni bölüm, yeni model, yeni özellik |
| Terim tutarsızlığı bildirimi | Upstream'deki hataların düzeltilmesi |
| Türkçe örnek veri katkısı | Kütüphane sürümü yükseltme |
| Bozuk bağlantı bildirimi | Notebook çıktılarını yeniden üretme |

**Kod ya da içerik hatası bulduysanız** — yani çeviriden bağımsız, orijinalde de var olan bir sorun — lütfen [upstream depoya](https://github.com/rasbt/LLMs-from-scratch/issues) bildirin. Düzeltme oraya girdiğinde bu depoya da yansıtırız.

&nbsp;

## Değişmezler (bunlar bozulursa PR birleştirilmez)

Bu depo, upstream'den **yalnızca çeviri kadar** farklı olacak şekilde tasarlandı. Aşağıdaki beş kural bunu güvence altına alır:

**1. Sadece gerçek yorum ve metin değişir.**
Çevrilebilir olanlar: Markdown dosyaları, notebook markdown hücreleri, kod içindeki `#` yorum satırları.

**2. Çalıştırılabilir kod dokunulmazdır.**
Değişken/fonksiyon adları, argümanlar, metin sabitleri (string), model istemleri (prompt), sohbet şablonları — hiçbiri çevrilmez. `"Every effort moves you"` çevrilirse model çıktısı değişir ve kitapla uyuşmaz.

> ⚠️ **Sık yapılan hata:** Bir docstring ya da `"""..."""` bloğu içindeki `#` işareti yorum *değildir*.
> Örnek: `"a large # of unicode characters"` — buradaki `#` cümlenin parçası. Satır bazlı arayıp
> değiştirmeyin; bu hata bu depoda bir kez yaşandı ve bir docstring'i bozdu.

**3. Kayıtlı çıktılar korunur.**
Notebook hücre çıktıları, `execution_count` değerleri, eğitim günlükleri, başarım tabloları, markdown içindeki ``` kod blokları — hiçbiri değişmez. Notebook'u çalıştırdıysanız, PR açmadan önce çıktıları eski hâline döndürün:

```bash
git checkout -- <notebook.ipynb>
```

**4. Lisans ve atıf metinleri dokunulmazdır.**
Telif bildirimleri, `LICENSE.txt`, üçüncü taraf atıfları (OpenAI MIT, LitGPT Apache 2.0), notebook başlarındaki "Supplementary code for..." atıf tablosu ve PEP 723 betik üst verisi (`# /// script`) çevrilmez.

**5. Terim tutarlılığı zorunludur.**
Aynı İngilizce terim depo genelinde aynı Türkçe karşılığı alır. Karşılıklar: [SOZLUK.md](SOZLUK.md). Yeni bir terim öneriyorsanız önce bir issue açın; sözlüğe eklemeden çeviriye sokmayın.

&nbsp;

## Değişikliğinizi nasıl doğrularsınız?

PR açmadan önce şu üçünü çalıştırın:

```bash
# 1) Değişen dosyalar beklediğiniz gibi mi?
git status --short

# 2) Notebook'larda yalnızca metin/yorum mu değişti?
#    (çıktı veya kod değiştiyse burada görünür)
git diff --stat

# 3) Python dosyalarında kod bozulmadı mı?
python -m compileall -q <degistirdiginiz_dosya.py> && echo "sözdizimi tamam"
```

Notebook düzenlediyseniz, JSON'un bozulmadığını da doğrulayın:

```bash
python -c "import json,sys; json.load(open(sys.argv[1])); print('geçerli JSON')" <notebook.ipynb>
```

&nbsp;

## Dal yapısı

| Dal | İçerik | PR hedefi |
|---|---|---|
| `tr` | Türkçe çeviri (varsayılan dal) | ✅ Buraya açın |
| `main` | Dokunulmamış İngilizce upstream | ❌ Asla PR açmayın |

`main` dalı, upstream ile karşılaştırma yapabilmek için bilerek değiştirilmeden tutulur:

```bash
git show main:ch03/README.md        # bir dosyanın İngilizce aslı
git diff main tr -- ch03/README.md  # yan yana fark
```

&nbsp;

## 🔐 Güvenlik: API anahtarınızı commit'lemeyin

Depoda dört tane `config.json` dosyası var (`ch05/07_gpt_to_llama/`, `ch07/02_dataset-utilities/`, `ch07/03_model-evaluation/`, `ch07/05_dataset-generation/`). Bunlar **yer tutucudur** (`"sk-..."`, `"hf-..."`) ve bölümlerin çalışması için kendi anahtarınızı yazmanız beklenir.

**Tehlike:** Bu dosyalar git tarafından izleniyor. `.gitignore` izlenen dosyaları korumaz — anahtarınızı yazıp `git commit -a` yaparsanız anahtar depoya girer ve public bir depoda anında ele geçirilir.

Kendinizi korumak için, anahtarı yazmadan önce:

```bash
git update-index --skip-worktree ch07/02_dataset-utilities/config.json
```

Bu, dosyadaki yerel değişikliklerinizi git'ten gizler. Geri almak için `--no-skip-worktree` kullanın.

Yanlışlıkla bir anahtar commit'lerseniz: **önce anahtarı sağlayıcıdan iptal edin**, sonra geçmişi temizleyin. Geçmişten silmek anahtarı geçersiz kılmaz — iptal etmek kılar.

&nbsp;

## PR açarken

- Başlığı Türkçe ve açıklayıcı yazın: *"ch03 README: 'nedensel dikkat' terim tutarsızlığı düzeltildi"*
- Tek PR'da tek konu ele alın; 12 dosyada 12 ayrı düzeltme birleştirmeyi zorlaştırır
- Neyi neden değiştirdiğinizi bir cümleyle açıklayın
- Anlamı değiştiren bir düzeltme yapıyorsanız (sadece üslup değil), orijinal İngilizce cümleyi de PR açıklamasına ekleyin

&nbsp;

## Lisans

Katkınız, deponun geri kalanıyla aynı lisans altında yayımlanır: **Apache License 2.0**. Ayrıntı: [LICENSE.txt](LICENSE.txt) ve [NOTICE](NOTICE).
