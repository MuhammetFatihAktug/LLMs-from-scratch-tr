# 🇹🇷 Türkçe Örnek Veri Setleri

Bu depodaki bölümler, İngilizce metin ve veri dosyalarıyla çalışır. Bu dosya, **aynı deneyleri Türkçe veriyle tekrarlayabilmen** için eklenen Türkçe karşılıkları anlatır.

> **Önemli:** Orijinal İngilizce veri dosyalarının hiçbiri değiştirilmedi veya silinmedi.
> Türkçe dosyalar onların **yanına** eklendi. Kitaptaki token sayıları, kayıp değerleri ve
> örnek çıktılar orijinal dosyalarla eşleşir; Türkçe dosyalarla çalıştırdığında **farklı
> sayılar görmen normaldir** — zaten mesele de budur.

&nbsp;

## Neden Türkçe veriyle de denemelisin?

Türkçe **sondan eklemeli** bir dildir: `göz → gözlük → gözlükçü → gözlükçüler → gözlükçülerden`.
GPT-2'nin BPE tokenizer'ı ağırlıklı olarak İngilizce metinle eğitildiği için Türkçe kelimeleri
çok daha fazla parçaya böler. Bu, kitapta anlatılan şu kavramları **somut olarak** görmeni sağlar:

- Aynı uzunluktaki metin neden farklı sayıda token üretir?
- Sözcük dağarcığı (vocabulary) neden dile bağımlıdır?
- Bağlam uzunluğu (context length) pratikte ne kadar metne denk gelir?
- Aynı model, eğitim verisinin dili değişince nasıl davranır?

Bu depodaki dosyalarla ölçülen gerçek sonuç (GPT-2 BPE tokenizer'ı ile):

| Dosya | Dil | Karakter | Token | Karakter/token |
|---|---|---:|---:|---:|
| `the-verdict.txt` | İngilizce | 20.479 | 5.145 | 3,98 |
| `karar.txt` | Türkçe | 23.987 | 12.793 | **1,88** |
| `small-text-sample.txt` | İngilizce | 1.962 | 412 | 4,76 |
| `kucuk-metin-ornegi.txt` | Türkçe | 1.705 | 892 | **1,91** |

Yani GPT-2'nin tokenizer'ı, aynı miktarda Türkçe metni işlemek için **yaklaşık iki katı token**
harcıyor. Bunun iki doğrudan sonucu var: aynı `context_length` ile yarı yarıya daha az metin
görebilirsin ve aynı metin için eğitim maliyetin iki katına çıkar.

Kendin ölçmek istersen:

```python
import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")

for dosya in ["the-verdict.txt", "karar.txt"]:
    with open(dosya, "r", encoding="utf-8") as f:
        metin = f.read()
    tokenlar = tokenizer.encode(metin)
    print(f"{dosya:18} karakter: {len(metin):6} | token: {len(tokenlar):6} "
          f"| karakter/token: {len(metin)/len(tokenlar):.2f}")
```

&nbsp;

## Dosya listesi

### Düz metin (ön eğitim ve tokenizasyon)

| Türkçe dosya | Orijinali | Nerede |
|---|---|---|
| `karar.txt` | `the-verdict.txt` | [ch02/01_main-chapter-code/](ch02/01_main-chapter-code/), [ch05/05_bonus_hparam_tuning/](ch05/05_bonus_hparam_tuning/), [appendix-D/01_main-chapter-code/](appendix-D/01_main-chapter-code/) |
| `kucuk-metin-ornegi.txt` | `small-text-sample.txt` | [ch03/01_main-chapter-code/](ch03/01_main-chapter-code/) |

`karar.txt`, bu Türkçe sürüm için yazılmış **özgün** bir öyküdür (telifi bu deponun lisansına tabidir).
Orijinal `the-verdict.txt` gibi birinci tekil kişi anlatımlı, diyalog içeren edebî bir metindir ve
uzunluğu ona yakındır (23.987 karakter / 3.349 kelime; orijinali 20.479 karakter / 3.634 kelime).
Konusu da bölümle bağlantılıdır: yazıyı *biçim* olarak görmekle *anlam* olarak görmek arasındaki fark —
yani tam olarak bir tokenizer'ın yaptığı ayrım.

`kucuk-metin-ornegi.txt` de aynı mantıkla, ch03'ün konusuna (dikkat) bağlanan kısa bir anlatıdır
(1.705 karakter; orijinali 1.962 karakter).

### Talimat verisi (ch07)

| Türkçe dosya | Orijinali | Kayıt | Nerede |
|---|---|---|---|
| `talimat-verisi.json` | `instruction-data.json` | 160 | [ch07/01_main-chapter-code/](ch07/01_main-chapter-code/) |
| `talimat-ornekleri.json` | `instruction-examples.json` | 65 | [ch07/02_dataset-utilities/](ch07/02_dataset-utilities/) |
| `degerlendirme-ornek-verisi.json` | `eval-example-data.json` | 30 | [ch07/03_model-evaluation/](ch07/03_model-evaluation/) |
| `talimat-verisi-tercihli.json` | `instruction-data-with-preference.json` | 160 | [ch07/04_preference-tuning-with-dpo/](ch07/04_preference-tuning-with-dpo/) |
| `talimat-verisi-llama3-8b.json` | `instruction-data-llama3-7b.json` | 5 | [ch07/05_dataset-generation/](ch07/05_dataset-generation/) |

Alan adları orijinaliyle **birebir aynıdır** (`instruction`, `input`, `output`, `chosen`, `rejected`,
`model 1 response`, `model 2 response`), böylece kod hiç değişmeden çalışır — yalnızca dosya adını
değiştirmen yeterli.

> **Boyut farkı:** Orijinal `instruction-data.json` 1.100 kayıttır; Türkçe karşılığı 160 kayıttır.
> Bu, ince ayar boru hattının **çalıştığını görmen** için yeterlidir ama modelin niteliği doğal olarak
> daha düşük olur. Veri kümesini büyütmek istersen aşağıdaki "Kendi verinle çalışmak" bölümüne bak.

Bazı dosyaların içeriği bilinçli olarak o klasörün amacına göre hazırlandı:

- `talimat-ornekleri.json`, [find-near-duplicates.py](ch07/02_dataset-utilities/find-near-duplicates.py)
  demosunun anlamlı çalışması için **kasıtlı yakın-kopya** talimat çiftleri içerir
  (ör. "Aşağıdaki cümleyi edilgen çatıya çevir." / "Aşağıdaki cümleyi edilgen yapıya dönüştür.").
- `degerlendirme-ornek-verisi.json` içinde `model 1 response` doğru, `model 2 response` ise
  bilerek kısmen yanlış ya da eksiktir (ör. "Japonya'nın başkenti Kyoto'dur.") — böylece
  LLM-hakem puanlaması gerçekten bir fark ölçer.
- `talimat-verisi-tercihli.json` içinde `chosen` kibar, `rejected` ise sert/küçümseyici bir üslupla
  yazılmıştır; DPO'nun öğrenmesi beklenen şey tam olarak bu üslup farkıdır.

&nbsp;

## Nasıl kullanılır?

Tek yapman gereken, ilgili hücredeki dosya adını değiştirmek.

**ch02 — tokenizasyon ve veri yükleyici**

```python
# ch02/01_main-chapter-code/ch02.ipynb, dataloader.ipynb, exercise-solutions.ipynb
with open("the-verdict.txt", "r", encoding="utf-8") as f:   # önce
with open("karar.txt", "r", encoding="utf-8") as f:         # sonra
```

**ch03 — dikkat mekanizması**

```python
# ch03/01_main-chapter-code/multihead-attention.ipynb
with open("small-text-sample.txt", "r", encoding="utf-8") as f:   # önce
with open("kucuk-metin-ornegi.txt", "r", encoding="utf-8") as f:  # sonra
```

**ch05 — ön eğitim**

`ch05/01_main-chapter-code/ch05.ipynb` dosyayı indirir; indirme hücresini atlayıp
`file_path` değerini `"karar.txt"` yapman ve dosyayı o klasöre kopyalaman yeterli.
Betiklerde (`gpt_train.py`, `gpt_train_muon.py`, `hparam_search.py`) ise:

```python
file_path = "the-verdict.txt"   # önce
file_path = "karar.txt"         # sonra
# indirme satırını (url / urlretrieve) yorum satırı yap
```

**ch07 — talimat ince ayarı**

```python
# ch07/01_main-chapter-code/ch07.ipynb ve gpt_instruction_finetuning.py
file_path = "instruction-data.json"   # önce
file_path = "talimat-verisi.json"     # sonra
# indirme satırını yorum satırı yap
```

Diğer ch07 klasörlerinde de aynı mantık geçerlidir: `json_file` / `file_path` değerini
yukarıdaki tablodaki Türkçe dosya adıyla değiştir.

&nbsp;

## Ne beklemelisin?

- **ch02/ch05:** Yukarıdaki tabloda görüldüğü gibi Türkçe metin, karakter başına yaklaşık **iki kat**
  token üretir. `karar.txt` 12.793 token verir (`the-verdict.txt` 5.145); yani ch05 ön eğitimi için
  aslında daha fazla eğitim verin olur ama aynı `context_length` daha az kelimeyi kapsar. Kayıp değerleri de
  İngilizce koşuya göre daha yüksek başlar; GPT-2'nin sözcük dağarcığı Türkçe için uygun değildir.
- **ch07:** 160 kayıtla eğitilen model, üslubu ve biçimi öğrenir ama bilgi doğruluğu zayıf kalır.
  Amaç doğru cevaplar almak değil, **boru hattını Türkçe veriyle uçtan uca çalıştırmaktır**.
- **Bu bir kusur değil, dersin kendisidir.** Tokenizer'ın dili, model başarımını doğrudan belirler.

&nbsp;

## Kendi verinle çalışmak

Bu dosyalar bir başlangıç noktasıdır; kendi metnini koymak en öğretici yoldur.

- **Düz metin için:** UTF-8 kodlu, paragrafları boş satırla ayrılmış herhangi bir `.txt` yeterlidir.
  Ön eğitim (ch05) için en az ~20.000 karakter öneririm; daha azıyla eğitim döngüsü anlamlı çalışmaz.
  Telif konusuna dikkat et: kamuya açık (public domain) metinler ya da kendi yazdıkların güvenlidir.
- **Talimat verisi için:** Aynı JSON şemasını koru (`instruction`, `input`, `output`).
  `input` alanı boş olabilir; orijinal veri kümesinde kayıtların yaklaşık %30'unda doludur.
  Kayıt sayısını artırmak için [ch07/05_dataset-generation/](ch07/05_dataset-generation/) klasöründeki
  yöntemi kullanabilirsin: yerel bir modele (Ollama + Llama 3) Türkçe talimat ürettirip
  çıktıyı bu şemaya dökebilirsin.
- **Tercih verisi (DPO) için:** `chosen` ve `rejected` alanlarının **tek bir eksende** farklılaşması
  önemlidir (burada: kibarlık). İki alan birden çok boyutta farklılaşırsa model neyi öğreneceğini
  bulamaz.

&nbsp;

## Lisans

Bu klasördeki Türkçe metin ve veri dosyalarının tamamı bu depo için **özgün olarak üretilmiştir**;
başka bir eserden alıntı ya da çeviri değildir. Deponun geri kalanıyla aynı lisans altındadır
(Apache License 2.0). Ayrıntı için [NOTICE](NOTICE) dosyasına bakabilirsin.

Orijinal İngilizce veri dosyaları (`the-verdict.txt`, `small-text-sample.txt`, `instruction-data.json`
ve diğerleri) Sebastian Raschka'nın deposundan gelir ve **değiştirilmemiştir**.
