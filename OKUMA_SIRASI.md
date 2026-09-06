# 📚 Okuma Sırası ve Depo Haritası (Türkçe)

Bu belge, [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) deposunu **nereden başlayıp hangi sırayla** okuyacağını ve **nerede ne olduğunu** özetler.

> Bu dosya çeviri değil, bu depo için hazırlanmış Türkçe bir rehberdir. Bu `tr` dalındaki tüm dosyalar Türkçedir; orijinal İngilizce sürüm `main` dalında dokunulmadan durur.

---

## ⏱️ Kısa Cevap: Nereden Başlamalıyım?

```
setup/  →  ch01  →  ch02  →  ch03  →  ch04  →  ch05  →  ch06  →  ch07
              ↑
        (PyTorch bilmiyorsan önce appendix-A)
```

**İlk oturumda yapılacaklar:**

1. [setup/README.md](setup/README.md) — ortamı kur (`pip install -r requirements.txt`)
2. [ch01/README.md](ch01/README.md) — kod yok, sadece kavramsal giriş
3. PyTorch'a yeniysen: [appendix-A/01_main-chapter-code/](appendix-A/01_main-chapter-code/) → `code-part1.ipynb`
4. [ch02/01_main-chapter-code/ch02.ipynb](ch02/01_main-chapter-code/ch02.ipynb) — ilk gerçek kod

**Altın kural:** Ana bölümleri (`01_main-chapter-code`) sırayla bitir. Bonus klasörleri (`02_…`, `03_…` ve sonrası) **isteğe bağlıdır** ve ilk turda atlanmalıdır — bunlar birer yan yol, ana yol değil.

---

## 🗺️ Depo Haritası: Nerede Ne Var?

| Klasör | Ne işe yarar | İlk turda? |
|---|---|---|
| [setup/](setup/) | Python/ortam kurulumu, Docker, bulut seçenekleri | ✅ Zorunlu |
| [ch01/](ch01/) | LLM'ler nedir, nasıl çalışır (kod yok) | ✅ Zorunlu |
| [ch02/](ch02/) | Tokenizasyon, BPE, gömme (embedding), veri yükleyici | ✅ Zorunlu |
| [ch03/](ch03/) | Dikkat mekanizmaları: basit → nedensel → çok başlı | ✅ Zorunlu |
| [ch04/](ch04/) | GPT mimarisini sıfırdan kurma, metin üretimi | ✅ Zorunlu |
| [ch05/](ch05/) | Ön eğitim (pretraining), kayıp fonksiyonu, GPT-2 ağırlıklarını yükleme | ✅ Zorunlu |
| [ch06/](ch06/) | Sınıflandırma için ince ayar (spam tespiti) | ✅ Zorunlu |
| [ch07/](ch07/) | Talimat takibi için ince ayar (instruction finetuning) | ✅ Zorunlu |
| [appendix-A/](appendix-A/) | PyTorch'a giriş (tensörler, autograd, GPU, DDP) | ⚠️ PyTorch bilmiyorsan |
| [appendix-B/](appendix-B/) | Kaynakça ve ileri okuma (kod yok) | ⏭️ Referans |
| [appendix-C/](appendix-C/) | Tüm alıştırma çözümlerine bağlantılar | ⏭️ Referans |
| [appendix-D/](appendix-D/) | Eğitim döngüsü iyileştirmeleri (warmup, cosine decay, grad clipping) | 🎯 ch05 sonrası |
| [appendix-E/](appendix-E/) | LoRA ile parametre-verimli ince ayar | 🎯 ch06 sonrası |
| [pkg/llms_from_scratch/](pkg/llms_from_scratch/) | Tüm bölüm kodunun `pip` paketi hâli | 🔧 Araç |
| [reasoning-from-scratch/](reasoning-from-scratch/) | **Devam kitabı** (akıl yürütme modelleri) — ayrı depo, submodule | 🚀 Kitap bitince |

---

## 📖 Adım Adım Okuma Planı

### 0. Hazırlık — [setup/](setup/)

| Sıra | Dosya | Not |
|---|---|---|
| 0.1 | [setup/README.md](setup/README.md) | Genel bakış + hızlı başlangıç |
| 0.2 | [setup/01_optional-python-setup-preferences/README.md](setup/01_optional-python-setup-preferences/README.md) | `uv` (önerilen) veya `conda` ile ortam |
| 0.3 | [setup/02_installing-python-libraries/README.md](setup/02_installing-python-libraries/README.md) | Paketleri kur + doğrula |
| 0.4 | [setup/03_optional-docker-environment/README.md](setup/03_optional-docker-environment/README.md) | *İsteğe bağlı* — Docker tercih edersen |
| 0.5 | [setup/04_optional-aws-sagemaker-notebook/README.md](setup/04_optional-aws-sagemaker-notebook/README.md) | *İsteğe bağlı* — AWS'de GPU |

En hızlı yol: depo kökünde `pip install -r requirements.txt`.

&nbsp;

### 0.5. PyTorch temeli — [appendix-A/](appendix-A/) *(gerekliyse)*

PyTorch'ta tensör, `nn.Module`, eğitim döngüsü ve `DataLoader` kavramları sana yabancıysa **ch02'ye geçmeden önce** burayı yap:

1. [appendix-A/01_main-chapter-code/code-part1.ipynb](appendix-A/01_main-chapter-code/code-part1.ipynb) — A.1–A.8: tensörler, autograd, eğitim döngüsü
2. [appendix-A/01_main-chapter-code/code-part2.ipynb](appendix-A/01_main-chapter-code/code-part2.ipynb) — A.9: GPU kullanımı
3. `DDP-script.py` — çok GPU'lu eğitim *(şimdilik atlanabilir)*
4. [appendix-A/02_setup-recommendations/](appendix-A/02_setup-recommendations/) — donanım ve ortam önerileri *(referans)*

&nbsp;

### 1. [ch01/](ch01/) — Büyük Dil Modellerini Anlamak

**Kod yok.** LLM'lerin ne olduğu, transformer mimarisine genel bakış, ön eğitim → ince ayar yaşam döngüsü. Türkçesi: [ch01/README.md](ch01/README.md)

&nbsp;

### 2. [ch02/](ch02/) — Metin Verileriyle Çalışmak

Metin → sayı dönüşümü. Burada tokenizer, byte pair encoding (BPE), token gömmeleri, konum gömmeleri ve kayan pencereli veri yükleyiciyi kurarsın.

- ▶️ **Ana kod:** [ch02/01_main-chapter-code/ch02.ipynb](ch02/01_main-chapter-code/ch02.ipynb)
- 📄 Özet not defteri: `dataloader.ipynb`
- ✏️ Alıştırma çözümleri: `exercise-solutions.ipynb`

<details>
<summary>Bonus klasörleri (ilk turda atla)</summary>

| Klasör | İçerik |
|---|---|
| [02_bonus_bytepair-encoder](ch02/02_bonus_bytepair-encoder) | Farklı BPE uygulamalarının hız karşılaştırması |
| [03_bonus_embedding-vs-matmul](ch02/03_bonus_embedding-vs-matmul) | Gömme katmanı = one-hot × doğrusal katman |
| [04_bonus_dataloader-intuition](ch02/04_bonus_dataloader-intuition) | Veri yükleyiciyi metin yerine sayılarla anlamak |
| [05_bpe-from-scratch](ch02/05_bpe-from-scratch) | **Tavsiye:** BPE'yi sıfırdan yazmak (çok öğretici) |
| [06_bonus_simple-tokenizer-v3](ch02/06_bonus_simple-tokenizer-v3) | Boşluk koruyan basit tokenizer varyantı |

</details>

&nbsp;

### 3. [ch03/](ch03/) — Dikkat (Attention) Mekanizmalarını Kodlamak

Kitabın **en kritik bölümü**. Basit dikkatten başlayıp sırasıyla eğitilebilir ağırlıklara, nedensel (causal) maskelemeye ve çok başlı dikkate (multi-head attention) ilerlersin. Burayı tam anlamadan ch04'e geçme.

- ▶️ **Ana kod:** [ch03/01_main-chapter-code/ch03.ipynb](ch03/01_main-chapter-code/ch03.ipynb)
- 📄 Özet not defteri: `multihead-attention.ipynb`

<details>
<summary>Bonus klasörleri</summary>

| Klasör | İçerik |
|---|---|
| [02_bonus_efficient-multihead-attention](ch03/02_bonus_efficient-multihead-attention) | MHA'nın hızlı varyantları (FlashAttention dahil) karşılaştırması |
| [03_understanding-buffers](ch03/03_understanding-buffers) | **Tavsiye:** `register_buffer` neden kullanılıyor? (ch03'te kafa karıştırırsa) |

</details>

&nbsp;

### 4. [ch04/](ch04/) — Sıfırdan GPT Modeli

Dikkat bloğunu LayerNorm, GELU, ileri beslemeli katman ve artık bağlantılarla birleştirip tam bir GPT-2 (124M) modeli kurarsın. Sonunda (henüz eğitilmemiş) model metin üretir.

- ▶️ **Ana kod:** [ch04/01_main-chapter-code/ch04.ipynb](ch04/01_main-chapter-code/ch04.ipynb)
- 📄 Tek dosyalık özet: `gpt.py`

<details>
<summary>Bonus klasörleri — modern LLM mimarileri (kitap bitince dön)</summary>

Bu klasörler modern LLM'lerin (Llama, Qwen, Gemma, DeepSeek) klasik GPT'den nasıl ayrıldığını anlatır. **Önerilen bonus sırası:**

| Sıra | Klasör | Konu |
|---|---|---|
| 1 | [03_kv-cache](ch04/03_kv-cache) | **En çok işe yarayan bonus.** Çıkarımı ~5× hızlandıran KV önbelleği |
| 2 | [04_gqa](ch04/04_gqa) | Gruplanmış Sorgu Dikkati — bugün neredeyse tüm LLM'lerde var |
| 3 | [06_swa](ch04/06_swa) | Kayan Pencere Dikkati (Gemma 3) |
| 4 | [10_kv-sharing](ch04/10_kv-sharing) | Katmanlar arası KV paylaşımı (Gemma 4) |
| 5 | [05_mla](ch04/05_mla) | Çok Başlı Gizil Dikkat (DeepSeek V3) |
| 6 | [09_dsa](ch04/09_dsa) | DeepSeek Seyrek Dikkati (V3.2) |
| 7 | [07_moe](ch04/07_moe) | Uzmanlar Karışımı — büyük modellerin sırrı |
| 8 | [08_deltanet](ch04/08_deltanet) | Gated DeltaNet, doğrusal dikkat (Qwen3-Next, Kimi Linear) |
| — | [02_performance-analysis](ch04/02_performance-analysis) | FLOPs analizi |

</details>

&nbsp;

### 5. [ch05/](ch05/) — Ön Eğitim (Pretraining)

Kayıp fonksiyonu, eğitim döngüsü, örnekleme stratejileri (temperature, top-k), model kaydetme/yükleme ve OpenAI'ın GPT-2 ağırlıklarını kendi modeline aktarma. **En yoğun bölüm.**

- ▶️ **Ana kod:** [ch05/01_main-chapter-code/ch05.ipynb](ch05/01_main-chapter-code/ch05.ipynb)
- 📄 Betikler: `gpt_train.py`, `gpt_generate.py`
- 🔁 Ağırlıklar indirilemezse: [ch05/02_alternative_weight_loading/](ch05/02_alternative_weight_loading/)

🎯 **Hemen ardından:** [appendix-D/01_main-chapter-code/appendix-D.ipynb](appendix-D/01_main-chapter-code/appendix-D.ipynb) — ısınma (warmup), kosinüs sönümleme ve gradyan kırpma. Kısa ve pratikte çok değerli.

<details>
<summary>Bonus klasörleri</summary>

**Pratik/eğitim tarafı:**

| Klasör | İçerik |
|---|---|
| [04_learning_rate_schedulers](ch05/04_learning_rate_schedulers) | appendix-D'ye yönlendirir |
| [10_llm-training-speed](ch05/10_llm-training-speed) | **Çok değerli.** 12k → 142k token/sn: adım adım optimizasyon |
| [18_muon](ch05/18_muon) | Muon optimize edici |
| [03_bonus_pretraining_on_gutenberg](ch05/03_bonus_pretraining_on_gutenberg) | Gerçek büyük derlem üzerinde ön eğitim (uzun sürer) |
| [05_bonus_hparam_tuning](ch05/05_bonus_hparam_tuning) | Hiperparametre ızgara araması |
| [08_memory_efficient_weight_loading](ch05/08_memory_efficient_weight_loading) | Büyük modelleri az RAM'le yükleme |
| [09_extending-tokenizers](ch05/09_extending-tokenizers) | Tokenizer'a yeni özel token ekleme |
| [06_user_interface](ch05/06_user_interface) | Chainlit ile sohbet arayüzü |

**Gerçek modelleri sıfırdan yazma (en eğlenceli kısım):**

| Klasör | Model |
|---|---|
| [07_gpt_to_llama](ch05/07_gpt_to_llama) | **Buradan başla.** GPT → Llama 2 → Llama 3.2 adım adım dönüşüm |
| [11_qwen3](ch05/11_qwen3) | Qwen3 (yoğun ve MoE varyantları) |
| [12_gemma3](ch05/12_gemma3) | Gemma 3 270M — küçük, hızlı deneme için ideal |
| [13_olmo3](ch05/13_olmo3) | Olmo 3 7B / 32B |
| [15_tiny-aya](ch05/15_tiny-aya) | Tiny Aya 3.35B — çok dilli (Türkçe dahil) |
| [16_qwen3.5](ch05/16_qwen3.5) | Qwen3.5 0.8B (doğrusal dikkat hibriti) |
| [17_gemma4](ch05/17_gemma4) | Gemma 4 E2B / E4B |
| [14_ch05_with_other_llms](ch05/14_ch05_with_other_llms) | Bölüm 5'i GPT-2 yerine başka modellerle çalıştırma |

</details>

&nbsp;

### 6. [ch06/](ch06/) — Sınıflandırma İçin İnce Ayar

Önceden eğitilmiş GPT'yi spam sınıflandırıcısına dönüştürürsün: çıkış başını değiştirme, hangi katmanların dondurulacağı, sınıflandırma kaybı ve doğruluk ölçümü.

- ▶️ **Ana kod:** [ch06/01_main-chapter-code/ch06.ipynb](ch06/01_main-chapter-code/ch06.ipynb)

🎯 **Hemen ardından:** [appendix-E/01_main-chapter-code/appendix-E.ipynb](appendix-E/01_main-chapter-code/appendix-E.ipynb) — LoRA. Pratikte en çok kullanacağın ince ayar yöntemi.

<details>
<summary>Bonus klasörleri</summary>

| Klasör | İçerik |
|---|---|
| [02_bonus_additional-experiments](ch06/02_bonus_additional-experiments) | **Çok öğretici.** 19 farklı tasarım tercihinin ölçülmüş etkisi |
| [03_bonus_imdb-classification](ch06/03_bonus_imdb-classification) | GPT-2 vs BERT/RoBERTa/ModernBERT karşılaştırması |
| [04_user_interface](ch06/04_user_interface) | Spam sınıflandırıcı için web arayüzü |

</details>

&nbsp;

### 7. [ch07/](ch07/) — Talimatları İzlemek İçin İnce Ayar

Kitabın finali: talimat veri kümesi biçimlendirme, özel `collate` fonksiyonu, maskeleme, SFT eğitimi ve modelin yanıtlarını değerlendirme. Çıktı: kendi küçük "ChatGPT"in.

- ▶️ **Ana kod:** [ch07/01_main-chapter-code/ch07.ipynb](ch07/01_main-chapter-code/ch07.ipynb)
- 📄 Betikler: `gpt_instruction_finetuning.py`, `ollama_evaluate.py`

<details>
<summary>Bonus klasörleri</summary>

| Klasör | İçerik |
|---|---|
| [04_preference-tuning-with-dpo](ch07/04_preference-tuning-with-dpo) | **En önemlisi.** DPO ile tercih hizalaması (RLHF'in basit alternatifi) |
| [03_model-evaluation](ch07/03_model-evaluation) | GPT-4 / Ollama ile otomatik değerlendirme |
| [05_dataset-generation](ch07/05_dataset-generation) | Sentetik talimat verisi üretme ve iyileştirme |
| [02_dataset-utilities](ch07/02_dataset-utilities) | Yakın kopya bulma, edilgen çatı girdileri |
| [06_user_interface](ch07/06_user_interface) | Sohbet arayüzü |

</details>

---

## 🚀 Kitaptan Sonra: [reasoning-from-scratch/](reasoning-from-scratch/)

Bu klasör, aynı yazarın devam kitabı [*Build A Reasoning Model (From Scratch)*](https://mng.bz/lZ5B) deposunun bir **git submodule**'üdür. Önceden eğitilmiş bir modelden başlayıp akıl yürütme (reasoning) yeteneği kazandırmayı anlatır.

Klasör boş görünüyorsa şu komutla doldur:

```bash
git submodule update --init --recursive reasoning-from-scratch
```

Bu komut alt modülü doğrudan **Türkçe** sürümde açar: `tr` dalında kayıtlı submodule işaretçisi,
alt deponun çevrilmiş commit'ini gösterir. (Alt deponun da kendi `main` ve `tr` dalları vardır;
İngilizce aslı için `cd reasoning-from-scratch && git show main:<dosya>`.)

Önerilen sıra (o depo içinde):

| Sıra | Klasör | Konu |
|---|---|---|
| 1 | `ch01/` | Akıl yürütme modellerini anlamak (kod yok) |
| 2 | `ch02/` | Önceden eğitilmiş bir LLM ile metin üretmek — Qwen3 yükleme, KV önbelleği, `torch.compile` |
| 3 | `ch03/` | Akıl yürütme modellerini değerlendirmek — MATH-500 doğrulayıcısı |
| 4 | `ch04/` | Çıkarım anında ölçekleme — düşünce zinciri, sıcaklık/top-p, öz tutarlılık |
| 5 | `ch05/` | Öz iyileştirme (self-refinement) ve log-olasılık ile puanlama |
| 6 | `ch06/` | **RLVR + GRPO** — pekiştirmeli öğrenme ile akıl yürütme eğitimi |
| 7 | `ch07/` | GRPO'yu iyileştirmek — kırpılmış politika oranı, KL terimi, biçim ödülü |
| 8 | `ch08/` | Damıtma (distillation) ile verimli akıl yürütme |
| — | `chC/` | **Ek C:** Qwen3 kaynak kodu (LLMs-from-scratch `ch05/11_qwen3` devamı) |
| — | `chD/` | **Ek D:** Daha büyük Qwen3 modellerini kullanmak (1.7B–32B) |
| — | `chE/` | **Ek E:** Yığınlama (batching) ve verim odaklı çalıştırma |
| — | `chF/` | **Ek F:** Değerlendirme yöntemleri — MMLU, liderlik tabloları, LLM-as-a-judge |
| — | `chG/` | **Ek G:** Sohbet arayüzü oluşturmak (chainlit) |

> `chC/` istersen `ch02/` ile birlikte okunabilir: `ch02` modeli *kullanmayı*, `chC` ise o modelin
> kodunun içini anlatır. Diğer ekler (D–G) ilgili bölümü bitirdikten sonra istediğin sırada okunabilir.

---

## 🧭 Alternatif Rotalar

**"Sadece transformer'ı anlamak istiyorum" (~1 hafta)**
`ch01` → `ch02` → `ch03` → `ch04` — burada durabilirsin. Model çalışır, metin üretir.

**"Kendi modelimi eğitmek istiyorum" (~3 hafta)**
Yukarıdakiler + `ch05` + `appendix-D` + `ch05/10_llm-training-speed`

**"Hazır modeli kendi işime uyarlamak istiyorum" (~4 hafta)**
Yukarıdakiler + `ch06` + `appendix-E` (LoRA) + `ch07` + `ch07/04_preference-tuning-with-dpo`

**"Modern LLM mimarilerini merak ediyorum"**
`ch04` sonrası → `ch04/03_kv-cache` → `ch04/04_gqa` → `ch05/07_gpt_to_llama` → `ch05/11_qwen3`

---

## 💡 Pratik İpuçları

- **Not defterlerini oku, kopyala-yapıştır yapma.** Her hücreyi kendin yazmak, anlama hızını belirgin biçimde artırır.
- **Alıştırmaları atlama.** Çözümleri her bölümün `01_main-chapter-code/exercise-solutions.ipynb` dosyasında, listesi [appendix-C](appendix-C/) içinde.
- **Tensör şekillerini yazdır.** Özellikle ch03'te `print(x.shape)` en iyi arkadaşın.
- **GPU şart değil.** Ana bölümler dizüstü bilgisayarda çalışır. GPU sadece ch05–ch07'yi hızlandırır.
- **Takılırsan:** [troubleshooting.md](troubleshooting.md) ve [GitHub Discussions](https://github.com/rasbt/LLMs-from-scratch/discussions).
- **Paket olarak kullanmak istersen:** [pkg/llms_from_scratch/README.md](pkg/llms_from_scratch/README.md) — `pip install llms-from-scratch`.

---

## 📁 Bu Türkçe Sürüm Hakkında

Bu depo, [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) deposunun Türkçe çevirisidir.

**Dal yapısı:**

| Dal | İçerik |
|---|---|
| `main` | Orijinal İngilizce sürüm — hiç değiştirilmedi, upstream ile birebir aynı |
| `tr` | Türkçe sürüm — dosyaların içi çevrildi (şu an bu daldasın) |

Bir dosyanın İngilizce aslını görmek istersen her Türkçe dosyanın başındaki banner upstream bağlantısını verir. Yerelde de bakabilirsin:

```bash
git show main:ch03/README.md        # tek dosyanın İngilizcesi
git diff main tr -- ch03/README.md  # yan yana fark
```

**Çeviri kuralları:**

- **Çevrilen:** markdown metinleri, başlıklar, tablo başlıkları, kod hücrelerindeki `#` yorumları ve
  Python (`.py`) kaynak dosyalarındaki `#` yorumları.
- **Çevrilmeyen (bilinçli):** kod içindeki string'ler ve promptlar (ör. `"Every effort moves you"` — çevrilirse model çıktısı değişir), chat template'ler, değişken ve fonksiyon adları, kütüphane API'leri, kayıtlı hücre çıktıları, eğitim günlükleri, ölçüm tabloları, markdown içindeki ``` kod blokları, lisans/atıf metinleri. Bunlar kopyalayıp çalıştıracağın komutlarla ve kitabın basılı koduyla eşleşmeli.
- **Terim tutarlılığı:** aynı İngilizce terim depo genelinde aynı Türkçe karşılığı alır. Karşılık listesi: [SOZLUK.md](SOZLUK.md). Bir terim bir bölümde ilk geçtiğinde İngilizcesi parantez içinde verilir — "dikkat (attention)" gibi.
- **Yapı korunur:** madde sayısı, mantıksal sıra ve şekil/bağlantı yerleşimi orijinaldeki gibidir; bunlar öğretim sırasını taşır.

**Lisans ve atıf:**

Orijinal eser Sebastian Raschka'ya aittir ve Apache License 2.0 ile lisanslanmıştır (`Copyright 2023-2026 Sebastian Raschka`). Bu çeviri de aynı lisans altındadır; [LICENSE.txt](LICENSE.txt) ve telif bildirimi korunmuştur. Apache 2.0 §4(b) gereği yapılan değişiklikler [NOTICE](NOTICE) dosyasında belirtilmiştir. Notebook'ların başındaki özgün atıf tablosu her dosyada olduğu gibi bırakılmıştır.

Bu resmî bir çeviri değildir; kitabın yazarı veya Manning tarafından onaylanmamıştır.
