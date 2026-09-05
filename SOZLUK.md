# 📖 Terim Sözlüğü (Glossary)

Bu depodaki Türkçe çevirinin terim karşılıkları. **Amaç tutarlılık:** aynı İngilizce terim, depo genelinde her yerde aynı Türkçe karşılığı alır. Bir kavramın bölüm 3'te "dikkat", bölüm 5'te "attention" olarak geçmesi okuyucunun kafasını karıştırır ve öğrenmeyi bozar.

**Kural 1 — İlk geçişte ikisi birden.** Bir terim bir dosyada ilk geçtiğinde İngilizcesi parantez içinde verilir: *"nedensel dikkat (causal attention)"*. Aynı dosyada sonraki geçişlerde sadece Türkçesi kullanılır.

**Kural 2 — Kod dokunulmazdır.** Aşağıdaki karşılıklar yalnızca **anlatı metni** içindir. Kod içindeki değişken adları, fonksiyon adları, argümanlar (`--max_new_tokens`), metin sabitleri ve model istemleri asla çevrilmez.

**Kural 3 — Yerleşik terim İngilizce kalır.** Türkçe teknik literatürde İngilizcesi yerleşmiş terimler zorlama çevrilmez (token, transformer, softmax, dropout…). Liste aşağıda.

---

## Mimari ve katmanlar

| İngilizce | Türkçe |
|---|---|
| attention | dikkat |
| self-attention | öz-dikkat |
| causal attention | nedensel dikkat |
| multi-head attention (MHA) | çok başlı dikkat |
| grouped-query attention (GQA) | gruplanmış sorgu dikkati |
| multi-head latent attention (MLA) | çok başlı gizil dikkat |
| sliding window attention (SWA) | kayan pencere dikkati |
| sparse attention | seyrek dikkat |
| linear attention | doğrusal dikkat |
| attention head | dikkat başı |
| head dimension | baş boyutu |
| query / key / value | sorgu / anahtar / değer |
| embedding | gömme |
| embedding layer | gömme katmanı |
| positional embedding | konum gömmesi |
| linear layer | doğrusal katman |
| fully connected layer | tam bağlantılı katman |
| feed-forward (FFN) | ileri beslemeli |
| projection | izdüşüm |
| residual connection | artık bağlantı |
| normalization | normalleştirme |
| activation | aktivasyon |
| hidden state | gizli durum |
| latent | gizil |
| recurrent | yinelemeli |
| dense | yoğun |
| sparse | seyrek |
| mixture-of-experts (MoE) | uzmanlar karışımı |
| expert | uzman |
| router | yönlendirici |
| gate / gating | kapı / kapılama |
| decay | sönümleme |
| mask / masking | maske / maskeleme |
| padding | dolgu |
| context length | bağlam uzunluğu |
| vocabulary | sözlük |
| weights | ağırlıklar |
| parameter | parametre |

## Eğitim

| İngilizce | Türkçe |
|---|---|
| pretraining | ön eğitim |
| finetuning | ince ayar |
| instruction finetuning | talimat ince ayarı |
| preference tuning | tercih ince ayarı |
| classification | sınıflandırma |
| training loop | eğitim döngüsü |
| loss | kayıp |
| loss function | kayıp fonksiyonu |
| gradient | gradyan |
| gradient clipping | gradyan kırpma |
| gradient accumulation | gradyan biriktirme |
| learning rate | öğrenme oranı |
| scheduler | zamanlayıcı |
| warmup | ısınma |
| cosine decay | kosinüs sönümleme |
| optimizer | optimize edici |
| epoch | dönem |
| step | adım |
| batch | yığın |
| batch size | yığın boyutu |
| batched | yığınlanmış |
| forward pass | ileri geçiş |
| backward pass | geri geçiş |
| overfitting | aşırı öğrenme |
| hyperparameter | hiperparametre |
| grid search | ızgara arama |
| checkpoint | kontrol noktası |
| baseline | temel çizgi |
| ablation study | ablasyon çalışması |
| distillation | damıtma |
| teacher (model) | öğretmen (model) |
| reinforcement learning | pekiştirmeli öğrenme |
| reward | ödül |
| policy | politika |
| advantage | avantaj |

## Veri

| İngilizce | Türkçe |
|---|---|
| dataset | veri kümesi |
| data loader | veri yükleyici |
| training / validation / test split | eğitim / doğrulama / test ayrımı |
| sample, example | örnek |
| corpus | derlem |
| near duplicate | yakın kopya |
| ground truth | doğru (referans) yanıt |

## Çıkarım ve üretim

| İngilizce | Türkçe |
|---|---|
| inference | çıkarım |
| inference-time scaling | çıkarım zamanı ölçeklendirme |
| text generation | metin üretimi |
| decoding | kod çözme |
| greedy decoding | açgözlü kod çözme |
| sampling | örnekleme |
| prompt | istem |
| chain-of-thought (CoT) | düşünce zinciri |
| self-consistency | öz tutarlılık |
| self-refinement | öz iyileştirme |
| majority vote | çoğunluk oyu |
| reasoning | akıl yürütme |
| KV cache | KV önbelleği |
| cache | önbellek |
| streaming | akış |

## Değerlendirme

| İngilizce | Türkçe |
|---|---|
| evaluation | değerlendirme |
| benchmark | kıyaslama |
| accuracy | doğruluk |
| verifier | doğrulayıcı |
| leaderboard | liderlik tablosu |
| LLM-as-a-judge | hakem olarak LLM |
| scorer | puanlayıcı |
| teacher forcing | öğretmen zorlaması |
| tie-breaker | eşitlik bozucu |

## Başarım ve donanım

| İngilizce | Türkçe |
|---|---|
| memory | bellek |
| memory usage | bellek kullanımı |
| allocation | tahsis |
| pre-allocate | önceden tahsis etmek |
| throughput | verim |
| speed-up | hızlanma |
| bottleneck | darboğaz |
| fused | kaynaşık |
| kernel | çekirdek |
| compilation | derleme |
| trade-off | ödünleşim |

## Yazılım ve ortam

| İngilizce | Türkçe |
|---|---|
| repository | depo |
| branch | dal |
| fork | çatal / çatallamak |
| merge | birleştirme |
| diff | fark |
| notebook | not defteri |
| cell | hücre |
| script | betik |
| package | paket |
| library | kütüphane |
| dependency | bağımlılık |
| environment | ortam |
| virtual environment | sanal ortam |
| setup | kurulum |
| wrapper | sarmalayıcı |
| drop-in replacement | doğrudan yerine geçen |
| standalone | bağımsız |
| self-contained | kendi kendine yeten |
| edge case | uç durum |
| workaround | geçici çözüm |
| flag | bayrak |
| default | varsayılan |
| optional | isteğe bağlı |

---

## İngilizce kalan terimler

Türkçe teknik kullanımda yerleşmiş oldukları için çevrilmez:

`token` · `tokenizer` · `transformer` · `softmax` · `dropout` · `bias` · `logit` ·
`state dict` · `buffer` · `rollout` · `temperature` · `top-k` · `top-p` ·
`commit` · `pipeline` (bağlama göre "hat") · `LLM` · `GPU` · `CPU` · `RAM` · `VRAM`

Model, kütüphane, dosya ve API adları da olduğu gibi kalır:
`GPT`, `Llama`, `Qwen3`, `Gemma`, `PyTorch`, `AdamW`, `RMSNorm`, `SwiGLU`, `RoPE`,
`LoRA`, `GRPO`, `DPO`, `MATH-500`, `MMLU`, `nn.Module`, `load_state_dict` …

---

## Çevrilmeyenler (kod bütünlüğü için)

Bunlara dokunmak kodu bozar ya da kitabın basılı hâliyle uyumu kırar:

- Kod içindeki metin sabitleri: `prompt = "Every effort moves you"` gibi
- Model istemleri ve sohbet şablonları (`<|im_start|>`, `<think>` …)
- Değişken, fonksiyon, sınıf adları; komut satırı argümanları (`--num_rollouts`)
- Not defterlerinde kayıtlı hücre çıktıları
- Eğitim günlükleri, başarım ölçüm tabloları, kıyaslama sayıları
- Dosya ve klasör adları

---

*Yeni bir terimle karşılaşırsan buraya ekle. Bir terimin karşılığını değiştirirsen, depo genelinde hepsini birden değiştir — yarısı eski yarısı yeni kalmasın.*
