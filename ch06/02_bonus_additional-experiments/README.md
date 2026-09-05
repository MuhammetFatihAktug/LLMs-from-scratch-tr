# Ek Sınıflandırma İnce Ayarı Deneyleri

> 🇹🇷 **Türkçe çeviri.** Orijinal İngilizce sürüm: [README.md](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch06/02_bonus_additional-experiments/README.md) · Sonuç tablosu ve komutlar birebir korunmuştur.

Aşağıdaki tablo, çeşitli tasarım tercihleriyle ilgili ek soruları yanıtlamak için deneyler ekler. İlk satır ana bölümdekiyle aynı ayarları kullanır ve referans olarak alınır.
Örneğin,

- 1. ve 2. satırların karşılaştırılması şu soruyu yanıtlar: "Son token'ı mı yoksa ilk token'ı mı eğittiğimizde performans farkı ne olur?";
- 1. ve 3. satırların karşılaştırılması şu soruyu yanıtlar: "Son bloğun yerine yalnızca son katmanı eğittiğimizde performans farkı ne olur?";
- ve benzeri.

&nbsp;

|      | Model              | Weights    | Trainable token position | Trainable layers | Context length                                         | Training acc | Validation acc | Test acc | Training time | CPU/GPU |
| ---- | ------------------ | ---------- | ------------------------ | ---------------- | ------------------------------------------------------ | ------------ | -------------- | -------- | ------------- | ------- |
| 1    | gpt2-small (124M)  | pretrained | last                     | last_block       | longest train ex. (120)                                | 96.63%       | 99.33%         | 95.00%   | 0.28 min      | A100    |
| 2    | gpt2-small (124M)  | pretrained | first                    | last_block       | longest train ex. (120)                                | 78.46%       | 80.54%         | 75.00%   | 0.28 min      | A100    |
| 3    | gpt2-small (124M)  | pretrained | last                     | last_layer       | longest train ex. (120)                                | 78.65%       | 79.87%         | 72.00%   | 0.25 min      | A100    |
| 4    | gpt2-small (124M)  | pretrained | last                     | last_two_blocks  | longest train ex. (120)                                | 98.85%       | 98.66%         | 98.33%   | 0.33 min      | A100    |
| 5    | gpt2-small (124M)  | pretrained | last                     | all              | longest train ex. (120)                                | 99.62%       | 96.64%         | 96.67%   | 0.69 min      | A100    |
| 6    | gpt2-medium (355M) | pretrained | last                     | last_block       | longest train ex. (120)                                | 87.50%       | 91.28%         | 84.67%   | 0.75 min      | A100    |
| 7    | gpt2-large (774M)  | pretrained | last                     | last_block       | longest train ex. (120)                                | 99.52%       | 98.66%         | 96.67%   | 1.50 min      | A100    |
| 8    | gpt2-xl (1558M)    | pretrained | last                     | last_block       | longest train ex. (120)                                | 99.81%       | 99.81%         | 98.33%   | 2.83 min      | A100    |
| 9    | gpt2-xl (1558M)    | pretrained | last                     | all              | longest train ex. (120)                                | 100.00%      | 98.66%         | 98.67%   | 8.12 min      | A100    |
| 10   | gpt2-small (124M)  | random     | last                     | all              | longest train ex. (120)                                | 100.00%      | 96.64%         | 93.67%   | 0.69 min      | A100    |
| 11   | gpt2-small (124M)  | pretrained | last                     | LoRA             | longest train ex. (120)                                | 100.00%      | 97.32%         | 96.67%   | 0.75 min      | A100    |
| 12   | gpt2-xl (1558M)    | pretrained | last                     | LoRA             | longest train ex. (120)                                | 100.00%      | 98.66%         | 98.33%   | 5.79 min      | A100    |
| 13   | gpt2-small (124M)  | pretrained | last                     | last_block       | context length (1024)                                  | 83.08%       | 87.92%         | 78.33%   | 2.46 min      | A100    |
| 14   | gpt2-small (124M)  | pretrained | last                     | last_block       | variable: no padding (batch size 1)                    | 100.00%      | 98.66%         | 98.00%   | 1.75 min      | A100    |
| 15   | gpt2-small (124M)  | pretrained | last                     | last_block       | variable: no padding (batch size 8)                    | 99.33%       | 98.66%         | 98.33%   | 1.70 min      | A100    |
| 16   | gpt2-small (124M)  | pretrained | last                     | last_block       | flexible (last non-padding position)                   | 99.42%       | 98.66%         | 98.33%   | 0.30 min      | A100    |
| 17   | gpt2-small (124M)  | pretrained | last                     | last_block       | longest train ex. (120); but no causal mask            | 99.23%       | 98.66%         | 95.33%   | 0.29 min      | A100    |
| 18   | gpt2-small (124M)  | pretrained | last                     | last_block       | longest train ex. (120) and `ignore_index` for padding | 96.63%       | 99.33%         | 95.00%   | 0.28 min      | A100    |
| 19   | gpt2-small (124M)  | pretrained | last + pooled embeddings | last_block       | longest train ex. (120)                                | 97.79%       | 99.33%         | 96.33%   | 0.32 min      | A100    |

&nbsp;

### Kullanım

Deneyleri yeniden üretmek için aşağıdaki kodu kullanabilirsiniz:

- Satır 1: `python additional_experiments.py`
- Satır 2: `python additional_experiments.py --trainable_token_pos first`
- Satır 3: `python additional_experiments.py --trainable_layers last_layer`
- Satır 4: `python additional_experiments.py --trainable_layers last_two_blocks`
- Satır 5: `python additional_experiments.py --trainable_layers all`
- Satır 6: `python additional_experiments.py --model_size "gpt2-medium (355M)"`
- Satır 7: `python additional_experiments.py --model_size "gpt2-large (774M)"`
- Satır 8: `python additional_experiments.py --model_size "gpt2-xl (1558M)"`
- Satır 9: `python additional_experiments.py --model_size "gpt2-xl (1558M)"--trainable_layers all`
- Satır 10: `python additional_experiments.py --weights random --trainable_layers all`
- Satır 11: `python additional_experiments.py --trainable_layers lora --lora_rank 16 --lora_alpha 16`
- Satır 12: `python additional_experiments.py --trainable_layers lora --lora_rank 16 --lora_alpha 8 --model_size "gpt2-xl (1558M)"`
- Satır 13: `python additional_experiments.py --context_length "model_context_length"`
- Satır 14: `python additional_experiments.py --no_padding --batch_size 1`
- Satır 15: `python additional_experiments.py --no_padding --batch_size 1 --accumulation_steps 8`
- Satır 16: `python additional_experiments.py --trainable_token_pos "flexible"`
- Satır 17: `python additional_experiments.py --disable_causal_mask`
- Satır 18: `python additional_experiments.py --ignore_index 50256`
- Satır 19: `python additional_experiments.py --average_embeddings`

LLM'i ve veri kümesini bilinçli olarak küçük tuttum; böylece bir GPU'ya erişiminiz yoksa eğitimi MacBook Air M3 gibi sıradan bir dizüstü bilgisayarda (varsayılan ayarla) yaklaşık 15 dakikada çalıştırabilirsiniz.

&nbsp;

### Yorum

1. **Son ve İlk Çıkış Token Konumunu Eğitmek (1. ve 2. satır)**: Son çıkış token konumunu eğitmek, ilkine kıyasla belirgin biçimde daha iyi performans verir. Bu iyileşme, nedensel (causal) öz-dikkat maskesi nedeniyle beklenen bir sonuçtur.
2. **Son Transformer Bloğunu ve Son Katmanı Eğitmek (1. ve 3. satır)**: Son transformer bloğunun tamamını eğitmek de yalnızca son katmanı eğitmekten belirgin biçimde daha iyi sonuçlar verir.
3. **Son ve Son İki Transformer Bloğunu Eğitmek (1. ve 4. satır)**: Yalnızca son blok yerine son iki transformer bloğunu eğitmek, doğrulukta gözle görülür %3,33'lük bir artış sağlar.
4. **Son Transformer Bloğunu ve Tüm Katmanları Eğitmek (1. ve 5. satır)**: Tüm katmanları eğitmek, yalnızca son transformer bloğunu eğitmeye kıyasla ~%2'lik mütevazı bir iyileşme gösterir, ancak eğitim süresi açısından neredeyse üç kat daha uzun sürer. Ayrıca, 12 transformer bloğundan yalnızca son ikisini eğitmek kadar iyi performans göstermez.
5. **Daha Büyük Önceden Eğitilmiş Modeller Kullanmak (1'e karşı 6. satır ile 1'e karşı 7. ve 8. satırlar)**: 3 kat daha büyük, önceden eğitilmiş bir model kullanmak daha kötü sonuçlara yol açar. Ancak 5 kat daha büyük bir model kullanmak, beklendiği gibi ilk modele kıyasla performansı artırır. Benzer şekilde, 12 kat daha büyük model tahmin performansını daha da iyileştirir. (Orta boy model belki iyi ön eğitilmemişti veya bu ince ayar yapılandırması bu model için o kadar iyi çalışmıyor.)
6. **Rastgele Ağırlıklı Model ile Önceden Eğitilmiş Ağırlıklı Modeli Karşılaştırmak (1. ve 5. satıra karşı 10. satır)**: Rastgele ağırlıklı bir model kullanmak, önceden eğitilmiş ağırlıklara kıyasla yalnızca biraz daha kötü sonuçlar verir (%3 ve %1,3 fark).
7. **LoRA (Low-Rank Adaptation) Kullanmak ile Tüm Katmanları Eğitmek (11'e karşı 5. satır ve 12'ye karşı 9. satır)**: Modeli dondurup eğitilebilir LoRA katmanları eklemek (ayrıntılar için bkz. [Ek E](../../appendix-E/01_main-chapter-code/appendix-E.ipynb)), tüm model parametrelerini eğitmeye uygulanabilir bir alternatiftir ve performansı 1 puan bile artırır (11'e karşı 5. satır). LoRA kullanıldığında eğitim ile doğrulama doğruluğu arasındaki farkın ~%1 daha düşük olmasından görülebileceği gibi, bu muhtemelen daha az aşırı öğrenmeden (overfitting) kaynaklanır. Ayrıca, daha az parametrenin güncellenmesi gerektiği için LoRA kullanmak bellek açısından da daha verimlidir. Daha büyük modeli eğitirken (12'ye karşı 9. satır), LoRA'nın çok daha hızlı eğitildiğini de görebiliriz (8,12 dakika yerine 5,79 dakika).
8. **Girdiyi Tam Bağlam Uzunluğuna ve En Uzun Eğitim Örneğine Göre Doldurmak (1. ve 13. satır)**: Girdiyi desteklenen tam bağlam uzunluğuna kadar doldurmak (padding) kayda değer ölçüde daha kötü sonuç verir.
9. **Dolgulu ve dolgusuz (1'e karşı 14, 15 ve 16. satırlar)**: `--no_padding` seçeneği veri kümesindeki dolguyu devre dışı bırakır; girdiler değişken uzunlukta olduğu için modelin 1 yığın boyutuyla eğitilmesini gerektirir. Bu, daha iyi bir test doğruluğu verir ancak eğitim daha uzun sürer. 15. satırda, diğer deneylerdekiyle aynı yığın boyutuna ulaşmak için ek olarak 8 adımlı gradyan biriktirme (gradient accumulation) etkinleştirilir; bu, aşırı öğrenmeyi azaltmaya ve test kümesi doğruluğunu biraz artırmaya yardımcı olur. 16. satırda dolgu uygulanır, ancak token konumu son dolgusuz token'a göre seçilir. 16. satır, gradyan biriktirme kullanan 15. satırla matematiksel olarak benzer olmalıdır. Ancak eşit olmayan token sayıları durumunda gradyan biriktirmeyle ilgili bazı zorluklar nedeniyle küçük farklılıklar olabilir (bu, [şu](https://unsloth.ai/blog/gradient) blog yazısında tartışılmaktadır).
10. **Nedensel dikkat maskesini devre dışı bırakmak (1. ve 17. satır)**: Çok başlı dikkat modülünde kullanılan nedensel dikkat maskesini devre dışı bırakır. Bu, tüm token'ların diğer tüm token'lara dikkat edebileceği anlamına gelir. Model doğruluğu, nedensel maskeli GPT modeline kıyasla biraz iyileşir.
11. **Kayıp ve geri yayılımda dolgu indekslerini yok saymak (1. ve 18. satır)**: `--ignore_index 50256` ayarı, PyTorch'taki `cross_entropy` kayıp fonksiyonunda `<|endoftext|>` dolgu token'larını hariç tutar. Bu durumda hiçbir etkisi yoktur; çünkü çıkış katmanlarını, ikili sınıflandırma örneğinde token kimlikleri 0 veya 1 olacak şekilde değiştirdik. Ancak bu ayar, 7. bölümde modellere talimat ince ayarı yaparken faydalıdır.
12. **Gömmelerin tüm token'lar üzerinden ortalamasını almak (1. ve 19. satır)**: `--average_embeddings` ayarı, gömmelerin tüm token'lar üzerinden ortalamasını alır. Bu seçenek kullanılmazsa (varsayılan), yalnızca seçilen token konumundaki (`--trainable_token_pos` ile belirtilen) çıkış gömmeleri dikkate alınır; örneğin son token'ın gömmeleri. `--average_embeddings` etkinleştirildiğinde, tüm token'ların gömmeleri `--trainable_token_pos` ile seçilen konuma (varsayılan olarak son token) ortalama havuzlama (mean-pooling) ile toplanır. Görüldüğü gibi bu, çalışma süresinde yalnızca çok küçük bir artışla (0,28 dakikadan 0,32 dakikaya) performansı %95,00'ten %96,33'e çıkarır ve pratikte değerlendirmeye değer olabilir.
