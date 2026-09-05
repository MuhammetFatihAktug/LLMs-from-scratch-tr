# Bölüm 5: Etiketlenmemiş Veri Üzerinde Ön Eğitim (Pretraining)

> 🇹🇷 **Türkçe çeviri.** Orijinal İngilizce sürüm: [README.md](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch05/01_main-chapter-code/README.md)

### Ana Bölüm Kodu

- [ch05.ipynb](ch05.ipynb) bölümde geçtiği haliyle tüm kodu içerir
- [previous_chapters.py](previous_chapters.py) önceki bölümlerdeki `MultiHeadAttention` modülünü ve `GPTModel` sınıfını içeren bir Python modülüdür; GPT modelini ön eğitmek için [ch05.ipynb](ch05.ipynb) içinde bu modülü içe aktarırız
- [gpt_download.py](gpt_download.py) önceden eğitilmiş GPT model ağırlıklarını indirmek için yardımcı fonksiyonları içerir
- [exercise-solutions.ipynb](exercise-solutions.ipynb) bu bölüme ait alıştırma çözümlerini içerir

### İsteğe Bağlı Kod

- [gpt_train.py](gpt_train.py) GPT modelini eğitmek için [ch05.ipynb](ch05.ipynb) içinde uyguladığımız kodu barındıran bağımsız bir Python betik dosyasıdır (bu bölümü özetleyen bir kod dosyası olarak düşünebilirsiniz)
- [gpt_generate.py](gpt_generate.py) OpenAI'ın önceden eğitilmiş model ağırlıklarını yüklemek ve kullanmak için [ch05.ipynb](ch05.ipynb) içinde uyguladığımız kodu barındıran bağımsız bir Python betik dosyasıdır
