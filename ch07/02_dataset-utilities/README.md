# Bölüm 7: Talimatları İzlemek İçin İnce Ayar

> 🇹🇷 **Türkçe çeviri.** Orijinal İngilizce sürüm: [README.md](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch07/02_dataset-utilities/README.md) · Komut ve çıktı blokları birebir korunmuştur.

Bu klasör, bir talimat veri kümesi hazırlamak için kullanılabilecek yardımcı kodları içerir.

Ek paket gereksinimlerini şu komutla kurun:

```bash
pip install -r requirements-extra.txt
```




### Yakın Kopyaları Bulmak

`find-near-duplicates.py` fonksiyonu, bir talimat veri kümesindeki kopyaları ve yakın kopyaları belirlemek için kullanılabilir. Örneğin,



```bash
python find-near-duplicates.py --json_file instruction-examples.json
```

```
scikit-learn version: 1.3.1


==================================================
Searching 'instruction' for duplicates ...
==================================================
Duplicate pair found with similarity 0.94:
1. Edit the following sentence to make it more formal.
2. Edit the sentence to make it more formal.

Duplicate pair found with similarity 1.00:
1. Name a dwarf planet in our solar system.
2. Name a dwarf planet in our solar system.

Duplicate pair found with similarity 0.91:
1. Change the sentences from active voice to passive voice.
2. Change the sentence from passive to active voice.



==================================================
Searching 'input' for duplicates ...
==================================================
No duplicates found


==================================================
Searching 'output' for duplicates ...
==================================================
Duplicate pair found with similarity 1.00:
1. One dwarf planet in our solar system is Pluto.
2. One dwarf planet in our solar system is Pluto.


```

&nbsp;
Duyarlılığı azaltmak veya artırmak için `--threshold` ayarını 0 ile 1 arasında bir değerle kullanabilirsiniz.
Varsayılan eşik değeri 0.9'dur.



&nbsp;
 ## Edilgen Çatılı (Passive Voice) Girdiler Oluşturmak

 - [create-passive-voice-entries.ipynb](create-passive-voice-entries.ipynb) not defteri, aşağıdaki örnekte gösterildiği gibi bir talimat veri kümesi için "edilgen çatılı" girdiler oluşturmak üzere OpenAI'ın GPT-4 modelini kullanır

 ```python
 {  
    'instruction': 'Identify the verb in the following sentence',
    'input': 'The cat sleeps on the couch.',
    'output': 'The verb in the sentence is "sleeps."',
    'output_2': 'The sentence is "sleeps."'   #  <---- Newly created entry
 }  
 ```
