# Bölüm 7: Talimatları İzlemek İçin İnce Ayar

> 🇹🇷 **Türkçe çeviri.** Orijinal İngilizce sürüm: [README.md](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch07/01_main-chapter-code/README.md) · Komut ve çıktı blokları birebir korunmuştur.

### Ana Bölüm Kodu

- [ch07.ipynb](ch07.ipynb) bölümde geçtiği haliyle tüm kodu içerir
- [previous_chapters.py](previous_chapters.py) önceki bölümlerde kodlayıp eğittiğimiz GPT modelini ve bu bölümde yeniden kullandığımız pek çok yardımcı fonksiyonu içeren bir Python modülüdür
- [gpt_download.py](gpt_download.py) önceden eğitilmiş GPT model ağırlıklarını indirmek için yardımcı fonksiyonları içerir
- [exercise-solutions.ipynb](exercise-solutions.ipynb) bu bölüme ait alıştırma çözümlerini içerir


### İsteğe Bağlı Kod

- [load-finetuned-model.ipynb](load-finetuned-model.ipynb) bu bölümde oluşturduğumuz talimat ince ayarlı modeli yüklemek için bağımsız bir Jupyter not defteridir

- [gpt_instruction_finetuning.py](gpt_instruction_finetuning.py) ana bölümde anlatıldığı gibi modele talimat ince ayarı yapan bağımsız bir Python betiğidir (ince ayar kısımlarına odaklanan bir bölüm özeti olarak düşünün)

Kullanım:

```bash
python gpt_instruction_finetuning.py
```

```
matplotlib version: 3.9.0
tiktoken version: 0.7.0
torch version: 2.3.1
tqdm version: 4.66.4
tensorflow version: 2.16.1
--------------------------------------------------
Training set length: 935
Validation set length: 55
Test set length: 110
--------------------------------------------------
Device: cpu
--------------------------------------------------
File already exists and is up-to-date: gpt2/355M/checkpoint
File already exists and is up-to-date: gpt2/355M/encoder.json
File already exists and is up-to-date: gpt2/355M/hparams.json
File already exists and is up-to-date: gpt2/355M/model.ckpt.data-00000-of-00001
File already exists and is up-to-date: gpt2/355M/model.ckpt.index
File already exists and is up-to-date: gpt2/355M/model.ckpt.meta
File already exists and is up-to-date: gpt2/355M/vocab.bpe
Loaded model: gpt2-medium (355M)
--------------------------------------------------
Initial losses
   Training loss: 3.839039182662964
   Validation loss: 3.7619192123413088
Ep 1 (Step 000000): Train loss 2.611, Val loss 2.668
Ep 1 (Step 000005): Train loss 1.161, Val loss 1.131
Ep 1 (Step 000010): Train loss 0.939, Val loss 0.973
...
Training completed in 15.66 minutes.
Plot saved as loss-plot-standalone.pdf
--------------------------------------------------
Generating responses
100%|█████████████████████████████████████████████████████████| 110/110 [06:57<00:00,  3.80s/it]
Responses saved as instruction-data-with-response-standalone.json
Model saved as gpt2-medium355M-sft-standalone.pth
```

- [ollama_evaluate.py](ollama_evaluate.py) ana bölümde anlatıldığı gibi ince ayarlı modelin yanıtlarını değerlendiren bağımsız bir Python betiğidir (değerlendirme kısımlarına odaklanan bir bölüm özeti olarak düşünün)

Kullanım:

```bash
python ollama_evaluate.py --file_path instruction-data-with-response-standalone.json
```

```
Ollama running: True
Scoring entries: 100%|███████████████████████████████████████| 110/110 [01:08<00:00,  1.62it/s]
Number of scores: 110 of 110
Average score: 51.75
```

- [exercise_experiments.py](exercise_experiments.py) alıştırma çözümlerini uygulayan isteğe bağlı bir betiktir; daha fazla ayrıntı için bkz. [exercise-solutions.ipynb](exercise-solutions.ipynb)
