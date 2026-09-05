# Bölüm 7: Talimatları İzlemek İçin İnce Ayar

> 🇹🇷 **Türkçe çeviri.** Orijinal İngilizce sürüm: [README.md](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch07/03_model-evaluation/README.md) · Kod blokları birebir korunmuştur.

Bu klasör, model değerlendirmesi için kullanılabilecek yardımcı kodları içerir.



&nbsp;
## Talimat Yanıtlarını OpenAI API ile Değerlendirmek


- [llm-instruction-eval-openai.ipynb](llm-instruction-eval-openai.ipynb) not defteri, talimat ince ayarı yapılmış modellerin ürettiği yanıtları değerlendirmek için OpenAI'ın GPT-4 modelini kullanır. Aşağıdaki biçimdeki bir JSON dosyasıyla çalışır:

```python
{
    "instruction": "What is the atomic number of helium?",
    "input": "",
    "output": "The atomic number of helium is 2.",               # <-- The target given in the test set
    "model 1 response": "\nThe atomic number of helium is 2.0.", # <-- Response by an LLM
    "model 2 response": "\nThe atomic number of helium is 3."    # <-- Response by a 2nd LLM
},
```

&nbsp;
## Talimat Yanıtlarını Ollama ile Yerelde Değerlendirmek

- [llm-instruction-eval-ollama.ipynb](llm-instruction-eval-ollama.ipynb) not defteri, yukarıdakine alternatif olarak Ollama aracılığıyla yerelde indirilmiş bir Llama 3 modelini kullanır.
