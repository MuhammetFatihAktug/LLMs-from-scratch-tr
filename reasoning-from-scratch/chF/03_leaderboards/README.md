
# Liderlik Tablosu Sıralamaları

> 🇹🇷 **Türkçe çeviri.** Orijinal İngilizce sürüm: [README.md](https://github.com/rasbt/reasoning-from-scratch/blob/main/chF/03_leaderboards/README.md) · Komut ve çıktı blokları birebir korunmuştur.

Bu bonus materyal, ikili karşılaştırmalardan LM Arena (eski adıyla Chatbot Arena) tarzı liderlik tabloları oluşturmanın iki farklı yolunu uygular.

Her iki uygulama da `--path` argümanı aracılığıyla bir json dosyasından ikili tercihler listesi (solda: kazanan, sağda: kaybeden) alır. Sağlanan [votes.json](votes.json) dosyasından bir kesit:

```json
[
  ["GPT-5", "Claude-3"],
  ["GPT-5", "Llama-4"],
  ["Claude-3", "Llama-3"],
  ["Llama-4", "Llama-3"],
  ...
]
```



<br>

---

**Not**: `uv` kullanıcısı değilseniz, aşağıdaki örneklerde `uv run ...py` yerine `python ...py` yazın.

---

&nbsp;
## Yöntem 1: Elo puanları

- LM Arena tarafından başlangıçta kullanılan, popüler Elo puanlama yöntemini (satranç sıralamalarından esinlenmiştir) uygular
- Ayrıntılar için [ana not defterine](../01_main-chapter-code/chF_main.ipynb) bakın

```bash
➜  03_leaderboards git:(main) ✗ uv run 1_elo_leaderboard.py --path votes.json

Leaderboard (Elo) 
-----------------------
 1. GPT-5       1095.9
 2. Claude-3    1058.7
 3. Llama-4      958.2
 4. Llama-3      887.2
```






&nbsp;
## Yöntem 2: Bradley-Terry modeli

- Resmî makalede ([Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference](https://arxiv.org/abs/2403.04132)) açıklanan yeni LM Arena liderlik tablosuna benzer şekilde bir [Bradley-Terry modeli](https://en.wikipedia.org/wiki/Bradley–Terry_model) uygular
- LM Arena liderlik tablosunda olduğu gibi, puanlar orijinal Elo puanlarına benzer olacak şekilde yeniden ölçeklenir
- Buradaki kod, modeli uydurmak için PyTorch'un Adam optimize edicisini kullanır (daha iyi kod tanıdıklığı ve okunabilirliği için)



```bash
➜  03_leaderboards git:(main) ✗ uv run 2_bradley_terry_leaderboard.py --path votes.json 

Leaderboard (Bradley-Terry)
-----------------------------
 1. GPT-5       1140.6
 2. Claude-3    1058.7
 3. Llama-4      950.3
 4. Llama-3      850.4
```
