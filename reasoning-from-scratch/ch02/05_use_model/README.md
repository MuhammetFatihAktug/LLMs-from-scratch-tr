# Modelle Çıkarım Çalıştırmak ve Sohbet Etmek

> 🇹🇷 **Türkçe çeviri.** Orijinal İngilizce sürüm: [README.md](https://github.com/rasbt/reasoning-from-scratch/blob/main/ch02/05_use_model/README.md) · Komut, yardım metni ve terminal çıktıları birebir korunmuştur.

&nbsp;

<img src="https://sebastianraschka.com/images/reasoning-from-scratch-images/bonus/chat/chat.gif?1" width=600px>

&nbsp;

Bu klasör, 2. bölümde (ve alıştırmalarda) yüklediğimiz modelle metin üretmek için bağımsız örnek betikler içerir:

- `generate_simple.py`: Ana bölümdekine benzer şekilde metin üretir.
- `chat.py`: Yukarıdaki koda benzer, ancak etkileşimli bir sarmalayıcı olarak çalışır; böylece modeli her seferinde belleğe yeniden yüklemek zorunda kalmadan birden çok kez istem verebiliriz.
- `chat_multiturn.py`: Yukarıdakiyle aynı, ancak mesaj geçmişini hatırlayan bir bellek özelliği içerir.



Daha fazla kullanım ayrıntısı aşağıdaki bölümlerde verilmiştir.

&nbsp;
## generate_simple.py

Bu basit fonksiyon, modeli 2. bölümde anlatıldığı gibi yükler ve 2. bölüm alıştırmalarındaki `generate_text_simple_cache_stream` fonksiyonunu kullanır. Fonksiyonu şu şekilde kullanabilirsiniz (`uv` kullanmıyorsanız `uv run` yerine `python` yazın):

```bash
uv run ch02/05_use_model/generate_simple.py
Using Apple Silicon GPU (MPS)
✓ qwen3/qwen3-0.6B-base.pth already up-to-date

============================================================
torch     : 2.7.1
device    : mps
cache     : True
compile   : False
reasoning : False
============================================================

 Large language models are artificial intelligence systems that can understand, generate, and process human language, enabling them to perform a wide range of tasks, from answering questions to writing essays.

Time: 1.52 sec
22 tokens/sec
```

Temel veya akıl yürütme varyantıyla farklı istemleri hızlıca denemek istiyorsanız bu fonksiyon faydalıdır. Ek seçenekler aşağıda listelenmiştir:

```bash
usage: generate_simple.py [-h] [--device DEVICE]
                          [--max_new_tokens MAX_NEW_TOKENS] [--compile]
                          [--reasoning] [--prompt PROMPT]

Run Qwen3 text generation

options:
  -h, --help            show this help message and exit
  --device DEVICE       Device to run on (e.g. 'cpu', 'cuda', 'mps'). If not
                        provided, will auto-detect with get_device().
  --max_new_tokens MAX_NEW_TOKENS
                        Maximum number of new tokens to generate (default:
                        2048).
  --compile             Compile PyTorch model (default: False).
  --reasoning           Use reasoning model variant (default: False).
  --prompt PROMPT       Use a custom prompt. If not explicitly provided, uses
                        the following defaults: 'Explain large language models
                        in a single sentence.' for the base model, and 'Find
                        all c in Z_3 such that Z_3[x]/(x^2 + c) is a field.'
                        for the reasoning model.
```

&nbsp;
## chat.py

Yukarıdaki fonksiyona benzer şekilde, bu fonksiyon da temel ve akıl yürütme modellerinde farklı istemleri denemek için faydalıdır.

Ancak önceki fonksiyondan farklı olarak, bu fonksiyon kullanıcıyı etkileşimli bir modda tutar; böylece modelin her seferinde yeniden yüklenmesi gerekmez:

```bash
uv run ch02/05_use_model/chat.py        
Using Apple Silicon GPU (MPS)
✓ qwen3/qwen3-0.6B-base.pth already up-to-date

============================================================
torch     : 2.7.1
device    : mps
cache     : True
compile   : False
reasoning : False
memory    : False
============================================================

Interactive REPL (no memory). Type '\exit' or '\quit' to quit.

>> Explain language models in 1 sentence

------------------------------------------------------------
[User]
Explain language models in 1 sentence

[Model]

Language models are algorithms that analyze and predict the likelihood of future words in a text based on the words already seen, enabling them to generate coherent and contextually relevant text.

[Stats]
Time: 1.53 sec
22 tokens/sec
------------------------------------------------------------
>> Explain machine learning in 1 sentence.

------------------------------------------------------------
[User]
Explain machine learning in 1 sentence.

[Model]
 Machine learning is a subset of artificial intelligence that enables computers to learn from data and improve their performance over time without being explicitly programmed.

[Stats]
Time: 1.04 sec
24 tokens/sec
------------------------------------------------------------
```

Ek seçenekler aşağıda listelenmiştir:

```bash
usage: chat.py [-h] [--device DEVICE] [--max_new_tokens MAX_NEW_TOKENS] [--compile]
               [--reasoning]

Run Qwen3 text generation (interactive REPL)

options:
  -h, --help            show this help message and exit
  --device DEVICE       Device to run on (e.g. 'cpu', 'cuda', 'mps'). If not provided,
                        will auto-detect with get_device().
  --max_new_tokens MAX_NEW_TOKENS
                        Maximum number of new tokens to generate (default: 2048).
  --compile             Compile PyTorch model (default: False).
  --reasoning           Use reasoning model variant (default: False).
```



&nbsp;

## chat_multiturn.py

Bu fonksiyon yukarıdakine benzer, ancak LLM'in önceki turlardaki konuşmayı hatırlaması için çok turlu (multi-turn) bir bellek ekler. Temel model sohbetlerde zorlandığı için burada akıl yürütme varyantını kullanmanız şiddetle önerilir:



```bash
uv run ch02/05_use_model/chat_multiturn.py --reasoning
Using Apple Silicon GPU (MPS)
✓ qwen3/qwen3-0.6B-reasoning.pth already up-to-date
✓ qwen3/tokenizer-reasoning.json already up-to-date

============================================================
torch     : 2.7.1
device    : mps
cache     : True
compile   : False
reasoning : True
memory    : True
max_new_tokens (per turn): 2048
context_length: 40960
============================================================

Interactive REPL with memory. Type '\exit' or '\quit' to quit.
Commands: \clear (forget memory), \history (show turn count)

>> What is 1+1 in short?

------------------------------------------------------------
[User]
What is 1+1 in short?

[Model]
<think>
Okay, the user is asking, "What is 1+1 in short?" Let me break this down. First, they want to know the result of adding 1 and 1. In math, 1 plus 1 equals 2. But the question says "in short," which probably means they want a concise answer without the full calculation.

So, the answer is straightforward. 1+1=2. But maybe they want a more concise way to write it? Like, "2" or "2+2"? But "2" is more direct. Let me check if there's any trick here. Sometimes people might think of 1+1 as something else, but no, it's just two ones. 

I should make sure to present the answer clearly. Since the user is asking in a short form, maybe they just want the number 2. So the final answer is 2.
</think>

1+1 equals 2.

[Stats]
Time: 8.27 sec
23 tokens/sec
------------------------------------------------------------
>> What were you just asked?

------------------------------------------------------------
[User]
What were you just asked?

[Model]
<think>
Okay, the user just asked, "What were you just asked?" and I responded with "1+1 equals 2." Now, they're asking again. Let me check if there's any hidden context or if they want more information. Since the previous answer was clear, maybe they want confirmation or a different interpretation. But since the user is asking again, perhaps they want to know if I provided the answer correctly. I should confirm that 1+1 is indeed 2 and that the answer is correct. No further information is needed here. Just a simple confirmation.

[Stats]
Time: 5.21 sec
22 tokens/sec
------------------------------------------------------------
```



Ek seçenekler aşağıda listelenmiştir:

```bash
usage: chat_multiturn.py [-h] [--device DEVICE] [--max_new_tokens MAX_NEW_TOKENS]
                         [--compile] [--reasoning]

Run Qwen3 text generation (interactive REPL)

options:
  -h, --help            show this help message and exit
  --device DEVICE       Device to run on (e.g. 'cpu', 'cuda', 'mps'). If not provided,
                        will auto-detect with get_device().
  --max_new_tokens MAX_NEW_TOKENS
                        Maximum number of new tokens to generate in each turn (default:
                        2048).
  --compile             Compile PyTorch model (default: False).
  --reasoning           Use reasoning model variant (default: False).
```
