# Bölüm 3: Gelişmiş Ayrıştırıcı (Bonus Materyal)

> 🇹🇷 **Türkçe çeviri.** Orijinal İngilizce sürüm: [README.md](https://github.com/rasbt/reasoning-from-scratch/blob/main/ch03/03_advanced-parser/README.md) · Kod blokları birebir korunmuştur.

Bu klasör, mevcut bölüm ayrıştırıcısının gözden kaçırabileceği uç durumları (edge case) ele almak üzere hibrit bir LaTeX ayrıştırıcısının önerildiği [issue #133](https://github.com/rasbt/reasoning-from-scratch/issues/133) kaydındaki ayrıştırıcı denemesini içerir.



&nbsp;

## Dosyalar

- [compare_with_current_parser.ipynb](compare_with_current_parser.ipynb): kullanım örnekleri içeren not defteri
- [math500_gpt_answers.json](math500_gpt_answers.json): yukarıdaki not defterinin bir bölümünde kullanılan, LLM yanıtlarıyla birlikte MATH-500 örnekleri
- [gen_llm_answers.py](gen_llm_answers.py): Qwen3 modelinden json biçiminde kutulanmış (boxed) yanıtlar almak için pratik bir betik
- [evaluate_math500_advanced.py](evaluate_math500_advanced.py): 3. bölümdeki LLM değerlendirme betiği [evaluate_math500.py](../02_math500-verifier-scripts/evaluate_math500.py) ile aynıdır, ancak alternatif hibrit ayrıştırıcıyı kullanmak için ek argüman olarak `--hybrid_parser` seçeneğini destekler, örneğin:

```python
uv run evaluate_math500_advanced.py --dataset_size 500 --hybrid_parser
```



&nbsp;
## Bunun Bölüm 3 Ayrıştırıcısından Farkı Nedir
evaluate_math500_advanced.py
[reasoning_from_scratch/ch03.py](../../reasoning_from_scratch/ch03.py) içindeki bölüm ayrıştırıcısı derli toplu ve öğretilebilir kalacak şekilde tasarlanmıştır:

- Hafif normalleştirme ve sembolik denklik kontrollerine odaklanır
- Yanıtları esas olarak aritmetik/sembolik ifadeler olarak ele alır

Bu klasördeki hibrit ayrıştırıcı (`latex_normalizer_hybrid.py`) önce örüntüye (pattern-first) bakar ve daha geniş kapsamlıdır:

- Yedek (fallback) ayrıştırmadan önce yanıt biçimlerini tanır.
- Aralıklar, birleşimler, denklemler, matrisler, küme gösterimi, üyelik (`\\in`) ve `\\pm` desteği ekler
- Taban-alt simge yanıtları (`52_8`) ve metin büyük/küçük harf durumu (`\\text{Evelyn}`) gibi önemli uç durumları daha iyi korur

Davranışın farklılaştığı örnekler:

- `52_8` -> bölüm yolu genellikle `528` olarak çözümler; hibrit `52_8` değerini korur
- `11,\\! 111,\\! 111,\\! 100` -> bölüm yolu bir demete (tuple) dönüşebilir; hibrit `11111111100` olarak normalleştirir
- `(0,9) \\cup (9,36)` -> bölüm yolu genellikle metin olarak kalır; hibrit sembolik bir birleşim döndürür

Ödünleşimler:

- Bölüm ayrıştırıcısı: daha basit, daha hızlı ve yorumlaması daha kolay
- Hibrit ayrıştırıcı: LaTeX uç durumlarında daha iyi kapsama, ancak daha fazla kural ve karmaşıklık; ayrıca SymPy LaTeX arka uç bağımlılıkları ekler

&nbsp;
## Kullanım

Hibrit ayrıştırıcıyı doğrudan paketten içe aktarabilirsiniz:

```python
from reasoning_from_scratch.bonus.parser import normalize_text_hybrid, sympy_parser_hybrid
```

Daha ayrıntılı kullanım örnekleri için bkz. [compare_with_current_parser.ipynb](compare_with_current_parser.ipynb).
