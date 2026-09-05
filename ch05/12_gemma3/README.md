# Sıfırdan Gemma 3 270M

> 🇹🇷 **Türkçe çeviri.** Orijinal İngilizce sürüm: [README.md](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch05/12_gemma3/README.md) · Tablo verileri birebir korunmuştur.

Bu klasördeki [standalone-gemma3.ipynb](standalone-gemma3.ipynb) Jupyter not defteri, Gemma 3 270M'in sıfırdan bir uygulamasını içerir. Çalıştırmak için yaklaşık 2 GB RAM gerektirir.

Alternatif [standalone-gemma3-plus-kvcache.ipynb](standalone-gemma3-plus-kvcache.ipynb) not defteri, daha iyi çalışma zamanı performansı için bir KV önbelleği ekler (ancak koda daha fazla karmaşıklık katar). KV önbellekleme hakkında daha fazla bilgi için [Understanding and Coding the KV Cache in LLMs from Scratch](https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms) yazıma bakın.

| Model             | Mod               | Donanım         | Token/saniye | GPU Belleği (VRAM) |
| ----------------- | ----------------- | --------------- | ---------- | ----------------- |
| Gemma3Model 270M  | Regular           | Mac Mini M4 CPU | 8          | -                 |
| Gemma3Model 270M  | Regular compiled  | Mac Mini M4 CPU | 9          | -                 |
| Gemma3Model 270M  | KV cache          | Mac Mini M4 CPU | 130        | -                 |
| Gemma3Model 270M  | KV cache compiled | Mac Mini M4 CPU | 224        | -                 |
|                   |                   |                 |            |                   |
| Gemma3Model 270M  | Regular           | Mac Mini M4 GPU | 16         | -                 |
| Gemma3Model 270M  | Regular compiled  | Mac Mini M4 GPU | Error      | -                 |
| Gemma3Model 270M  | KV cache          | Mac Mini M4 GPU | 23         | -                 |
| Gemma3Model 270M  | KV cache compiled | Mac Mini M4 GPU | Error      | -                 |
|                   |                   |                 |            |                   |
| Gemma3Model 270M  | Regular           | Nvidia A100 GPU | 28         | 1.84 GB           |
| Gemma3Model 270M  | Regular compiled  | Nvidia A100 GPU | 128        | 2.12 GB           |
| Gemma3Model 270M  | KV cache          | Nvidia A100 GPU | 26         | 1.77 GB           |
| Gemma3Model 270M  | KV cache compiled | Nvidia A100 GPU | 99         | 2.12 GB           |


Aşağıda, referans model olarak Qwen3 0.6B ile yan yana bir karşılaştırma yer alıyor; Qwen3 0.6B bağımsız not defteriyle ilgileniyorsanız [buradan](../11_qwen3) ulaşabilirsiniz.

<br>

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/gemma3/gemma3-vs-qwen3.webp">

<br>

Mimari farkları hakkında daha fazla bilgi edinmek ve diğer mimarilerle karşılaştırmaları okumak için [The Big LLM Architecture Comparison: From DeepSeek-V3 to Kimi K2: A Look At Modern LLM Architecture Design](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison) yazıma bakın.
