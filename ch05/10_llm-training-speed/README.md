# Daha Hızlı LLM Eğitimi İçin PyTorch Performans İpuçları

> 🇹🇷 **Türkçe çeviri.** Orijinal İngilizce sürüm: [README.md](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch05/10_llm-training-speed/README.md) · Komut, ölçüm ve çıktı blokları birebir korunmuştur.



Kitabın eğitim amaçlı yazıldığını, yani orijinal kodun bilinçli olarak basit tutulduğunu unutmayın. Bu, okunabilirliğe yardımcı olmak ve CPU'lar ile GPU'lar dahil farklı donanımlar arasında uyumluluğu güvence altına almak içindir. Yine de, LLM eğitimini daha performanslı kılacak bazı ileri düzey PyTorch ve GPU özelliklerini merak ediyor olabilirsiniz.

Bu klasör, 5. bölümde tanıtılan LLM ve eğitim fonksiyonu için performans optimizasyonlarını gösteren üç kod dosyası içerir:

1. [`00_orig.py`](00_orig.py): CPU ve tek GPU eğitimi için orijinal 5. bölüm kodu.  
   ➤ Çalıştırma: `python 00_orig.py`

2. [`01_opt_single_gpu.py`](01_opt_single_gpu.py): Tek GPU eğitimi için optimize edilmiş sürüm.  
   ➤ Çalıştırma: `python 01_opt_single_gpu.py`

3. [`02_opt_multi_gpu_ddp.py`](02_opt_multi_gpu_ddp.py): Dağıtık Veri Paralelliği (DDP) kullanan çoklu GPU eğitimi için optimize edilmiş sürüm.  
   ➤ Çalıştırma: `torchrun --nproc_per_node=4 02_opt_multi_gpu_ddp.py`  
   (**Not:** `01_opt_single_gpu.py` dosyasına kıyasla değişiklikleri asgari düzeyde tutmak için bu betik, çoklu işlemeyi yalnızca yukarıda gösterildiği gibi `torchrun` aracılığıyla destekler. Yani çoklu GPU desteği `python 02_opt_multi_gpu_ddp.py` ile **çalışmaz**.)

**Bu değişikliklerin eğitim hızını saniyede 12.525 token'dan (tek A100) saniyede 142.156 token'a (tek A100) ve saniyede 419.259 token'a (4x A100) çıkardığını unutmayın.**

Gelecekte farkları daha ayrıntılı bir yazıda genişletmeyi planlıyorum. Şimdilik, koda hangi iyileştirmelerin eklendiğini görmenin en kolay yolu dosyaları Visual Studio Code'da açıp "Compare Selected" özelliğiyle farklara bakmaktır.

![VS compare](https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/llm-training-speed/vs-code-compare.png)

![PyTorch Tips](https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/pytorch-tips/pytorch-tips.webp?1)


&nbsp;
## Tek GPU hız karşılaştırmaları

Yukarıda belirtildiği gibi, değişiklikleri gelecekte daha ayrıntılı ele almayı planlıyorum. Şimdilik bu bölüm, her değişiklik için token/saniye cinsinden basit bir performans özeti içerir. Tüm deneyler A100 GPU'larda çalıştırılmıştır.

&nbsp;
### Temel çizgi (baseline)

`00_orig.py` dosyasının temel çizgi görevi gördüğünü, kayda değer bir değişiklik içermediğini ve aşağıdakiler dışında 5. bölümdeki kodu olduğu gibi kullandığını unutmayın:

- 4 kat daha büyük bağlam uzunluğu (bu, `00_orig.py` dosyasının 5. bölüme kıyasla nispeten büyük bellek ayak izini açıklar);
- 4 kat yığın boyutu (batch size) değişikliği (`00_orig.py` dosyasının nispeten büyük bellek ayak izine katkıda bulunan bir diğer etken);
- eğitim verisi boyutunu artırmak için kamu malı daha büyük bir kitap.

Hiperparametreler kaybı en aza indirmek ve aşırı öğrenmeyi (overfitting) azaltmak için pek optimize edilmemiştir ve LLM'in en sonda ürettiği metin çok gelişmiş olmayabilir; ancak bu önemli değil, çünkü asıl çıkarım burada hız referansı olarak kullanılan `tok/sec` ölçütüdür (yüksek olan daha iyidir).

```bash
ubuntu@159-13-52-60:~$ python 00_orig.py
PyTorch version: 2.6.0+cu124
Using cuda
CUDA version: 12.4

Ep 1, Step 000000, Train: 9.535, Val: 9.609, Step tok/sec: 7238, Avg tok/sec: 0
Ep 1, Step 000015, Train: 6.201, Val: 6.152, Step tok/sec: 12545, Avg tok/sec: 12545
Ep 1, Step 000030, Train: 5.663, Val: 5.688, Step tok/sec: 12490, Avg tok/sec: 12517
Ep 1, Step 000045, Train: 5.316, Val: 5.362, Step tok/sec: 12541, Avg tok/sec: 12525
Every effort moves you, and's, and I am not be a

...

Ep 15, Step 000735, Train: 0.227, Val: 6.818, Step tok/sec: 11599, Avg tok/sec: 12248
Ep 15, Step 000750, Train: 0.300, Val: 6.895, Step tok/sec: 12530, Avg tok/sec: 12253
Ep 15, Step 000765, Train: 0.150, Val: 6.914, Step tok/sec: 12532, Avg tok/sec: 12259
Every effort moves you like best to think which he held in the room in him, the interest was the night, the realities of the affairs Bulstrode's duty, now!' the fact is another man, conquests

Allocated memory: 2.5069 GB
Reserved memory: 26.2617 GB
```

`01_opt_single_gpu.py` dosyasının aşağıda sırayla listelenen tüm değişiklikleri içerdiğini unutmayın.

Karşılaştırma her zaman, önceki bölümdeki ilk dönemden (epoch) sonraki ortalama tok/sec ve ayrılan bellek değerlerine dayanır.

&nbsp;
### 1. Nedensel maskeyi anlık olarak oluşturmak

- Nedensel (causal) maskeyi kaydetmek yerine, bellek kullanımını azaltmak için anlık olarak oluşturur (burada etkisi çok azdır, ancak 131 bin girdi token'ı destekleyen Llama 3.2 gibi uzun bağlamlı modellerde birikerek anlamlı hâle gelebilir)

Önce:
- `Avg tok/sec: 12525`
- `Reserved memory: 26.2617 GB`

Sonra:
- `Avg tok/sec: 12526`
- `Reserved memory: 26.2422 GB`

&nbsp;
### 2. Tensor çekirdeklerini (tensor cores) kullanmak

- Tensor çekirdeklerini kullanır (yalnızca A100 ve daha yeni Ampere GPU'larda çalışır)

Önce:
- `Avg tok/sec: 12526`
- `Reserved memory: 26.2422 GB`

Sonra:
- `Avg tok/sec: 27648`
- `Reserved memory: 26.2422 GB`

&nbsp;
### 3. Kaynaşık (fused) AdamW optimize edicisi

- `fused=True` ayarlayarak `AdamW` için kaynaşık çekirdekleri kullanır

Önce:
- `Avg tok/sec: 27648`
- `Reserved memory: 26.2422 GB`

Sonra:
- `Avg tok/sec: 28399`
- `Reserved memory: 26.2422 GB`

&nbsp;
### 4. Veri yükleyicide sabitlenmiş bellek (pinned memory)

- GPU belleğini önceden ayırmak ve yeniden kullanmak için veri yükleyicilerde `pin_memory=True` kullanır

Önce:
- `Avg tok/sec: 28399`
- `Reserved memory: 26.2422 GB`

Sonra:
- `Avg tok/sec: 28402`
- `Reserved memory: 26.2422 GB`

&nbsp;
### 5. bfloat16 hassasiyetini kullanmak

- 32 bit float'tan 16 bit brain float (bfloat16) hassasiyetine geçer (bu konu hakkında daha fazlası için [buradaki yazıma](https://magazine.sebastianraschka.com/p/the-missing-bits-llama-2-weights) bakın)

Önce:
- `Avg tok/sec: 28402`
- `Reserved memory: 26.2422 GB`

Sonra:
- `Avg tok/sec: 45486`
- `Reserved memory: 13.7871 GB`

&nbsp;
### 6. Sıfırdan yazılan kodu PyTorch sınıflarıyla değiştirmek

- Sıfırdan yazılan LayerNorm ve GeLU uygulamalarını PyTorch'un yerleşik uygulamalarıyla değiştirir

Önce:
- `Avg tok/sec: 45486`
- `Reserved memory: 13.7871 GB`

Sonra:
- `Avg tok/sec: 55256`
- `Reserved memory: 11.5645 GB`

&nbsp;
### 7. FlashAttention kullanmak

- Sıfırdan yazdığımız çok başlı dikkat uygulaması yerine PyTorch'un FlashAttention destekli öz-dikkat fonksiyonunu kullanır.


Önce:
- `Avg tok/sec: 55256`
- `Reserved memory: 11.5645 GB`

Sonra:
- `Avg tok/sec: 91901`
- `Reserved memory: 5.9004 GB`

&nbsp;
### 8. `pytorch.compile` kullanmak

- `torch.compile(model)` kullanır. İlk yinelemelerin, hız kazanmadan önce her zaman yavaş olduğunu unutmayın. `Avg tok/sec` ölçümü ortalama hesabına yalnızca ilk satırı dahil ettiği için, artık 1. dönemin sonundaki `Step tok/sec` değerini kullanıyoruz.


Önce:
- `Avg tok/sec: 91901`
- `Reserved memory: 5.9004 GB`

Sonra:
- `Step tok/sec: 112046`
- `Reserved memory: 6.1875 GB`

<br>

---

**Windows notu**

- Windows'ta derleme zorlu olabilir
- `torch.compile()`, çekirdekleri JIT ile derleyen ve çalışan bir C/C++ araç zinciri gerektiren Inductor'ı kullanır
- CUDA için Inductor ayrıca, topluluk paketi `triton-windows` üzerinden erişilebilen Triton'a da bağlıdır
  - `cl not found` hatası görürseniz, ["C++ workload" ile Visual Studio Build Tools kurun](https://learn.microsoft.com/en-us/cpp/build/vscpp-step-0-installation?view=msvc-170) ve Python'u "x64 Native Tools" komut isteminden çalıştırın
  - CUDA ile `triton not found` hatası görürseniz `triton-windows` kurun (örneğin `uv pip install "triton-windows<3.4"`).
- CPU için bir okur ayrıca şu [Windows için PyTorch Inductor rehberini](https://docs.pytorch.org/tutorials/unstable/inductor_windows.html) izlemeyi önerdi
  - Burada, bir UTF-8 hatasından kaçınmak için Visual Studio 2022 kurulumunda İngilizce dil paketini kurmak önemlidir
  - Ayrıca kodun bir not defteri yerine "Visual Studio 2022 Developer Command Prompt" üzerinden çalıştırılması gerektiğini unutmayın
- Bu kurulum zorlayıcı gelirse derlemeyi atlayabilirsiniz; **derleme isteğe bağlıdır ve tüm kod örnekleri onsuz da sorunsuz çalışır**

---

&nbsp;
### 9. Sözlük dolgusu (vocabulary padding)

- Burada sözlük boyutunu 50.257'den, 64'ün en yakın katı olan 50.304'e hafifçe artırıyoruz. Bu ipucunu bana, aslen Andrej Karpathy'den geldiğini belirten eski meslektaşım Carlos Mocholi önerdi (muhtemelen [bu gönderiden](https://x.com/karpathy/status/1621578354024677377)). Karpathy'nin önerisi, [Bertrand Maher](https://www.linkedin.com/feed/update/urn:li:activity:7309569006057795584?commentUrn=urn%3Ali%3Acomment%3A%28activity%3A7309569006057795584%2C7309754284185669632%29&dashCommentUrn=urn%3Ali%3Afsd_comment%3A%287309754284185669632%2Curn%3Ali%3Aactivity%3A7309569006057795584%29) tarafından aktarıldığı üzere, `torch.compile` hakkında tavsiye veren PyTorch ekibiyle yapılan bir etkileşime dayanıyor. Bu konuda iyi bir kaynak, yığın boyutlarının ve doğrusal katman boyutlarının yaygın olarak belirli değerlerin katları seçildiği [NVIDIA'nın tensör şekilleri kılavuzudur](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html#tensor-core-shape). Ayrıca sözlük dolgusu numarası, NVIDIA'nın Megatron ekibi tarafından uzun zaman önce anlatılmıştı (2019 tarihli [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053) makalesine bakın).

Önce:
- `Step tok/sec: 112046`
- `Reserved memory: 6.1875 GB`

Sonra:
- `Step tok/sec: 127345`
- `Reserved memory: 5.8906 GB`

&nbsp;
### 10. Yığın boyutunu artırmak

- Son olarak, yığın boyutunu GPU'nun desteklediği en büyük 2'nin kuvvetine çıkarıyoruz

Önce:
- `Step tok/sec: 127345`
- `Reserved memory: 5.8906 GB`

Sonra:
- `Step tok/sec: 142156`
- `Reserved memory: 22.5078 GB`


&nbsp;
## Çoklu GPU hız karşılaştırmaları

Artık 1 yerine 4 GPU kullandığımız için bu tamamen adil bir karşılaştırma olmayabilir; ancak eğitim sınırlı GPU belleğiyle darboğaza girmiyorsa kullanılabilecek en hızlı çoklu GPU tekniği olan dağıtık veri paralelliğini kullanmak elbette gözle görülür hızlanmalar sağlayabilir:

Önce (tek GPU):
- `Step tok/sec: 142156`
- `Reserved memory: 22.5078 GB`

Sonra (4 GPU):
- `Step tok/sec: 419259`
- `Reserved memory: 22.7969 GB`
