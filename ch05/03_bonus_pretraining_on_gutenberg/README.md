# GPT'yi Project Gutenberg Veri Kümesi Üzerinde Ön Eğitmek

> 🇹🇷 **Türkçe çeviri.** Orijinal İngilizce sürüm: [README.md](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch05/03_bonus_pretraining_on_gutenberg/README.md) · Komut ve çıktı blokları birebir korunmuştur.

Bu dizindeki kod, Project Gutenberg tarafından sunulan ücretsiz kitaplar üzerinde küçük bir GPT modeli eğitmeye yarayan kodu içerir.

Project Gutenberg web sitesinin belirttiği gibi, "Project Gutenberg e-kitaplarının büyük çoğunluğu ABD'de kamu malıdır (public domain)".

Project Gutenberg tarafından sağlanan kaynakların kullanımı hakkında daha fazla bilgi için lütfen [Project Gutenberg Permissions, Licensing and other Common Requests](https://www.gutenberg.org/policy/permission.html) sayfasını okuyun.

&nbsp;
## Bu Kod Nasıl Kullanılır

&nbsp;

### 1) Veri kümesini indirin

Bu bölümde, [`pgcorpus/gutenberg`](https://github.com/pgcorpus/gutenberg) GitHub deposundaki kodu kullanarak Project Gutenberg'den kitaplar indiriyoruz.

Bu yazının yazıldığı tarihte, bu işlem yaklaşık 50 GB disk alanı gerektirecek ve 10-15 saat sürecektir; ancak Project Gutenberg o zamandan bu yana ne kadar büyüdüyse süre daha da uzayabilir.

&nbsp;
#### Linux ve macOS kullanıcıları için indirme talimatları


Linux ve macOS kullanıcıları veri kümesini indirmek için şu adımları izleyebilir (Windows kullanıcısıysanız lütfen aşağıdaki nota bakın):

1. `gutenberg` deposunu bu klasörün içine yerel olarak klonlamak için `03_bonus_pretraining_on_gutenberg` klasörünü çalışma dizini yapın (sağlanan `prepare_dataset.py` ve `pretraining_simple.py` betiklerini çalıştırmak için bu gereklidir). Örneğin, `LLMs-from-scratch` deposunun klasöründeyken *03_bonus_pretraining_on_gutenberg* klasörüne şu komutla girin:
```bash
cd ch05/03_bonus_pretraining_on_gutenberg
```

2. `gutenberg` deposunu buraya klonlayın:
```bash
git clone https://github.com/pgcorpus/gutenberg.git
```

3. Yerel olarak klonlanan `gutenberg` deposunun klasörüne girin:
```bash
cd gutenberg
```

4. `gutenberg` deposunun klasöründeki *requirements.txt* dosyasında tanımlı gerekli paketleri kurun:
```bash
pip install -r requirements.txt
```

5. Veriyi indirin:
```bash
python get_data.py
```

6. `03_bonus_pretraining_on_gutenberg` klasörüne geri dönün
```bash
cd ..
```

&nbsp;
#### Windows kullanıcıları için özel talimatlar

[`pgcorpus/gutenberg`](https://github.com/pgcorpus/gutenberg) kodu hem Linux hem de macOS ile uyumludur. Ancak Windows kullanıcılarının küçük ayarlamalar yapması gerekir; örneğin `subprocess` çağrılarına `shell=True` eklemek ve `rsync` yerine başka bir çözüm kullanmak gibi.

Alternatif olarak, bu kodu Windows'ta çalıştırmanın daha kolay bir yolu, kullanıcıların Windows içinde Ubuntu kullanarak bir Linux ortamı çalıştırmasına olanak tanıyan "Windows Subsystem for Linux" (WSL) özelliğini kullanmaktır. Daha fazla bilgi için lütfen [Microsoft'un resmî kurulum talimatlarını](https://learn.microsoft.com/en-us/windows/wsl/install) ve [öğreticisini](https://learn.microsoft.com/en-us/training/modules/wsl-introduction/) okuyun.

WSL kullanırken lütfen Python 3'ün kurulu olduğundan emin olun (`python3 --version` ile kontrol edin veya örneğin Python 3.10 için `sudo apt-get install -y python3.10` ile kurun) ve orada şu paketleri kurun:

```bash
sudo apt-get update && \
sudo apt-get upgrade -y && \
sudo apt-get install -y python3-pip && \
sudo apt-get install -y python-is-python3 && \
sudo apt-get install -y rsync
```

> **Not:**
> Python kurulumu ve paket yükleme hakkındaki talimatlar [İsteğe Bağlı Python Kurulum Tercihleri](../../setup/01_optional-python-setup-preferences/README.md) ve [Python Kütüphanelerini Kurmak](../../setup/02_installing-python-libraries/README.md) bölümlerinde bulunabilir.
>
> İsteğe bağlı olarak, bu depoda Ubuntu çalıştıran bir Docker imajı da sunulmaktadır. Sağlanan Docker imajıyla bir konteynerin nasıl çalıştırılacağına dair talimatlar [İsteğe Bağlı Docker Ortamı](../../setup/03_optional-docker-environment/README.md) bölümünde bulunabilir.

&nbsp;
### 2) Veri kümesini hazırlayın

Ardından, (bu yazının yazıldığı tarihte 60.173 olan) metin dosyalarını daha verimli aktarılıp erişilebilmeleri için daha az sayıda ve daha büyük dosyalarda birleştiren `prepare_dataset.py` betiğini çalıştırın:

```bash
python prepare_dataset.py \
  --data_dir gutenberg/data/raw \
  --max_size_mb 500 \
  --output_dir gutenberg_preprocessed
```

```
...
Skipping gutenberg/data/raw/PG29836_raw.txt as it does not contain primarily English text.                                     Skipping gutenberg/data/raw/PG16527_raw.txt as it does not contain primarily English text.                                     100%|██████████████████████████████████████████████████████████| 57250/57250 [25:04<00:00, 38.05it/s]
42 file(s) saved in /Users/sebastian/Developer/LLMs-from-scratch/ch05/03_bonus_pretraining_on_gutenberg/gutenberg_preprocessed
```


> **İpucu:**
> Üretilen dosyaların basitlik adına düz metin biçiminde saklandığını ve önceden token'lara ayrılmadığını unutmayın. Ancak veri kümesini daha sık kullanmayı veya birden fazla dönem (epoch) boyunca eğitmeyi planlıyorsanız, hesaplama süresinden tasarruf etmek için kodları veri kümesini önceden token'lara ayrılmış biçimde saklayacak şekilde güncellemek isteyebilirsiniz. Daha fazla bilgi için bu sayfanın altındaki *Tasarım Kararları ve İyileştirmeler* bölümüne bakın.

> **İpucu:**
> Örneğin 50 MB gibi daha küçük dosya boyutları seçebilirsiniz. Bu, daha fazla dosya oluşmasına yol açar ancak test amacıyla az sayıda dosya üzerinde daha hızlı ön eğitim koşuları yapmak için faydalı olabilir.


&nbsp;
### 3) Ön eğitim betiğini çalıştırın

Ön eğitim betiğini şu şekilde çalıştırabilirsiniz. Ek komut satırı argümanlarının gösterim amacıyla varsayılan değerleriyle verildiğini unutmayın:

```bash
python pretraining_simple.py \
  --data_dir "gutenberg_preprocessed" \
  --n_epochs 1 \
  --batch_size 4 \
  --output_dir model_checkpoints
```

Çıktı şu şekilde biçimlendirilecektir:

> Total files: 3
> Tokenizing file 1 of 3: data_small/combined_1.txt
> Training ...
> Ep 1 (Step 0): Train loss 9.694, Val loss 9.724
> Ep 1 (Step 100): Train loss 6.672, Val loss 6.683
> Ep 1 (Step 200): Train loss 6.543, Val loss 6.434
> Ep 1 (Step 300): Train loss 5.772, Val loss 6.313
> Ep 1 (Step 400): Train loss 5.547, Val loss 6.249
> Ep 1 (Step 500): Train loss 6.182, Val loss 6.155
> Ep 1 (Step 600): Train loss 5.742, Val loss 6.122
> Ep 1 (Step 700): Train loss 6.309, Val loss 5.984
> Ep 1 (Step 800): Train loss 5.435, Val loss 5.975
> Ep 1 (Step 900): Train loss 5.582, Val loss 5.935
> ...
> Ep 1 (Step 31900): Train loss 3.664, Val loss 3.946
> Ep 1 (Step 32000): Train loss 3.493, Val loss 3.939
> Ep 1 (Step 32100): Train loss 3.940, Val loss 3.961
> Saved model_checkpoints/model_pg_32188.pth
> Book processed 3h 46m 55s
> Total time elapsed 3h 46m 55s
> ETA for remaining books: 7h 33m 50s
> Tokenizing file 2 of 3: data_small/combined_2.txt
> Training ...
> Ep 1 (Step 32200): Train loss 2.982, Val loss 4.094
> Ep 1 (Step 32300): Train loss 3.920, Val loss 4.097
> ...


&nbsp;
> **İpucu:**
> Pratikte macOS veya Linux kullanıyorsanız, günlük çıktılarını terminalde yazdırmanın yanı sıra bir `log.txt` dosyasına da kaydetmek için `tee` komutunu kullanmanızı öneririm:

```bash
python -u pretraining_simple.py | tee log.txt
```

&nbsp;
> **Uyarı:**
> `gutenberg_preprocessed` klasöründeki ~500 MB'lık metin dosyalarından biri üzerinde eğitim yapmanın bir V100 GPU'da yaklaşık 4 saat süreceğini unutmayın.
> Klasör 47 dosya içerir ve tamamlanması yaklaşık 200 saat (bir haftadan fazla) sürecektir. Daha az sayıda dosya üzerinde çalıştırmak isteyebilirsiniz.


&nbsp;
## Tasarım Kararları ve İyileştirmeler

Bu kodun eğitim amaçlı olarak işleri basit ve asgari düzeyde tutmaya odaklandığını unutmayın. Modelleme performansını ve eğitim verimliliğini artırmak için kod şu şekillerde iyileştirilebilir:

1. `prepare_dataset.py` betiğini, her kitap dosyasından Gutenberg standart metinlerini (boilerplate) temizleyecek şekilde değiştirin.
2. Veri hazırlama ve yükleme yardımcılarını, veri kümesini önceden token'lara ayırıp token'lanmış biçimde kaydedecek şekilde güncelleyin; böylece ön eğitim betiği her çağrıldığında yeniden token'lara ayrılması gerekmez.
3. `train_model_simple` betiğini, [Ek D: Eğitim Döngüsüne Ek Özellikler Eklemek](../../appendix-D/01_main-chapter-code/appendix-D.ipynb) bölümünde tanıtılan özellikleri (kosinüs sönümleme, doğrusal ısınma ve gradyan kırpma) ekleyerek güncelleyin.
4. Ön eğitim betiğini, optimize edici durumunu kaydedecek şekilde güncelleyin (5. bölümdeki *5.4 Loading and saving weights in PyTorch* kısmına bakın; [ch05.ipynb](../../ch05/01_main-chapter-code/ch05.ipynb)) ve eğitim koşusu yarıda kesilirse mevcut bir model ile optimize edici kontrol noktasını yükleyip eğitime devam etme seçeneğini ekleyin.
5. Kayıp ve doğrulama eğrilerini canlı izlemek için daha gelişmiş bir günlükleyici (örneğin Weights and Biases) ekleyin
6. Dağıtık veri paralelliği (DDP) ekleyip modeli birden çok GPU üzerinde eğitin (Ek A'daki *A.9.3 Training with multiple GPUs* kısmına bakın; [DDP-script.py](../../appendix-A/01_main-chapter-code/DDP-script.py)).
7. `previous_chapter.py` betiğindeki sıfırdan yazılmış `MultiheadAttention` sınıfını, PyTorch'un `nn.functional.scaled_dot_product_attention` fonksiyonu aracılığıyla Flash Attention kullanan ve [Verimli Çok Başlı Dikkat Uygulamaları](../../ch03/02_bonus_efficient-multihead-attention/mha-implementations.ipynb) bonus bölümünde uygulanan verimli `MHAPyTorchScaledDotProduct` sınıfıyla değiştirin.
8. Modeli [torch.compile](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) (`model = torch.compile`) veya [thunder](https://github.com/Lightning-AI/lightning-thunder) (`model = thunder.jit(model)`) ile optimize ederek eğitimi hızlandırın.
9. Ön eğitim sürecini daha da hızlandırmak için Gradient Low-Rank Projection (GaLore) uygulayın. Bu, yalnızca `AdamW` optimize edicisini [GaLore Python kütüphanesinde](https://github.com/jiaweizzhao/GaLore) sunulan `GaLoreAdamW` ile değiştirerek yapılabilir.
