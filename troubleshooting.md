# Sorun Giderme Rehberi

> 🇹🇷 **Türkçe çeviri.** Orijinal İngilizce sürüm: [troubleshooting.md](https://github.com/rasbt/LLMs-from-scratch/blob/main/troubleshooting.md) · Komut blokları birebir korunmuştur.

Bu sayfa, kitap boyunca ilerlerken karşılaşılan yaygın sorunları ve kurulum ipuçlarını bir araya toplar.

&nbsp;
## Not Defteri Görsel Yükleme Sorunları

Bölüm not defterleri, `https://sebastianraschka.com/images/LLMs-from-scratch-images/...` adresinde barındırılan Markdown görsel bağlantılarını kullanır. Bu, depo indirme boyutunu makul tutar; ancak görsellerin görsel sunucusuna ve internet bağlantınıza bağlı olduğu anlamına da gelir.

`.ipynb` not defterlerindeki görseller görüntülenmiyorsa:

- Görsel URL'lerinden birini doğrudan tarayıcınızda açın, örneğin [https://sebastianraschka.com/images/LLMs-from-scratch-images/ch02_compressed/02.webp](https://sebastianraschka.com/images/LLMs-from-scratch-images/ch02_compressed/02.webp).
- URL tarayıcıda da açılmıyorsa, sorun büyük olasılıkla not defteriyle değil; geçici bir web sitesi, DNS, VPN, proxy, güvenlik duvarı veya yerel ağ sorunuyla ilgilidir.
- URL'yi farklı bir cihazda veya ağda bir kez daha kontrol etmenizi öneririm (ör. görseli telefonunuzda açmayı deneyin); görsel telefonunuzda sorunsuz yükleniyorsa, bu büyük olasılıkla bilgisayarınızdaki bir VPN veya güvenlik duvarı sorununa işaret eder.
- Görseller telefonunuzda da yüklenmiyorsa, sorunu daha ayrıntılı incelememe yardımcı olmak için lütfen GitHub'da bir [Issue](https://github.com/rasbt/LLMs-from-scratch/issues) açmaktan çekinmeyin.

&nbsp;
## Depoyu Güncellerken Kişisel Not Defteri Değişikliklerinizi Korumak

Not defterlerini değiştirirken aynı zamanda depo güncellemelerini de almak istiyorsanız, önce depoyu çatallayın (fork) ve ardından kendi çatalınızı klonlayın. Kitabın ana not defterleri basılı kitapla eşzamanlı tutulur ve kritik düzeltmeler dışında genellikle değiştirilmez. Depo güncellemelerinin çoğu, bunun yerine bonus materyal ekler.

Not defteri dosyaları JSON dosyalarıdır; bu nedenle Git farklarını (diff) ve birleştirme çakışmalarını okumak zor olabilir. Gereksiz çakışmalardan kaçınmak için denemelerinizi, izlenen kitap not defterlerinden ayrı tutmanızı öneririm:

- Bir not defterini değiştirmeden önce kopyalayın, örneğin `ch02.ipynb` dosyasından `ch02_experiments.ipynb` oluşturun.
- Taslak not defterlerinizi ayrı bir klasörde veya kendi dalınızda (branch) tutun.
- Orijinal depodan güncellemeleri bir `upstream` uzak deposu (remote) ile çekin; ardından yalnızca bu güncellemelere ihtiyaç duyduğunuzda birleştirin (merge) veya yeniden temellendirin (rebase).

Bir çatal oluşturmak ve klonlamak için:

1. [https://github.com/rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) adresini açın.
2. GitHub'da sağ üst köşedeki **Fork** düğmesine tıklayın.
3. Çatalınızı klonlayın; `YOUR-USERNAME` yerine kendi GitHub kullanıcı adınızı yazın:

```bash
git clone https://github.com/YOUR-USERNAME/LLMs-from-scratch.git
cd LLMs-from-scratch
```

Ardından, ileride güncellemeleri çekebilmek için orijinal depoyu `upstream` olarak ekleyin:

```bash
git remote add upstream https://github.com/rasbt/LLMs-from-scratch.git
git fetch upstream
git merge upstream/main
```

Düzenlenmiş not defterlerini birleştirmeniz gerekiyorsa, not defterine duyarlı fark ve birleştirme araçları için [`nbdime`](https://nbdime.readthedocs.io/) kurmayı değerlendirin:

```bash
pip install nbdime
nbdime config-git --enable
```

Daha fazla bağlam için bkz. [#1015](https://github.com/rasbt/LLMs-from-scratch/issues/1015).

&nbsp;
## Apple Silicon ve MPS Desteği

Bazı not defterleri ve betikler, mevcut olduğunda `cuda` kullanır ve aksi hâlde `cpu` seçeneğine geri döner; Apple'ın `mps` arka ucunu seçmez. `mps` desteğinin bu şekilde dışarıda bırakılması pek çok yerde bilinçlidir; çünkü önceki PyTorch/MPS sürümleri, özellikle eğitim ve ince ayar sırasında birkaç örnekte kararsız veya farklı sonuçlar üretmiştir.

Apple Silicon bir Mac kullanıyor ve birbirinden uzaklaşan kayıp değerleri, ani kayıp sıçramaları, kötü üretilmiş metin ya da kitapla uyuşmayan sonuçlar görüyorsanız, örneği önce `cpu` üzerinde yeniden çalıştırın. Kitapla uyumlu davranışla daha hızlı eğitim için, yerel bir NVIDIA GPU'da veya bir bulut GPU'sunda `cuda` kullanmanızı öneririm.

Daha yeni PyTorch sürümleri MPS davranışını iyileştirebilir; sonuçları dikkatlice doğruladığınız sürece yerelde `mps` ile deneme yapabilirsiniz. Ancak bir betiğe `mps` desteğini kendiniz eklerseniz, `pin_memory=True`, `torch.compile` ve DDP/çoklu GPU kodu gibi CUDA'ya özgü seçeneklerin ayrı koruma koşulları (guard) gerektirebileceğini unutmayın.

Daha fazla bağlam için bkz. [#977](https://github.com/rasbt/LLMs-from-scratch/issues/977), [#625](https://github.com/rasbt/LLMs-from-scratch/discussions/625), [#644](https://github.com/rasbt/LLMs-from-scratch/discussions/644), [#442](https://github.com/rasbt/LLMs-from-scratch/discussions/442) ve [#846](https://github.com/rasbt/LLMs-from-scratch/issues/846).

&nbsp;
## Diğer Sorunlar

Diğer sorunlar için lütfen GitHub'da yeni bir [Issue](https://github.com/rasbt/LLMs-from-scratch/issues) açmaktan çekinmeyin.
