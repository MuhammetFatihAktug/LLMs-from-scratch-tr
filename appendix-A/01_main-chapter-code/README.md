# Ek A: PyTorch'a Giriş

> 🇹🇷 **Türkçe çeviri.** Orijinal İngilizce sürüm: [README.md](https://github.com/rasbt/LLMs-from-scratch/blob/main/appendix-A/01_main-chapter-code/README.md) · Komutlar birebir korunmuştur.

### Ana Bölüm Kodu

- [code-part1.ipynb](code-part1.ipynb) bölümde geçtiği haliyle A.1'den A.8'e kadar olan tüm kısımların kodunu içerir
- [code-part2.ipynb](code-part2.ipynb) bölümde geçtiği haliyle A.9 kısmındaki tüm GPU kodunu içerir
- [DDP-script.py](DDP-script.py) çoklu GPU kullanımını gösteren betiği içerir (Jupyter Notebook'ların yalnızca tek GPU'yu desteklediğini, bu nedenle bunun bir not defteri değil betik olduğunu unutmayın). `python DDP-script.py` şeklinde çalıştırabilirsiniz. Makinenizde 2'den fazla GPU varsa `CUDA_VISIBLE_DEVIVES=0,1 python DDP-script.py` şeklinde çalıştırın.
- [exercise-solutions.ipynb](exercise-solutions.ipynb) bu bölüme ait alıştırma çözümlerini içerir

### İsteğe Bağlı Kod

- [DDP-script-torchrun.py](DDP-script-torchrun.py), `DDP-script.py` betiğinin isteğe bağlı bir sürümüdür; birden çok süreci `multiprocessing.spawn` ile kendimiz oluşturup yönetmek yerine PyTorch'un `torchrun` komutu aracılığıyla çalışır. `torchrun` komutu, çok düğümlü (multi-node) koordinasyon dahil dağıtık başlatmayı otomatik olarak yönetme avantajına sahiptir; bu da kurulum sürecini biraz basitleştirir. Bu betiği `torchrun --nproc_per_node=2 DDP-script-torchrun.py` şeklinde kullanabilirsiniz
