# Bölüm 7: Pekiştirmeli Öğrenmede Politika Optimizasyonunu İyileştirmek

> 🇹🇷 **Türkçe çeviri.** Orijinal İngilizce sürüm: [README.md](https://github.com/rasbt/reasoning-from-scratch/blob/main/ch07/03_rlvr_grpo_scripts_advanced/README.md) · Dosya adları ve kaynak referansları birebir korunmuştur.

Bu bölüm, 6. bölüm uygulamasını ek izleme, kararlılaştırma ve ödül modelleme varyantlarıyla genişleten gelişmiş GRPO betiklerini içerir.


&nbsp;
## Betiklere Genel Bakış

&nbsp;
### Ana Betikler

- `7_3_plus_tracking.py` (*7.3 Daha gelişmiş GRPO performans ölçütlerini izlemek*): Ek performans ölçütlerini (avantaj istatistikleri ve entropi) izler
- `7_4_plus_clip_ratio.py` (*7.4 Kırpılmış politika oranlarıyla dizi düzeyinde GRPO'yu kararlı hâle getirmek*): Yukarıdaki gibi, ancak politika gradyanı kaybını kırpılmış politika oranlarıyla hesaplar
- `7_5_plus_kl.py` (*7.5 Modelin ne kadar değiştiğini bir KL terimiyle denetlemek*): Yukarıdaki gibi, ancak bir KL kayıp terimi ekler
- `7_6_plus_format_reward.py` (*7.6 Açık bir biçim ödülü eklemek*): Yukarıdaki gibi, ancak `<think>` token'ları için ek bir biçim ödülü ekler (diğer betiklerden temel farkı, ana bölümde tartışıldığı gibi bu token'lara zaten aşina olduğu için temel model yerine akıl yürütme modeline uygulanmasıdır)

<br>

&nbsp;
### GRPO İpuçları ve Püf Noktaları Bonus Betikleri

GRPO ilk kez Nisan 2024'te yayımlandığından ([DeepSeekMath](https://arxiv.org/abs/2402.03300)) ve Ocak 2025'te popülerleştiğinden ([DeepSeek-R1](https://arxiv.org/abs/2501.12948)) beri literatürde pek çok iyileştirme önerildi. En dikkat çekici olanlardan bazıları aşağıda listelenmiştir:

1. Sıfır gradyan sinyali filtreleme ([DAPO, Yu ve ark., 2025](https://arxiv.org/abs/2503.14476))
2. Etkin örnekleme (active sampling) (DAPO)
3. Token düzeyinde kayıp (DAPO)
4. KL kaybı yok (DAPO ve [Dr. GRPO, Liu ve ark., 2025](https://arxiv.org/abs/2503.20783))
5. Daha yüksek kırpma (clip higher) (DAPO)
6. Kesilmiş önem örneklemesi ([Yao ve ark., 2025](https://fengyao.notion.site/off-policy-rl))
7. Standart sapma normalleştirmesi yok (Dr. GRPO)
8. Alana özgü KL güçleriyle KL ayarı; matematik için sıfır ([DeepSeek V3.2](https://arxiv.org/abs/2512.02556))
9. Yeniden ağırlıklandırılmış KL (DeepSeek V3.2)
10. Politika dışı (off-policy) dizi maskeleme (DeepSeek V3.2)
11. top-p / top-k için örnekleme maskesini korumak (DeepSeek V3.2)
12. Orijinal GRPO avantaj normalleştirmesini korumak (DeepSeek V3.2)
13. Toplamadan önce ödül başına grup bazlı normalleştirme ([GDPO, Liu ve ark., 2026](https://arxiv.org/abs/2601.05242))
14. Dizi düzeyinde önem örneklemesi ve kırpma ([GSPO, Zheng ve ark., 2025](https://arxiv.org/abs/2507.18071))
15. Token güncellemeleri yerine önem örnekleme ağırlıklarını kırpmak ([CISPO, MiniMax ve ark., 2025](https://arxiv.org/abs/2506.13585))

(Ana içerikleri bitirdikten sonra bir gün daha ayrıntılı bir yazı hazırlamayı planlıyorum.)

<br>

Aşağıdaki betikler bu iyileştirmelerin bazılarını uygular:

- `7_7_improvements/olmo3_style.py`: Bu betik, [7_5_plus_kl.py](7_5_plus_kl.py) üzerine [Olmo 3](https://arxiv.org/abs/2512.13961) benzeri 1-7 numaralı iyileştirmeleri uygular

- `7_7_improvements/deepseek_v32_style.py`: Bu betik, [7_5_plus_kl.py](7_5_plus_kl.py) üzerine [DeepSeek-V3.2](https://arxiv.org/abs/2512.02556) benzeri 8-12 numaralı iyileştirmeleri uygular

- `7_7_improvements/gdpo.py`: [7_6_plus_format_reward.py](7_6_plus_format_reward.py) üzerine [GDPO](https://arxiv.org/abs/2601.05242) uygular (GDPO birden çok ödül için bir ayarlama olduğundan)

---

**Not**: `uv` kullanıcısı değilseniz, aşağıdaki örneklerde `uv run ...py` yerine `python ...py` yazın.

---


&nbsp;
