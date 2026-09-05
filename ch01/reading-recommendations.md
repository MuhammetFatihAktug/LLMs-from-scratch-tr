# Teknik Bir Kitaptan En İyi Şekilde Yararlanmak İçin Öneriler

> 🇹🇷 **Türkçe çeviri.** Orijinal İngilizce sürüm: [reading-recommendations.md](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch01/reading-recommendations.md) · Aşağıda, okurlar sıfırdan büyük dil modeli geliştirme kitap(lar)ımdan en iyi şekilde nasıl yararlanacaklarını sorduğunda daha önce paylaştığım birkaç not yer alıyor.


Ben de teknik kitaplar okurken benzer bir yaklaşım izliyorum. Bu, evrensel bir reçete olarak sunulmuyor; ancak faydalı bir başlangıç noktası olabilir.

Bu kitap özelinde, her bölüm bir öncekine dayandığı için kitabı sırayla okumanızı şiddetle öneriyorum. Ve her bölüm için aşağıdaki adımları tavsiye ediyorum.

&nbsp;
### 1) İlk okuma (bilgisayarsız)

Bölümü henüz hiç kod yazmadan baştan sona okumanızı öneririm.
Bu ilk okumanın amacı, önce büyük resmi kavramaktır.

İdeal olarak bölümü bilgisayardan uzakta okumanızı öneririm. Basılı bir nüsha iyi iş görür, ancak dikkat dağıtıcı unsurlar olmayan (tarayıcı, sosyal medya veya e-posta içermeyen) dijital bir cihaz da uygundur.

Ben hem kâğıttan hem de bir e-mürekkep tabletten okuyorum. 2018'den beri e-mürekkep tablet kullanıyor ve her zaman daha çok e-mürekkepten okumaya çalışıyor olsam da, basılı nüshaların odaklanmama daha çok yardımcı olduğunu hâlâ fark ediyorum. Zorlayıcı bulduğum ya da gerçekten ayrıntılı anlamak istediğim araştırma makalelerini bazen bu yüzden çıktı alıyorum.

Önerim, ilk okumayı dikkat dağıtıcı unsurları en aza indirdiğiniz, aşırı düşünmeden ve ayrıntılara takılmadan geçirdiğiniz kısa ve odaklı 20 dakikalık bir okuma seansı hâline getirmenizdir.

Kafa karıştırıcı veya ilginç kısımların altını çizmek ya da not almak sorun değil; ancak bu aşamada hiçbir şeyi araştırmazdım. Sadece okumanızı, henüz hiç kod çalıştırmamanızı öneriyorum. Bu ilk geçiş, büyük resmi anlamaya yöneliktir.

&nbsp;
### 2) İkinci okuma (kodla birlikte)

İkinci okumada, bölümdeki kodu yazıp çalıştırmanızı öneririm. Kodu kopyalamak cazip gelir; çünkü yeniden yazmak epey iş demektir. Ancak başka teknik kitaplar okurken bu, kod üzerinde biraz daha fazla düşünmeme (sadece göz gezdirmek yerine) genellikle yardımcı oluyor.

Kitaptakinden farklı sonuçlar alırsam, kitabın GitHub deposunu kontrol eder ve oradaki kodu denerdim. Yine farklı sonuçlar alıyorsam, bunun farklı paket sürümlerinden, rastgele tohum (seed) değerlerinden, CPU/CUDA farkından vb. kaynaklanıp kaynaklanmadığına bakardım. Bunu da çözemezsem, yazara sormak fena bir fikir olmaz (kitap forumu, açık GitHub deposundaki issue veya tartışmalar üzerinden; son çare olarak e-posta).

&nbsp;
### 3) Alıştırmalar

İkinci okumadan, kodu yeniden yazıp çalıştırmaktan sonra genellikle alıştırmaları denemek için iyi bir zamandır. Anlayışı pekiştirmek ya da bir problemle yarı yapılandırılmış bir şekilde uğraşmak için harikadır. Alıştırma çok zorlayıcıysa çözüme bakmak sorun değil. Yine de önce ciddi bir deneme yapmanızı öneririm.

&nbsp;
### 4) Notları gözden geçirin ve daha derine inin

Şimdi, bölümü okuduktan, kodu çalıştırdıktan ve alıştırmaları yaptıktan sonra, önceki iki okumadaki altını çizdiğiniz yerlere ve notlarınıza geri dönüp hâlâ belirsiz kalan bir şey olup olmadığına bakmanızı öneririm.

Bu aynı zamanda ek kaynaklara bakmak ya da hâlâ çözülmemiş hissettiren şeyleri netleştirmek için hızlı bir arama yapmak için de iyi bir zamandır. Ancak her şey anlaşılmış olsa bile, ilgi duyduğunuz bir konu hakkında daha fazla okumak fena bir fikir değildir.

Bu aşamada, faydalı içgörüleri, kod parçalarını vb. favori not alma uygulamanıza yazmak veya aktarmak da mantıklıdır.

&nbsp;
### 5) Fikirleri bir projede kullanın

Önceki adımların hepsi bilgiyi özümsemekle ilgiliydi. Şimdi, bir bölümün belirli yönlerini kendi projenizde kullanıp kullanamayacağınıza bakın. Ya da kitaptaki kodu başlangıç noktası alarak küçük bir proje geliştirin. İlham için bonus materyallere göz atın; bunlar temelde kendi merakımı gidermek için yaptığım mini projelerdir.

Örneğin, çok başlı dikkat mekanizmalarını okuyup LLM'i uyguladıktan sonra, gruplanmış sorgu dikkatine sahip bir modelin ne kadar iyi performans gösterdiğini ya da RMSNorm ile LayerNorm arasında gerçekte ne kadar fark olduğunu merak edebilirsiniz. Ve bu böyle devam eder.

Kendi projelerinizde işe yarayabilecek daha küçük noktalar da olabilir. Örneğin, bazen faydalı olan minik bir ayrıntıdır; `torch.manual_seed(seed)` tek başına kullanılırken `torch.mps.manual_seed(seed)` çağrısını açıkça yapmanın bir şeyi değiştirip değiştirmediğini test etmek gibi.

Ama sonunda bu bilgiyi bir şekilde kullanmak istiyorum. Bu, bölümün ana kavramını kullanmak olabileceği gibi, bazen yol boyunca öğrendiğim küçük ayrıntılar da olabilir; örneğin, projemde sadece `torch.manual_seed(seed)` yerine `torch.mps.manual_seed(seed)` çağrısını açıkça yapmanın gerçekten bir fark yaratıp yaratmadığı gibi basit şeyler.

&nbsp;
### Ek düşünceler

Elbette yukarıdakilerin hiçbiri değişmez kurallar değil. Konu genel olarak çok tanıdık veya kolaysa ve kitabı esas olarak sonraki bölümlerdeki bilgiler için okuyorsam, bir bölüme göz gezdirmek sorun değil (zamanımı boşa harcamamak için).

Ayrıca, hiç kod içermeyen bölümlerde (örneğin giriş niteliğindeki 1. bölüm) kodla ilgili adımları atlamak elbette mantıklıdır.

Her hâlükârda, umarım bu faydalı olmuştur. İyi okumalar ve iyi öğrenmeler!
