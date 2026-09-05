# AWS CloudFormation Şablonu: LLMs-from-scratch Deposuyla Jupyter Notebook

> 🇹🇷 **Türkçe çeviri.** Orijinal İngilizce sürüm: [README.md](https://github.com/rasbt/LLMs-from-scratch/blob/main/setup/04_optional-aws-sagemaker-notebook/README.md) · Bu CloudFormation şablonu, Amazon SageMaker'da bir yürütme rolü ve LLMs-from-scratch GitHub deposuyla birlikte GPU destekli bir Jupyter not defteri oluşturur.

## Ne yapar:

1. SageMaker not defteri örneği için gerekli izinlere sahip bir IAM rolü oluşturur.
2. Not defteri örneğini şifrelemek için bir KMS anahtarı ve takma ad (alias) oluşturur.
3. Şunları yapan bir not defteri örneği yaşam döngüsü yapılandırma betiği ayarlar:
   - Kullanıcının ev dizinine ayrı bir Miniconda kurulumu yapar.
   - Her ikisi de CUDA destekli olan TensorFlow 2.15.0 ve PyTorch 2.1.0 ile özel bir Python ortamı oluşturur.
   - Jupyter Lab, Matplotlib ve diğer faydalı kütüphaneler gibi ek paketleri kurar.
   - Özel ortamı bir Jupyter çekirdeği (kernel) olarak kaydeder.
4. GPU destekli örnek tipi, yürütme rolü ve varsayılan kod deposu dahil olmak üzere belirtilen yapılandırmayla SageMaker not defteri örneğini oluşturur.

## Nasıl kullanılır:

1. CloudFormation şablon dosyasını (`cloudformation-template.yml`) indirin.
2. AWS Management Console'da CloudFormation servisine gidin.
3. Yeni bir yığın (stack) oluşturun ve şablon dosyasını yükleyin.
4. Not defteri örneği için bir ad belirtin (ör. "LLMsFromScratchNotebook") (varsayılan olarak LLMs-from-scratch GitHub deposunu kullanır).
5. Şablonun parametrelerini gözden geçirip kabul edin, ardından yığını oluşturun.
6. Yığın oluşturma tamamlandığında, SageMaker not defteri örneği SageMaker konsolunda kullanılabilir olacaktır.
7. Not defteri örneğini açın ve LLMs-from-scratch projeleriniz üzerinde çalışmak için önceden yapılandırılmış ortamı kullanmaya başlayın.

## Önemli Noktalar:

- Şablon, 50 GB depolama alanına sahip GPU destekli (`ml.g4dn.xlarge`) bir not defteri örneği oluşturur.
- Her ikisi de CUDA destekli olan TensorFlow 2.15.0 ve PyTorch 2.1.0 ile özel bir Miniconda ortamı kurar.
- Özel ortam bir Jupyter çekirdeği olarak kaydedilir ve böylece not defterinde kullanılabilir hâle gelir.
- Şablon ayrıca not defteri örneğini şifrelemek için bir KMS anahtarı ve gerekli izinlere sahip bir IAM rolü de oluşturur.
