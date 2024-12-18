# **MakeMoons_NN_SVM_Classification**

## **Açıklama**
Bu proje, Makine Öğrenmesi (BLM5110) dersi kapsamında, `make_moons` kullanılarak iki sınıflı bir sınıflandırma  üzerine Yapay Sinir Ağları (ANN) ve Destek Vektör Makineleri (SVM) modelleri ile çalışmaktadır.

Bu projede, `make_moons` ile 400 veriden oluşturulan bir veri kümesi kullanılmıştır.Çalışmanın temel adımları şunlardır:
1. **Veri Hazırlığı**:
    - Veri kümesi oluşturulmuş ve %60 eğitim, %20 doğrulama, %20 test oranında bölünmektedir.
    - Veri görselleştirilerek, sınıf dağılımı analiz edilmektedir.

2. **Model Eğitimi**:
    - **ANN Modelleri**: TensorFlow kullanılarak çok katmanlı yapay sinir ağları eğitilmektedir. Öğrenme oranı, epoch sayısı, optimizasyon algoritması (SGD, BGD, MBGD) ve gizli katman sayısı gibi hiperparametreler test edilmektedir.
    - **SVM Modelleri**: scikit-learn kullanılarak lineer, polinom ve RBF çekirdek fonksiyonları ile SVM modelleri eğitilmektedir. `C`, `degree` ve `gamma` gibi parametreler optimize edilmektedir.

3. **Performans Analizi**:
    - Eğitim, doğrulama ve test setlerinde accuracy, precision, recall ve F1 skoru hesaplanmaktadır.
    - Doğrulama ve test setlerinde Modellerin karar sınırları görselleştirilmektedir.
    - Eğitim ve doğrulama kayıp grafikleri görselleştirilerek overfitting ve underfitting analizleri yapılmaktadır.
    - Eğitim,doğrulama ve test setleri için confusion matrix'ler görselleştirilerek modellerin hangi sınıfları doğru tahmin ettiğini ve hata yaptığı alanları incelenmektedir.

4. **Eğitim İzlenilebilirliği**:
    - ANN tensorflow verbose parametresi aktif edildiğine izlenebilirlik düştüğü için 0 değerine atanmaktadır. Bunun yerine hem terminalden hem de dosya üzerinden daha iyi bir log yapılması için logger yapısı geliştirilmiş ve sadece eğitimin ilk epoch'u ve son epoch'u gözlenebilir hale getirilmektedir.(Gerekirse verbose parametresi 1 e atanarak tüm epochlar gözlemlenebilir.)
    - Epoch bazında metrikler detaylı şekilde log dosyasına kaydedilmektedir.

## **Gereksinimler**
Bu projede kullanılan Python paketleri:
- numpy
- pandas
- scikit-learn
- matplotlib
- tensorflow
- seaborn

Gerekli paketleri aşağıdaki komutla yükleyebilirsiniz:
```
pip install -r requirements.txt
```
## Çalıştırma
1- Model eğitimi ve eğitim çıktılarını oluşturma
```
python train.py
```
2- Eğitime sokulan bir modeli değerlendirme ve çıktılarını oluşturma
```
python eval.py
```

### Dosya Düzeni
/MakeMoons_NN_SVM_Classification
    ├── ann_model.py              # ANN modeli
    ├── dataset.py                # Veri kümesinin oluşturulması ve bölünmesi
    ├── eval.py                   # Modellerin test edilmesi ve değerlendirilmesi
    ├── logger.py                 # Eğitim loglama işlemleri
    ├── epoch_logger.py           # Her epoch'un ilk ve son metriklerini loglama
    ├── metrics.py                # Performans metriklerinin hesaplanması
    ├── svm_model.py              # SVM modeli
    ├── train.py                  # Modellerin eğitimi
    ├── visualize.py              # Görselleştirme (sınıf dağılımı, karar sınırları vb.)
    ├── requirements.txt          # Gerekli kütüphaneler
    ├── README.md                 # Proje açıklaması
    ├── dataset/                  # Veri kümesi ve görselleştirilmeleri
    ├── train_results/            # Eğitim çıktıları ve görseller
        ├── models                # Eğitilen modeller
        ├── plots                 # Eğitim Görselleri (Kayıp Grafikleri,Confusion matrixleri ve KararSınırı Görselleri)
        |── combined_metrics.txt  # Eğitim Metrikleri (Model ve hiperparemetre bazlı train ve val sonuçları tablosu)
        |── train.log             # Eğitim Logları
    ├── evaluation_results/       # Değerlendirme çıktıları
        ├── plots                 # Değerlendirme Görselleri (Confusion matrixleri ve Karar Sınırı Görselleri)
        |── combined_metrics.txt  # Değerlendirme Metrikleri (Model ve hiperparemetre bazlı test sonuçları tablosu)

