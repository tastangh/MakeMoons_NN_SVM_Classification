from sklearn.svm import SVC

class SVMModel:
    """
    Destek Vektör Makineleri (SVM) modeli oluşturma sınıfı.

    Bu sınıf, farklı kernel ve hiperparametrelerle SVM modeli oluşturmayı 
    sağlar. Yalnızca modeli oluşturma işlemi için tasarlanmıştır.
    """
    def __init__(self, kernel='linear', C=1.0, degree=3, gamma='scale'):
        """
        SVMModel sınıfının yapıcı metodu.

        Verilen parametrelerle bir SVM modeli tanımlar. Varsayılan değerlerle 
        lineer bir kernel kullanan SVM modeli oluşturulur.

        Args:
            kernel (str): SVM kernel türü. Desteklenen türler: 
                - 'linear', 'poly', 'rbf', 'sigmoid' (default: 'linear').
            C (float): Regularization (düzenleme) parametresi. Daha yüksek 
                değerler daha az hata toleransı sağlar (default: 1.0).
            degree (int): Polinom çekirdek için derece (default: 3). 
                Sadece 'poly' kernel seçildiğinde kullanılır.
            gamma (str or float): RBF ve polinom kernel için gamma değeri.
                'scale' veya 'auto' gibi ölçeklendirme modları da desteklenir (default: 'scale').
        """
        self.kernel = kernel
        self.C = C
        self.degree = degree
        self.gamma = gamma
        self.model = None

    def build_model(self):
        """
        SVM modelini oluşturur ve döndürür.

        Bu metot, sınıfın tanımlı kernel, C, degree ve gamma gibi 
        parametrelerine uygun olarak bir SVM modeli oluşturur. 
        Model, sınıfın `self.model` değişkenine atanır.

        Returns:
            sklearn.svm.SVC: SVM modeli. Model, olasılık tahminlerini 
                etkinleştirmek için `probability=True` parametresi ile oluşturulur.
        """
        self.model = SVC(kernel=self.kernel, C=self.C, degree=self.degree, gamma=self.gamma, probability=True)
        return self.model
