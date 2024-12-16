from sklearn.svm import SVC

class SVMModel:
    """
    Modüler bir SVM modeli sınıfı.
    Sadece modeli oluşturmak için kullanılır.
    """

    def __init__(self, kernel='linear', C=1.0, degree=3, gamma='scale'):
        """
        SVMModel sınıfını başlatır.

        Args:
        - kernel (str): SVM kernel türü (default: 'linear').
        - C (float): Regularization parametresi (default: 1.0).
        - degree (int): Polinom çekirdek için derece (default: 3).
        - gamma (str or float): RBF veya polinom çekirdek için gamma parametresi (default: 'scale').
        """
        self.kernel = kernel
        self.C = C
        self.degree = degree
        self.gamma = gamma
        self.model = None

    def build_model(self):
        """
        SVM modelini oluşturur ve geri döndürür.

        Returns:
        - model (sklearn.svm.SVC): SVM modeli.
        """
        self.model = SVC(kernel=self.kernel, C=self.C, degree=self.degree, gamma=self.gamma, probability=True)
        return self.model
