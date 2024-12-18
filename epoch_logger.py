from tensorflow.keras.callbacks import Callback

class EpochLogger(Callback):
    """
    Sadece ilk ve son epoch'taki doğruluk (accuracy) ve kayıp (loss) değerlerini loglamak için bir Callback sınıfı.
    """
    def __init__(self, logger):
        """
        EpochLogger sınıfının yapıcı fonksiyonu.

        Args:
            logger (logging.Logger): Loglama işlemleri için kullanılan logger nesnesi.
        """
        super().__init__()
        self.logger = logger

    def on_epoch_end(self, epoch, logs=None):
        """
        Epoch'ların sonunda çağrılan metot.

        Bu metot, yalnızca ilk ve son epoch'ların sonunda, doğruluk (accuracy) ve kayıp (loss) 
        değerlerini loglamak için çalışır. Bu değerler hem eğitim hem de doğrulama seti için kaydedilir.

        Args:
            epoch (int): Mevcut epoch'un sırası (0 tabanlı indeks).
            logs (dict): Epoch sırasında hesaplanan metrik değerlerini içeren sözlük.
                - loss (float): Eğitim kayıp değeri.
                - accuracy (float): Eğitim doğruluk değeri.
                - val_loss (float): Doğrulama kayıp değeri.
                - val_accuracy (float): Doğrulama doğruluk değeri.
        """
        total_epochs = self.params['epochs']
        if epoch == 0 or epoch == total_epochs - 1:  # İlk epoch veya son epoch ayrımının yapıldığı yer!
            self.logger.info(
                f"Epoch {epoch+1}/{total_epochs}: "
                f"loss: {logs['loss']:.4f}, accuracy: {logs['accuracy']:.4f}, "
                f"val_loss: {logs['val_loss']:.4f}, val_accuracy: {logs['val_accuracy']:.4f}"
            )
