from tensorflow.keras.callbacks import Callback

class EpochLogger(Callback):
    """
    Sadece ilk ve son epoch'taki accuracy ve loss değerlerini loglar.
    """
    def __init__(self, logger):
        super().__init__()
        self.logger = logger

    def on_epoch_end(self, epoch, logs=None):
        """
        İlk epoch ve son epoch sonunda çalışır.
        """
        total_epochs = self.params['epochs']
        if epoch == 0 or epoch == total_epochs - 1:  # İlk epoch veya son epoch
            self.logger.info(
                f"Epoch {epoch+1}/{total_epochs}: "
                f"loss: {logs['loss']:.4f}, accuracy: {logs['accuracy']:.4f}, "
                f"val_loss: {logs['val_loss']:.4f}, val_accuracy: {logs['val_accuracy']:.4f}"
            )
