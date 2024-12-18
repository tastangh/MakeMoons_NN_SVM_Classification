import logging
import os

class Logger:
    """
    Projede loglama işlemlerini yönetmek için Logger sınıfı.
    """
    def __init__(self, save_dir="logs", log_filename="process.log"):
        os.makedirs(save_dir, exist_ok=True)
        log_path = os.path.join(save_dir, log_filename)

        # Dosya içeriğini sıfırlamak için 'w' modunu kullanıyoruz
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s]: %(message)s",
            handlers=[
                logging.FileHandler(log_path, mode='w'),  # Dosyayı sıfırla ve yaz
                logging.StreamHandler()                  # Konsola yaz
            ]
        )
        self.logger = logging.getLogger()

    def info(self, message):
        """Bilgi seviyesinde log yazar."""
        self.logger.info(message)

    def error(self, message):
        """Hata seviyesinde log yazar."""
        self.logger.error(message)
