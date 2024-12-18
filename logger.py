import logging
import os

class Logger:
    """
    Train loglama için bir sınıf.

    Bu sınıf, logların hem bir dosyaya hem de konsola yazılmasını sağlar. 
    Loglama işlemi, belirli bir dizinde log dosyasını oluşturur ve log seviyelerine 
    göre mesajları kaydeder.
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
        """
        Bilgi seviyesinde (INFO) log yazar.

        Args:
            message (str): Loglanacak mesaj.
        """
        self.logger.info(message)

    def error(self, message):
        """
        Hata seviyesinde (ERROR) log yazar.

        Args:
            message (str): Loglanacak hata mesajı.
        """
        self.logger.error(message)
