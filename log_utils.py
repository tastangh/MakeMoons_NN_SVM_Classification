import sys

class LogEpoch(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch == 0 or epoch == self.params['epochs'] - 1:
            print(f"Epoch {epoch+1}: loss={logs['loss']:.4f}, accuracy={logs['accuracy']:.4f}")

class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    
    def flush(self):
        for f in self.files:
            f.flush()

def redirect_stdout(log_file_path):
    log_file = open(log_file_path, "w")
    sys.stdout = Tee(sys.stdout, log_file)
    return log_file
