from keras.callbacks import History
import json 

class HistoryMem(History):
    def __init__(self,filepath=None,history=None):
        super().__init__()
        if history != None:
            self.history=history
        self.path = filepath

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        if self.path != None:
            with open(self.path, 'w') as file_pi:
                json.dump(self.history, file_pi)