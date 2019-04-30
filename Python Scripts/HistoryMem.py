from keras.callbacks import History
import matplotlib.pyplot as plt
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
				
    def plot_history(self, name):
        # list all data in history
        print(self.history.keys())
        # summarize history for accuracy
        plt.plot(self.history['acc'])
        plt.plot(self.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig("accuracy_history-"+name+".png")
        plt.show()
        # summarize history for loss
        plt.plot(self.history['loss'])
        plt.plot(self.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig("loss_history-"+name+".png")
        plt.show()
        