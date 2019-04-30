from HistoryMem import HistoryMem
import json

if __name__ == "__main__":
    file = open(("..\\models\\conv-mfccs\\history "), "r")
    history = HistoryMem(history = json.load(file))
    history.plot_history("conv-mfccs")