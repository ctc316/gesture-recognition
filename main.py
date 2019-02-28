import os

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from joblib import load

from hmm_model import HMM

TRIM_DATA = 230
TRIM_THRESH = 1000
PLT_COL = 2
PLT_ROW = 2
PATH_MODELS = "./models/"
PATH_TEST_DATA = "./Test_Set/"


def main():
    kmeans = load(os.path.join(PATH_MODELS, "kmeans.joblib"))
    hmm_models = load(os.path.join(PATH_MODELS, "hmm_models.joblib"))
    gestures = list(hmm_models.keys())
    
    for filename in os.listdir(PATH_TEST_DATA):
        if filename.startswith("."):
            continue
            
        raw = np.genfromtxt(os.path.join(PATH_TEST_DATA, filename), delimiter='\t')
        
        raw = raw[TRIM_DATA:-TRIM_DATA] if len(raw) > TRIM_THRESH else raw[TRIM_DATA:]
        smoothed = savgol_filter(raw[:, 1:], window_length=5, polyorder=2, deriv=0, delta=1, mode="interp")
        quantized = kmeans.predict(smoothed).reshape(-1, 1)
        scores = np.array([-1/hmm_models[g].score(quantized) for g in gestures])
        
        print(filename.split(".")[0] + ":", " > ".join([gestures[idx] for idx in reversed(np.argsort(scores))]))

        fig = plt.figure(figsize=[12, 6])
        fig.suptitle(filename)
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=0.5)

        plt.subplot(PLT_ROW, PLT_COL, 1)
        plt.title("Raw (Trimmed)")
        plt.plot(raw[:, 1:])
        plt.legend(["Ax", "Ay", "Az", "Wx", "Wy", "Wz"], loc=1)

        plt.subplot(PLT_ROW, PLT_COL, 2)
        plt.title("Smoothed")
        plt.plot(smoothed)
        plt.legend(["Ax", "Ay", "Az", "Wx", "Wy", "Wz"], loc=1)

        plt.subplot(PLT_ROW, PLT_COL, 3)
        plt.title("Quantized by K-Means")
        plt.plot(quantized)

        plt.subplot(PLT_ROW, PLT_COL, 4)
        plt.title("Score (-1/log-likelihood)")
        plt.bar(gestures, scores)

        plt.show()
        


if __name__ == '__main__':
    main()