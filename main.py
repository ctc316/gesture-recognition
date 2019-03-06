import os

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from joblib import load

from hmm_model2 import HMM

TRIM_DATA = 200
TRIM_THRESH = 500
PLT_COL = 2
PLT_ROW = 2
PATH_MODELS = "./models/"
PATH_TEST_DATA = "./Test_Set/"

TRIM = [0, 210, 300, 260, 250, 230, 250, 220]

def main():
    kmeans = load(os.path.join(PATH_MODELS, "kmeans.joblib"))
    hmm_models = load(os.path.join(PATH_MODELS, "hmm_models.joblib"))
    gestures = list(hmm_models.keys())
    
    scores_list = [] 
    filesnames = []
    trim_idx = 0
    for filename in os.listdir(PATH_TEST_DATA):
        if filename.startswith("."):
            continue

        filesnames.append(filename)
            
        raw = np.genfromtxt(os.path.join(PATH_TEST_DATA, filename), delimiter='\t')
        raw = raw[TRIM[trim_idx]:]
        trim_idx += 1
        # raw = raw[TRIM_DATA:] if len(raw) > TRIM_THRESH else raw
        # raw = raw[TRIM_DATA:-TRIM_DATA] if len(raw) > TRIM_THRESH else raw[TRIM_DATA:]
        smoothed = savgol_filter(raw[:, 1:], window_length=5, polyorder=2, deriv=0, delta=1, mode="interp")
        quantized = kmeans.predict(smoothed).reshape(-1, 1)
        scores = np.array([-1/hmm_models[g].score(quantized) for g in gestures])
        scores_list.append(scores)
        print(filename.split(".")[0] + ":", " > ".join([gestures[idx] for idx in reversed(np.argsort(scores))]))

        # fig = plt.figure(figsize=[12, 6])
        # fig.suptitle(filename)
        # plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=0.5)

        # plt.subplot(PLT_ROW, PLT_COL, 1)
        # plt.title("Raw (Trimmed)")
        # plt.plot(raw[:, 1:])
        # plt.legend(["Ax", "Ay", "Az", "Wx", "Wy", "Wz"], loc=1)

        # plt.subplot(PLT_ROW, PLT_COL, 2)
        # plt.title("Smoothed")
        # plt.plot(smoothed)
        # plt.legend(["Ax", "Ay", "Az", "Wx", "Wy", "Wz"], loc=1)

        # plt.subplot(PLT_ROW, PLT_COL, 3)
        # plt.title("Quantized by K-Means")
        # plt.plot(quantized)

        # plt.subplot(PLT_ROW, PLT_COL, 4)
        # plt.title("Score (-1/log-likelihood)")
        # plt.bar(gestures, scores)

        # plt.show()
        
    plt.rc('ytick', labelsize=8) 
    plt.rc('xtick', labelsize=8) 
    fig = plt.figure(figsize=[14, 6])
    fig.suptitle("Scores (-1/log-likelihood)")
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=0.5)
    for i in range(len(scores_list)):
        if i == 3:
            print(scores_list[i])
        plt.subplot(2, 4, i + 1)
        plt.title(filesnames[i])
        plt.bar(gestures, scores_list[i])

    plt.show()

if __name__ == '__main__':
    main()