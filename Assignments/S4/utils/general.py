import yaml
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix



def load_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.load(f)

def create_save_dir(dir_path, exist_ok=False):
    Path(dir_path).mkdir(exist_ok=exist_ok, parents=True)
    return dir_path

def plot_confusion_matrix(y_true, y_pred, normalize="true"):

    data = confusion_matrix(y_true, y_pred, normalize=normalize)
    df_cm = pd.DataFrame(data, columns=np.unique(y_true), index=np.unique(y_true))
    df_cm.index.name = "Actual"
    df_cm.columns.name = "Predicted"
    plt.figure(figsize=(10, 7))
    sn.set(font_scale=1.4)  # for label size
    sn.heatmap(df_cm, cmap="Blues", annot=True, annot_kws={"size": 16})  # font size
