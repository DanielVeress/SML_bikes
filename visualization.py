import numpy as np
import matplotlib.pyplot as plt


def plot_confusion_matrix(conf_mtx:np.ndarray):
    plt.figure(figsize=(6, 5))
    plt.imshow(conf_mtx, cmap='Blues', interpolation='nearest')
    plt.title('True/False Positive Matrix')
    plt.colorbar()

    # annotations
    classes = ['Negative/Low', 'Positive/High']
    tick_marks = np.arange(len(classes))
    all_pred_count = conf_mtx.sum().sum()
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    for i in range(2):
        for j in range(2):
            color = 'green' if i == j else 'red'
            count = f'{conf_mtx[i, j]}\n{conf_mtx[i,j]/all_pred_count:0.2%}'
            plt.text(j, i, count, ha='center', va='center', color=color, fontsize=12)

    plt.xlabel('Predicted Label/Demand')
    plt.ylabel('True Label/Demand')
    plt.tight_layout()
    plt.show()