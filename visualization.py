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


def plot_correlations(df, add_annotations=True):
    corr_mtx = df.corr().round(2)

    plt.figure(figsize=(10, 8))
    plt.imshow(corr_mtx, cmap='coolwarm', interpolation='nearest')
    plt.colorbar(label='Correlation Coefficient')
    plt.title('Correlation Matrix Heatmap')
    plt.xticks(ticks=range(len(corr_mtx.columns)), labels=corr_mtx.columns, rotation=45, ha='right')
    plt.yticks(ticks=range(len(corr_mtx.index)), labels=corr_mtx.index)
    plt.tight_layout()

    # Add annotations
    if add_annotations:
        for i in range(len(corr_mtx.columns)):
            for j in range(len(corr_mtx.columns)):
                plt.text(j, i, corr_mtx.iloc[i, j], ha='center', va='center', color='black')

    plt.show()