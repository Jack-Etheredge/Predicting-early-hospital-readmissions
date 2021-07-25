"""Plotting functions used for EDA and model performance evaluation"""

from sklearn.metrics import confusion_matrix, precision_recall_curve
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def boxplot(results, labels, title='boxplot', figsize=(10,10)):
    """
    This function plots a boxplot from a list of results.
    """
    fig = plt.figure(figsize=figsize)
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(labels)
    plt.show()
    return fig


def precision_recall_plot(y_test, y_pred_proba, figsize=(10,10)):
    """
    This function plots a precision recall curve from predicted and true labels.
    """
    p, r, t = precision_recall_curve(y_test, y_pred_proba[:, 1])

    # adding last threshold of 1. to threshold list
    t = np.concatenate((t, np.array([1.])))
    
    # boxplot algorithm comparison
    fig = plt.figure(figsize=figsize)
    fig.suptitle('Precision Recall Curve')
    ax = fig.add_subplot(111)
    plt.plot(t, p, label="precision")
    plt.plot(t, r, label="recall")
    plt.legend()
    plt.show()

    return fig


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=sns.color_palette("Blues")):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Only use the labels that appear in the data
#     classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    ax = sns.heatmap(cm, cmap = cmap, annot=True, xticklabels=classes, yticklabels=classes)

    ax.set(title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the x labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Vertically center y labels
    plt.setp(ax.get_yticklabels(), va="center")
    
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    
    return ax
