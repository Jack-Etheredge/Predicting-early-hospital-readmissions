"""Extracting the differences between the duplicated notebooks"""

from sklearn import naive_bayes
from sklearn.metrics import classification_report
import click
import pandas as pd
import patsy as patsy
import seaborn as sns
import pickle
import numpy as np

import preprocess


@click.command()
@click.option("--smote", default=True, help="Use smote oversampling (True or False)?")
@click.option("--scale", default=True, help="Standard scale features (True or False)?")
@click.option("--remove_expired", default=True, help="Remove expired patients (True or False)?")
@click.option("--remove_duplicates", default=True, help="Remove duplicate (returned) patients (True or False)?")
@click.option("--binary_classification", default=True, help="Reduce classes to binary <30 day vs >30 readmission by combining >30 and NO (True or False)?")
def train(smote, scale, remove_expired, remove_duplicates, binary_classification):
    """
    Train classifiers using preprocessed data from preprocess module
    """

    X_train, X_test, y_train, y_test = preprocess(smote=smote, scale=False, 
                                                  remove_expired=remove_expired, 
                                                  remove_duplicates=remove_duplicates, 
                                                  binary_classification=binary_classification)
    NBmodel = naive_bayes.GaussianNB()
    NBmodel.fit(X_train, y_train)
    y_pred = NBmodel.predict(X_test)

    print("model performance:")
    print(classification_report(y_test, y_pred))

if __name__ == '__main__':
    train()
