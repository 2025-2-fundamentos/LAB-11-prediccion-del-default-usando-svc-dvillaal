import pandas as pd
import numpy as np
import os
import gzip
import pickle
import json

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    balanced_accuracy_score, confusion_matrix
)

def clean_dataset(data):
    temp = data.copy()
    temp.rename(columns={'default payment next month': 'default'}, inplace=True)
    temp.drop(columns=['ID'], inplace=True)
    temp['EDUCATION'].replace(0, np.nan, inplace=True)
    temp['MARRIAGE'].replace(0, np.nan, inplace=True)
    temp.dropna(inplace=True)
    temp.loc[temp['EDUCATION'] > 4, 'EDUCATION'] = 4
    return temp

def separate_xy(data, target_col):
    features = data.drop(columns=[target_col])
    labels = data[target_col]
    return features, labels

def create_pipeline(features):
    cat_cols = ['SEX', 'EDUCATION', 'MARRIAGE']
    num_cols = [c for c in features.columns if c not in cat_cols]

    transformers = ColumnTransformer([
        ('categorical', OneHotEncoder(handle_unknown='ignore'), cat_cols),
        ('numeric', StandardScaler(), num_cols)
    ])

    workflow = Pipeline([
        ('prep', transformers),
        ('dim_red', PCA()),
        ('selector', SelectKBest(score_func=f_classif)),
        ('clf', SVC())
    ])

    return workflow

def grid_search_model(pipeline, X, y):
    params = {
        'dim_red__n_components': [21],
        'selector__k': [12],
        'clf__C': [0.8],
        'clf__kernel': ['rbf'],
        'clf__gamma': [0.1]
    }

    tuner = GridSearchCV(
        pipeline,
        params,
        scoring='balanced_accuracy',
        cv=10,
        n_jobs=-1
    )

    tuner.fit(X, y)
    return tuner

def save_model(model_obj, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with gzip.open(output_path, 'wb') as f:
        pickle.dump(model_obj, f)

def compute_results(model_obj, X_train, y_train, X_test, y_test):
    reports = []
    matrices = []

    for label, feats, real in [('train', X_train, y_train), ('test', X_test, y_test)]:
        preds = model_obj.predict(feats)

        rep = {
            'type': 'metrics',
            'dataset': label,
            'precision': round(precision_score(real, preds), 3),
            'balanced_accuracy': round(balanced_accuracy_score(real, preds), 3),
            'recall': round(recall_score(real, preds), 3),
            'f1_score': round(f1_score(real, preds), 3)
        }

        cm = confusion_matrix(real, preds)

        mat = {
            'type': 'cm_matrix',
            'dataset': label,
            'true_0': {"predicted_0": int(cm[0, 0]), "predicted_1": int(cm[0, 1])},
            'true_1': {"predicted_0": int(cm[1, 0]), "predicted_1": int(cm[1, 1])}
        }

        reports.append(rep)
        matrices.append(mat)

    return reports + matrices

def write_results(data, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    pd.DataFrame(data).to_json(file_path, orient='records', lines=True)

def run():
    train_path = 'files/input/train_data.csv.zip'
    test_path = 'files/input/test_data.csv.zip'
    model_out = 'files/models/model.pkl.gz'
    metrics_out = 'files/output/metrics.json'

    df_train = pd.read_csv(train_path, compression='zip')
    df_test = pd.read_csv(test_path, compression='zip')

    df_train = clean_dataset(df_train)
    df_test = clean_dataset(df_test)

    X_train, y_train = separate_xy(df_train, 'default')
    X_test, y_test = separate_xy(df_test, 'default')

    pipe = create_pipeline(X_train)
    model = grid_search_model(pipe, X_train, y_train)

    save_model(model, model_out)

    results = compute_results(model, X_train, y_train, X_test, y_test)
    write_results(results, metrics_out)

if __name__ == '__main__':
    run()