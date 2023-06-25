import pandas as pd
import numpy as np
import os

import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import ElasticNet

from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score,accuracy_score,roc_auc_score
from sklearn.model_selection import train_test_split

import argparse

def get_data():

    url="https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv"

    try:
        df=pd.read_csv(url,sep=";")
        return df
    except Exception as e:
        raise e

def evaluate(actual, pred,pred_proba):
    accuracy = accuracy_score(actual, pred)
    rc_score=roc_auc_score(actual,pred_proba,multi_class='ovr')
    return accuracy,rc_score


def main(n_estimators,max_depth):

    df=get_data()
    train,test=train_test_split(df)
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    '''lr=ElasticNet()
    lr.fit(train_x,train_y)
    pred=lr.predict(test_x)
    rmse, mae, r2= evaluate(test_y, pred)'''
    with mlflow.start_run():
        rf=RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth)
        rf.fit(train_x,train_y)
        pred=rf.predict(test_x)

        pred_proba=rf.predict_proba(test_x)

        accuracy,rcscore=evaluate(test_y,pred,pred_proba)


        print("  Accuracy score: %s" % accuracy)

        mlflow.log_param("n_estimators",n_estimators)
        mlflow.log_param("max_depth",max_depth)

        mlflow.log_metric("accuracy",accuracy)
        mlflow.log_metric("rou-auc-score",rcscore)

        mlflow.sklearn.log_model(rf,"random forest model")

if __name__=='__main__':
    args=argparse.ArgumentParser()
    args.add_argument("--n_estimators","-n",default=100,type=int)
    args.add_argument("--max_depth","-m",default=5,type=int)
    parse_args=args.parse_args()
    try:
        main(n_estimators=parse_args.n_estimators,max_depth=parse_args.max_depth)
    except Exception as e:
        raise e