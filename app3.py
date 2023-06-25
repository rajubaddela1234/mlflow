import pandas as pd
import numpy as np
import os

import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import ElasticNet

from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score,accuracy_score
from sklearn.model_selection import train_test_split

import argparse

def get_data():

    url="https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv"

    try:
        df=pd.read_csv(url,sep=";")
        return df
    except Exception as e:
        raise


def evaluate(actual, pred):
    accuracy = accuracy_score(actual, pred)
    return accuracy


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

    rf=RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth)
    rf.fit(train_x,train_y)
    pred=rf.predict(test_x)
    accuracy=evaluate(test_y,pred)

    print("  Accuracy score: %s" % accuracy)

if __name__=='__main__':
    args=argparse.ArgumentParser()
    args.add_argument("--n_estimators","-n",default=100,type=int)
    args.add_argument("--max_depth","-m",default=5,type=int)
    parse_args=args.parse_args()
    try:
        main(n_estimators=parse_args.n_estimators,max_depth=parse_args.max_depth)
    except Exception as e:
        raise e