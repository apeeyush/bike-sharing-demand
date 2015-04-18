import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv as csv
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import train_test_split
from math import *
from datetime import datetime
import sklearn.cluster as cluster
from sklearn.decomposition import PCA

def split_list(alist, wanted_parts=1):
    length = len(alist)
    return [ alist[i*length // wanted_parts: (i+1)*length // wanted_parts] 
            for i in range(wanted_parts) ]

def rmsle(predicted, actual):
    error = 0
    for i in range(len(actual)):
        error += pow(log(actual[i]+1)-log(predicted[i]+1), 2)
    return sqrt(error/len(actual))

def remove_negative(items):
    newlist = []
    for item in items:
        if item>0:
            newlist.append(item)
        else:
            newlist.append(0)
    return newlist

def getTimeData(df):
    datetime_values = df['datetime'].values
    hour_values = []
    for datetime_value in datetime_values:
        datetime_object = datetime.strptime(datetime_value, '%Y-%m-%d %H:%M:%S')
        hour_values.append(datetime_object.hour)
    df['hour'] = hour_values
    return df

def getMonthData(df):
    datetime_values = df['datetime'].values
    month_values = []
    for datetime_value in datetime_values:
        datetime_object = datetime.strptime(datetime_value, '%Y-%m-%d %H:%M:%S')
        month_values.append(datetime_object.month)
    df['month'] = month_values
    return df

def transform_data(df):
    epoch = datetime.utcfromtimestamp(0)
    datetime_values = df['datetime'].values
    time_values = []
    date_values = []
    month_values = []
    year_values =[]
    weekday_values = []
    isSunday_values = []
    time_since_epoch_values = []
    hour_cluster_values = []
    month_cluster_values = []
    for datetime_value in datetime_values:
        datetime_object = datetime.strptime(datetime_value, '%Y-%m-%d %H:%M:%S')
        time_values.append(datetime_object.hour)
        date_values.append(datetime_object.day)
        month_values.append(datetime_object.month)
        year_values.append(datetime_object.year-2011)
        weekday_values.append(datetime_object.weekday())
        isSunday_values.append(1 if datetime_object.weekday() == 6 else 0)
        time_since_epoch_values.append(int((datetime_object-epoch).total_seconds()/3600))
        hour_cluster_values.append(hour_clusters[datetime_object.hour])
        month_cluster_values.append(month_clusters[datetime_object.month-1])
    df['time'] = time_values
    df['date'] = date_values
    df['month'] = month_values
    df['year'] = year_values
    df['weekday'] = weekday_values
    df['isSunday'] = isSunday_values
    df['time_since_epoch'] = time_since_epoch_values
    df['hourCluster'] = hour_cluster_values
    df['monthCluster'] = month_cluster_values
    return df

if __name__ == '__main__':
    df = pd.read_csv('../data/train.csv')
    test_df = pd.read_csv('../data/test.csv')

    hour_df = getTimeData(df)
    hour_cluster_data = hour_df.groupby(['hour']).agg(lambda x: x.mean())[['count']]
    hour_clust = cluster.KMeans(n_clusters=6)
    hour_clusters = np.array(hour_clust.fit_predict(split_list(hour_cluster_data.iloc[:,0].values,24)))

    month_df = getMonthData(df)
    month_cluster_data = month_df.groupby(['month']).agg(lambda x: x.mean())[['count']]
    month_clust = cluster.KMeans(n_clusters=4)
    month_clusters = np.array(month_clust.fit_predict(split_list(month_cluster_data.iloc[:,0].values,12)))

    df = transform_data(df)
    test_df = transform_data(test_df)

    df['count'] = [log(1+x) for x in df['count']]
    df['casual'] = [log(1+x) for x in df['casual']]
    df['registered'] = [log(1+x) for x in df['registered']]

    features = ['season','holiday','workingday','weather','temp','atemp','humidity','windspeed','time','weekday','year','monthCluster', 'hourCluster', 'isSunday', 'month', 'date']
    X_train_date = df[['date']].values
    X_train_data = df[features].values
    y_train_data = df[['count', 'casual', 'registered']].values
    test_data = test_df[features].values

    pca = PCA(n_components=5)
    pca.fit(X_train_data)
    PCA(copy=True, n_components=2, whiten=False)
    print(pca.explained_variance_ratio_)
