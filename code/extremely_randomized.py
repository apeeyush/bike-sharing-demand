import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv as csv
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.cross_validation import train_test_split
from math import *
from datetime import datetime

def rmsle(predicted, actual):
    error = 0
    for i in range(len(actual)):
        error += pow(actual[i]-predicted[i], 2)
    return sqrt(error/len(actual))

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
    for datetime_value in datetime_values:
        datetime_object = datetime.strptime(datetime_value, '%Y-%m-%d %H:%M:%S')
        time_values.append(datetime_object.hour)
        date_values.append(datetime_object.day)
        month_values.append(datetime_object.month)
        year_values.append(datetime_object.year)
        weekday_values.append(datetime_object.weekday())
        isSunday_values.append(1 if datetime_object.weekday() == 6 else 0)
        time_since_epoch_values.append(int((datetime_object-epoch).total_seconds()/3600))
    df['time'] = time_values
    df['date'] = date_values
    df['month'] = month_values
    df['year'] = year_values
    df['weekday'] = weekday_values
    df['isSunday'] = isSunday_values
    df['time_since_epoch'] = time_since_epoch_values
    return df

df = pd.read_csv('../data/train.csv')
test_df = pd.read_csv('../data/test.csv')

df = transform_data(df)
test_df = transform_data(test_df)

df['count'] = [log(1+x) for x in df['count']]

# # Convert data to categorical
# df['weather'] = df['weather'].astype('category')
# df['holiday'] = df['holiday'].astype('category')
# df['workingday'] = df['workingday'].astype('category')
# df['season'] = df['season'].astype('category')
# df['time'] = df['time'].astype('category')
# print df.dtypes

features = ['season','holiday','workingday','weather','temp','atemp','humidity','windspeed','time','weekday','month', 'year', 'isSunday', 'time_since_epoch']
train_features = features + ['count']
train_data = df[train_features].values
test_data = test_df[features].values

forest = ExtraTreesRegressor(n_estimators=1000, min_samples_split=6)

# Train rf
forest.fit(train_data[0::,0:-1], train_data[0::,-1])
output = forest.predict(test_data[0::,0::])
output = [int(exp(x)-1) for x in output]
# Prepare Kaggle Submission
datetimes = test_df['datetime'].values
predictions_file = open("../data/extremely_randomized_submission.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["datetime","count"])
open_file_object.writerows(zip(datetimes, output))
predictions_file.close()
print 'Done.'

# # Validation
# X_train, X_test, y_train, y_test = train_test_split(train_data[0::,0:-1], train_data[0::,-1], test_size=0.2, random_state=0)
# forest.fit(X_train, y_train)
# output = forest.predict(X_test)
# print rmsle(output, y_test)
# print forest.feature_importances_
