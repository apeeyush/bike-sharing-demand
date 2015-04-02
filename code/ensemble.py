import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv as csv
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import train_test_split
from math import *
from datetime import datetime

validation = False

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
        year_values.append(datetime_object.year-2011)
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
df['casual'] = [log(1+x) for x in df['casual']]
df['registered'] = [log(1+x) for x in df['registered']]

features = ['season','holiday','workingday','weather','temp','atemp','humidity','windspeed','time','weekday','year']
X_train_data = df[features].values
y_train_data = df[['count', 'casual', 'registered']].values
test_data = test_df[features].values

if validation:
    # Validation
    X_train, X_test, y_train, y_test = train_test_split(X_train_data[0::,0::], y_train_data[0::,0::], test_size=0.2, random_state=0)

    ## Casual GBM
    gbm_casual = GradientBoostingRegressor(n_estimators=100, max_depth = 6, random_state = 0)
    gbm_casual.fit(X_train, y_train[0::,1])
    output_gbm_casual = gbm_casual.predict(X_test)
    output_gbm_casual = [int(exp(x)-1) for x in output_gbm_casual]

    ## Resistered GBM
    gbm_registered = GradientBoostingRegressor(n_estimators=100, max_depth = 6, random_state = 0)
    gbm_registered.fit(X_train, y_train[0::,2])
    output_gbm_registered = gbm_registered.predict(X_test)
    output_gbm_registered = [int(exp(x)-1) for x in output_gbm_registered]

    ## Combining GBM output
    output_gbm = [x + y for x, y in zip(output_gbm_casual, output_gbm_registered)]

    ## Casual Random Forest
    rf_casual = RandomForestRegressor(n_estimators=2000, random_state = 0, min_samples_split = 11, oob_score = False, n_jobs = -1)
    rf_casual.fit(X_train, y_train[0::,1])
    output_rf_casual = rf_casual.predict(X_test)
    output_rf_casual = [int(exp(x)-1) for x in output_rf_casual]

    ## Resistered Random Forest
    rf_registered = RandomForestRegressor(n_estimators=2000, random_state = 0, min_samples_split = 11, oob_score = False, n_jobs = -1)
    rf_registered.fit(X_train, y_train[0::,2])
    output_rf_registered = rf_registered.predict(X_test)
    output_rf_registered = [int(exp(x)-1) for x in output_rf_registered]

    ## Combine rf output
    output_rf = [x + y for x, y in zip(output_rf_casual, output_rf_registered)]

    combined_output = [x + y for x, y in zip(output_gbm, output_rf)]
    combined_output[:] = [x/2.0 for x in combined_output]
    output = [log(1+x) for x in combined_output]

    print rmsle(output, y_test[0::,0])
else:
    ## Casual GBM
    gbm_casual = GradientBoostingRegressor(n_estimators=100, max_depth = 6, random_state = 0)
    gbm_casual.fit(X_train_data, y_train_data[0::,1])
    ## Resistered GBM
    gbm_registered = GradientBoostingRegressor(n_estimators=100, max_depth = 6, random_state = 0)
    gbm_registered.fit(X_train_data, y_train_data[0::,2])
    ## Casual Random Forest
    rf_casual = RandomForestRegressor(n_estimators=2000, random_state = 0, min_samples_split = 11, oob_score = False, n_jobs = -1)
    rf_casual.fit(X_train_data, y_train_data[0::,1])
    ## Resistered Random Forest
    rf_registered = RandomForestRegressor(n_estimators=2000, random_state = 0, min_samples_split = 11, oob_score = False, n_jobs = -1)
    rf_registered.fit(X_train_data, y_train_data[0::,2])

    ## GBM prediction
    output_gbm_casual = gbm_casual.predict(test_data)
    output_gbm_casual = [exp(x)-1 for x in output_gbm_casual]
    output_gbm_casual = remove_negative(output_gbm_casual)

    output_gbm_registered = gbm_registered.predict(test_data)
    output_gbm_registered = [exp(x)-1 for x in output_gbm_registered]
    output_gbm_registered = remove_negative(output_gbm_registered)

    output_gbm = [x + y for x, y in zip(output_gbm_casual, output_gbm_registered)]
    ## rf prediction
    output_rf_casual = rf_casual.predict(test_data)
    output_rf_casual = [exp(x)-1 for x in output_rf_casual]
    output_rf_casual = remove_negative(output_rf_casual)

    output_rf_registered = rf_registered.predict(test_data)
    output_rf_registered = [exp(x)-1 for x in output_rf_registered]
    output_rf_registered = remove_negative(output_rf_registered)

    output_rf = [x + y for x, y in zip(output_rf_casual, output_rf_registered)]

    combined_output = [x + y for x, y in zip(output_gbm, output_rf)]
    combined_output[:] = [x/2.0 for x in combined_output]

    # Prepare Kaggle Submission
    datetimes = test_df['datetime'].values
    predictions_file = open("../data/ensemble.csv", "wb")
    open_file_object = csv.writer(predictions_file)
    open_file_object.writerow(["datetime","count"])
    open_file_object.writerows(zip(datetimes, combined_output))
    predictions_file.close()
    print 'Done.'
