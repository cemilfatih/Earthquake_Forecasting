import glob
import datetime
import time
from math import sqrt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeRegressor
import joblib



# IN CSV column names translate to english
data = pd.read_csv("test.csv",  delimiter=',')
# convert object to timestamp numeric
timestamp = []
for d in data['Date']:
    ts = datetime.datetime.strptime(d, '%Y-%m-%d %H:%M:%S')
    timestamp.append(time.mktime(ts.timetuple()))

timeStamp = pd.Series(timestamp)
data['Timestamp'] = timeStamp.values

data = data[['Timestamp', 'Latitude', 'Longitude', 'Depth', 'Magnitude']]

X  = data.iloc[:,:-1].values
y = data.iloc[:, -1].values

# Random Forest
model = joblib.load('random_forest_model.pkl')
rf_pred = model.predict(X)
print("Random Forest Regressor Prediction:")
print(rf_pred)
print("Real Values:")
print(y)