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
data = pd.read_csv("turkey_1965-2024.csv",  delimiter=',')
print(data.shape)
# convert object to timestamp numeric
timestamp = []
for d in data['time']:
    ts = datetime.datetime.strptime(d, '%Y-%m-%dT%H:%M:%S.%fZ')
    timestamp.append(time.mktime(ts.timetuple()))

timeStamp = pd.Series(timestamp)
data['Timestamp'] = timeStamp.values

data = data[['Timestamp', 'latitude', 'longitude', 'depth', 'mag']]

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

sc = StandardScaler()
x_train_scale = sc.fit_transform(X_train)
sc2 = StandardScaler()
y_train_scale = np.ravel(sc2.fit_transform(y_train.reshape(-1,1)))


# Random Forest
# max depth versus error
md = 20
md_errors = np.zeros(md)
# random forest regression
for i in range(1, md + 1):
    rf_reg = RandomForestRegressor(n_estimators=100, max_depth=i, random_state=0)
    rf_reg.fit(X_train, y_train)
    r_pred = rf_reg.predict(X_test)
    # finding error
    md_errors[i - 1] = sqrt(mean_squared_error(y_test, r_pred))

joblib.dump(rf_reg, 'random_forest_model.pkl')


r_df = pd.DataFrame({'Actual': y_test, 'Predicted': r_pred})
print("Random Forest RMSE ", md_errors[i - 1])
rf_score = r2_score(y_test, r_pred)
print("Random Forest r2 score ", rf_score)
rf_ms = mean_squared_error(y_test, r_pred)
print("Random Forest mean squared error ", rf_ms)
rf_m = mean_absolute_error(y_test, r_pred)
print("Random Forest  mean absolute error ", rf_m)



# Random
plt.figure(figsize=(6, 5))
plt.scatter(r_df['Actual'], r_df['Predicted'], color='y')
plt.plot([min(r_df['Actual']), max(r_df['Actual'])], [min(r_df['Predicted']), max(r_df['Actual'])], '--k')
plt.axis('tight')
plt.title("Random Forest true and predicted value comparison")
plt.xlabel("y_test")
plt.ylabel("pred")
plt.tight_layout()
plt.show()