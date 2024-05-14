import datetime
import time
import pandas as pd
import joblib

#Date,Latitude,Longitude,Depth,Magnitude
#2023-02-06 01:17:34,37.2260,37.0140,10.0,7.8


class rf_regressor:
    def __init__(self, latitude, longitude, depth, time):
        self.latitude = latitude
        self.longitude = longitude
        self.depth = depth
        self.time = time
    
    def create_timestamp(self):

        # all the inputs will be string type so we need to convert them to datetime

        self.time = datetime.datetime.strptime(self.time, '%Y-%m-%d %H:%M:%S')
        timestamp = time.mktime(self.time.timetuple())
        return timestamp
    
    def prepare_data(self):
        timestamp = self.create_timestamp()
        data = {'Timestamp': timestamp, 'Latitude': [self.latitude], 'Longitude': [self.longitude], 'Depth': [self.depth]}
        data = pd.DataFrame(data)
        data = data[['Timestamp', 'Latitude', 'Longitude', 'Depth']].values
        return data
    
    def predict(self):
        data = self.prepare_data()
        model = joblib.load('random_forest_model.pkl')
        prediction = model.predict(data)
        return prediction