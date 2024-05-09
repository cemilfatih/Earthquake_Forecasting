Hi everyone!

This repo has a basic aproach for forecasting earthquake's magnitude with time, latitude, longitude and depth

Dataset is looking like this:
data[['Timestamp', 'latitude', 'longitude', 'depth', 'mag']]

We are using Random Forest Regressor in order to maintance the frequency of an earthquake that happens in some regions. 

Regions are in Turkey for now, we will add all around the world soon. 

Time is added Dd/Mm/Yy and transformed to a timestamp for better result. 

This is an open source project, feel free to contribute.


