from model import rf_regressor
import json
from fastapi import FastAPI, HTTPException

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/predict/")
def predict_earthquake(data: dict):
    
    latitude = data['Latitude']
    longitude = data['Longitude']
    depth = data['Depth']
    time = data['Date']

    latitude = "37.2260"
    longitude = "37.0140"
    depth = "10.0"
    time = "2023-02-06 01:17:34"

    model = rf_regressor(latitude, longitude, depth, time)
    prediction = model.predict()
    return {"Prediction": str(prediction)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8003)


"""
{
"Date":"2023-02-06 01:17:34",
"Latitude":"37.2260",
"Longitude":"37.0140",
"Depth":"10.0"
}
"""
