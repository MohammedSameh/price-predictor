# run using: uvicorn main:app --reload

from fastapi import FastAPI, Body
from pydantic import BaseModel
import joblib

# Load the saved model
model = joblib.load('price_range_predictor.pkl')

app = FastAPI()

# Class for mapping phone object for prediction
class PhoneData(BaseModel):
  battery_power: int
  blue: bool  # True or False
  clock_speed: float
  dual_sim: bool  # True or False
  fc: float
  four_g: bool  # True or False
  int_memory: float
  m_dep: float
  mobile_wt: float
  n_cores: int
  pc: float
  px_height: int
  px_width: int
  ram: int
  sc_h: float
  sc_w: float
  talk_time: int
  three_g: int
  touch_screen: bool  # True or False
  wifi: bool  # True or False

@app.post("/predict")
async def predict_price_range(data: PhoneData = Body(...)):
  """
  Predict the price range for a phone based on its specifications.
  """
  # Convert data to a list matching model input format (assuming your model expects a list)
  X = [
      data.battery_power, data.blue, data.clock_speed, data.dual_sim, data.fc,
      data.four_g, data.int_memory, data.m_dep, data.mobile_wt, data.n_cores,
      data.pc, data.px_height, data.px_width, data.ram, data.sc_h, data.sc_w,
      data.talk_time, data.three_g, data.touch_screen, data.wifi
  ]

  prediction = model.predict([X])

  # Convert prediction from numpy array to int64 array, which is price_range's type
  predicted_price_range = int(prediction[0])

  return {"predicted_price_range": predicted_price_range}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)


############ E.g of request payload:
# POST Request to http://127.0.0.1:8000/predict

# Body:

# {
#   "battery_power": 5000,
#   "blue": true,
#   "clock_speed": 2.0,
#   "dual_sim": true,
#   "fc": 8.0,
#   "four_g": true,
#   "int_memory": 64,
#   "m_dep": 8.0,
#   "mobile_wt": 180,
#   "n_cores": 8,
#   "pc": 16.0,
#   "px_height": 1080,
#   "px_width": 1920,
#   "ram": 8,
#   "sc_h": 15.0,
#   "sc_w": 7.0,
#   "talk_time": 2,
#   "three_g": 0,
#   "touch_screen": true,
#   "wifi": true
# }
