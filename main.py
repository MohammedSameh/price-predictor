from fastapi import FastAPI, Body
from pydantic import BaseModel

# Load the saved model
import joblib
model = joblib.load('price_range_predictor.pkl')

app = FastAPI()

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
      data.touch_screen, data.wifi
  ]

  prediction = model.predict([X])

  # Convert prediction from numpy array to int64 array, which is price_range's type
  predicted_price_range = int(prediction[0])

  return {"predicted_price_range": predicted_price_range}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
