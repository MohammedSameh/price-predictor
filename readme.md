# Phone Price Range Prediction API with FastAPI

This project implements a FastAPI API for predicting phone price ranges based on phone specifications. It utilizes a pre-trained Random Forest model to make predictions.

## Dependencies:

Python 3.x (https://www.python.org/downloads/)

scikit-learn (https://scikit-learn.org/)

pandas (https://pandas.pydata.org/)

pydantic (https://docs.pydantic.dev/latest/)

uvicorn (https://www.uvicorn.org/)

## Installation:

1. Create a virtual environment (optional) for managing dependencies:

```
python -m venv venv
source venv/bin/activate  # Activate virtual environment (Linux/macOS)
venv\Scripts\activate.bat  # Activate virtual environment (Windows)
```

3. Install required libraries within the virtual environment:
```
pip install scikit-learn pandas pydantic uvicorn
```

4. Ensure the pre-trained Random Forest model is saved as price_range_predictor.pkl. Ensure this file is in the same directory as your main.py script.

5. Run the API:
```
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
Note: Replace 8000 with your desired port number if needed.

## Request

The API provides a POST endpoint at /predict for making predictions. You can use tools like Postman or curl to send requests with JSON data representing phone specifications according to the schema defined in the PhoneData class within main.py.

### Example Request:

```
{
  "battery_power": 5000,
  "blue": true,
  "clock_speed": 2.0,
  "dual_sim": true,
  "fc": 8.0,
  "four_g": true,
  "int_memory": 64,
  "m_dep": 8.0,
  "mobile_wt": 180,
  "n_cores": 8,
  "pc": 16.0,
  "px_height": 1080,
  "px_width": 1920,
  "ram": 8,
  "sc_h": 15.0,
  "sc_w": 7.0,
  "talk_time": 2,
  "three_g": 0,
  "touch_screen": true,
  "wifi": true
}
```

## Response:

The API will respond with a JSON object containing the predicted price range for the provided phone specifications.

### Example Response:

```
{
  "predicted_price_range": 1  
}
```
