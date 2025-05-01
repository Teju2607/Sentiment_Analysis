from fastapi import FastAPI
from pydantic import BaseModel
from predict import load_model, predict

app = FastAPI()

# Load the model at startup
model = load_model("data/model.dat.gz")

# Request schema
class InputText(BaseModel):
    text: str

@app.get("/")
def root():
    return {"message": "Sentiment Analysis API is running."}

@app.post("/predict")
def predict_sentiment(input: InputText):
    result = predict(model, input.text)
    return {"prediction": result}
