from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import uvicorn
from PIL import Image
import io

# Create FastAPI app
app = FastAPI()

# Load Model
model = YOLO(r'weights\100e203.pt')  # Change to your custom model if needed

@app.get("/")  # Optional: homepage test route
def read_root():
    return {"message": "Server is alive."}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents))

    results = model.predict(img)
    return results[0].tojson()
