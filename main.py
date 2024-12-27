from fastapi import FastAPI,File,UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn 
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import keras

app = FastAPI()
origins = [
    "https://potatodisease-deeplearning.netlify.app",
    # "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware  ,
    allow_origins=origins,  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

MODEL = keras.layers.TFSMLayer(r"./1")

CLASSNAME = ["EARLY BLIGHT", "LATE BLIGHT", "HEALTHY"]

@app.get("/ping")
async def ping():
    return "Hello i am alive"


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


@app.post("/predict")
async def predict(file: UploadFile=File(...)):
    image = read_file_as_image( await file.read())
    img_batch_np = np.expand_dims(image,0)
    predictions = MODEL.call(img_batch_np)
    predicted_class = CLASSNAME[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    # print(predictions)
    return{
        'class':predicted_class,
        'confidence':float(confidence)
    }



if __name__ == "__main__":
    uvicorn.run(app,host='localhost',port=8000)