from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
import tensorflow as tf
import numpy as np
import cv2
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()
origins = [
    "http://localhost:4200",
   
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)
model = tf.keras.models.load_model("Model_sm", compile=False)
print("✅ Model loaded successfully")
Mapping = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H',
    8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P',
    16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X',
    24: 'Y', 25: 'Z', 26: 'del', 27: 'nothing', 28: 'space'
}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    np_arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (32, 32)) 
    img_array = np.expand_dims(img.astype("float32") / 255.0, axis=0)

    pred = model.predict(img_array)
    pred_index = np.argmax(pred)
    pred_char = Mapping[pred_index]
    confidence = float(pred[0][pred_index] * 100)

    return {"prediction": pred_char, "confidence": f"{confidence:.2f}%"}
@app.get("/", response_class=HTMLResponse)
async def home():
    html_content = """
    <html>
        <head>
            <title>Sign Language API</title>
        </head>
        <body>
            <h1>Welcome to Sign Language Recognition API</h1>
            <p>Use /predict endpoint to send an image.</p>
        </body>
    </html>
    """
    return html_content
