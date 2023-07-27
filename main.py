import os.path
import shutil
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import cv2
import torch
import base64
import numpy as np
from inference import infer


def decode(base64_string):
    image_data = base64.b64decode(base64_string)
    np_array = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_UNCHANGED)
    return image


def encode(image_path):
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()
        base64_string = base64.b64encode(image_data).decode("utf-8")
    return base64_string


model = torch.hub.load('yolov5', 'custom', path='yolov5/runs/train/exp/weights/best.pt', source='local')
model.conf = 0.5
app = FastAPI()


# Request Body Model
class InputData(BaseModel):
    image: str


# Response Model
class OutputData(BaseModel):
    image: str
    detection: dict


@app.post("/inference/", response_model=OutputData)
def inference(data: InputData):
    try:
        os.makedirs("tempdir", exist_ok=True)
        img_string = data.image
        img = decode(img_string)
        cv2.imwrite(os.path.join('tempdir', 'input_image.png'), img)
        res = infer(model, os.path.join('tempdir', 'input_image.png'))
        img_string = encode(os.path.join("runs", "detect", "exp", "input_image.jpg"))
        shutil.rmtree("tempdir")
        shutil.rmtree("runs")
        return {"image": img_string, "detection": res}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Error occurred while generating dictionary")
