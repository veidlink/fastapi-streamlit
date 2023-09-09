import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from io import BytesIO
from PIL import Image

app = FastAPI()
model = torch.hub.load(
    'yolo',
    'custom',
    path='api/utils/weights.pt',
    source='local'
)

@app.get('/')
def return_info():
    return 'Based on FastAPI'

@app.post('/detect')
def detect(file: UploadFile = File(...)):

    image_buffer = file.file.read()
    print(image_buffer)
    source_image = Image.open(BytesIO(image_buffer))
    print(type(source_image))
    results = model(source_image)
    df = results.pandas().xyxy[0]
    bboxNlabel = []
    for index, row in df.iterrows():
        row = row.tolist()
        xA = int(row[0])
        yA = int(row[1])
        xB = int(row[2])
        yB = int(row[3])
        label = row[6]
        bboxNlabel.append((xA, yA, xB, yB, label))
    bboxNlabel = jsonable_encoder({"data": bboxNlabel})
    return JSONResponse(content=bboxNlabel)