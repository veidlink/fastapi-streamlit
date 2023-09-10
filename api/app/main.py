import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from io import BytesIO
from PIL import Image

app = FastAPI()

model = torch.hub.load( # Загружаем тренированную модель
    'yolo',
    'custom',
    path='api/utils/weights.pt', 
    source='local'
)

@app.get('/') # 0.0.0.0:8000/
def return_info():
    return 'Based on FastAPI' 

@app.post('/detect') # 0.0.0.0:8000/detect - адрес по которому мы будем в стримлит-файле
def detect(file: UploadFile = File(...)):  # отправлять post-запрос и получать ответ

    image_buffer = file.file.read()                  # получаем файл
    source_image = Image.open(BytesIO(image_buffer)) # читаем его содеражние (байтсы) 
    results = model(source_image)                    
    df = results.pandas().xyxy[0]
    bboxNlabel = []                                  # создаем список для координат bbox'сов и class label'лов
    for index, row in df.iterrows():
        row = row.tolist()
        xA = int(row[0])
        yA = int(row[1])
        xB = int(row[2])
        yB = int(row[3])
        label = row[6]
        bboxNlabel.append((xA, yA, xB, yB, label))
    bboxNlabel = jsonable_encoder({"data": bboxNlabel}) # кодируем в json словарь
    return JSONResponse(content=bboxNlabel)             # отправляем в streamlit