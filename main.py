import base64
import cv2
import numpy as np
from fastapi import FastAPI, Depends, APIRouter, HTTPException, status, Response, UploadFile, File
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
import os
import time
from matplotlib import pyplot as plt
from sqlalchemy.ext.asyncio import AsyncSession
from ultralytics import YOLO
from OCR.classifier.classifier_model import extract_sentence_detections
from OCR.detection.sentence_model import extract_sentences_from_image, CustomCRAFT
from OCR.ocr.crnn_ocr import ocr_tuple_list, CRNN
from dependencies import get_db_session
import database.db_repo as repo
import torch
from recommendation.recommendation import get_recommendations
from misc.recommendation_request_dto import recommendation_request_dto

app = FastAPI()
security = HTTPBearer()
router = APIRouter()
SECRET_KEY = os.getenv("SECRET")
ALGORITHM = os.getenv("ALGORITHM")


device = 'cuda' if torch.cuda.is_available() else 'cpu'

detection_model = CustomCRAFT()
model_path = "OCR/detection/small_sentence_model.pth"
detection_model.load_state_dict(torch.load(model_path, map_location=device))
detection_model.to(device)
detection_model.eval()  # important for inference

with open("OCR/classifier/data/classes.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]
classifier_model = YOLO("OCR/classifier/runs/yolo/weights/best.pt")

# charset = "abcčćdđefghijklmnopqrsštuvwxyzž0123456789-'& "
charset = "abcčćdđefghijklmnopqrsštuvwxyzž0123456789-'&,) "
model_path = "OCR/ocr/best_small_ocr_model.pth"
ocr_model = CRNN(num_classes=len(charset))
ocr_model.load_state_dict(torch.load(model_path, map_location=device))
ocr_model.to(device)
ocr_model.eval()




print(torch.__version__)       # should show 2.4.0+cu121
print(torch.version.cuda)      # should show 12.1
print(torch.cuda.is_available())

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    try:
        payload = jwt.decode(
            token,
            SECRET_KEY,
            algorithms=[ALGORITHM],
            issuer="PubQuizBackend",
            audience="Admin",
        )
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


@app.post("/recommendations")
async def protected_route(request: recommendation_request_dto, user=Depends(get_current_user)):
    if user['http://schemas.microsoft.com/ws/2008/06/identity/claims/role'] != 'Admin':
        raise HTTPException(status_code=403, detail="Forbidden")
    await get_recommendations(request.id)

    return Response(status_code=200)


@router.get("/feedback")
async def fetch_feedback(session: AsyncSession = Depends(get_db_session)):
    feedback = await repo.get_feedback(session)
    return feedback.to_dict(orient="records")

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}


def encode_image_to_base64(img: np.ndarray) -> str:
    _, buffer = cv2.imencode(".png", img)
    return base64.b64encode(buffer).decode("utf-8")

def visualize_results_list(results_list):
    """
    Visualize the YOLO detections stored in results_list.
    Each item in results_list is a tuple (Number_image, Answer_or_Multiple_image)
    """
    for idx, (number_img, other_img) in enumerate(results_list):
        plt.figure(figsize=(10, 4))

        # Number image
        plt.subplot(1, 2, 1)
        if number_img.ndim == 2:  # grayscale
            plt.imshow(number_img, cmap='gray')
        else:  # color
            plt.imshow(cv2.cvtColor(number_img, cv2.COLOR_BGR2RGB))
        plt.title(f"Tuple {idx} - Number")
        plt.axis('off')

        # Answer/Multiple image
        plt.subplot(1, 2, 2)
        if other_img.ndim == 2:  # grayscale
            plt.imshow(other_img, cmap='gray')
        else:
            plt.imshow(cv2.cvtColor(other_img, cv2.COLOR_BGR2RGB))
        plt.title(f"Tuple {idx} - Answer/Multiple")
        plt.axis('off')

        plt.show()

from PIL import Image
import io

def verify_image(byte_data):
    try:
        img = Image.open(io.BytesIO(byte_data))
        img.verify()  # Validate image structure
        print("VALID IMAGE")
        return True
    except Exception as e:
        print("INVALID IMAGE:", e)
        return False

def save_debug_image(byte_data, name="received.png"):
    with open(name, "wb") as f:
        f.write(byte_data)
    print("Saved image as", name)


@app.post("/sentence")
async def get_sentence(image: UploadFile = File(...)):
    # if user['http://schemas.microsoft.com/ws/2008/06/identity/claims/role'] != 'Admin':
    #     raise HTTPException(status_code=403, detail="Forbidden")
    start = time.perf_counter()

    contents = await image.read()

    verify_image(contents)
    save_debug_image(contents)
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    images = extract_sentences_from_image(detection_model, img)
    extractions = extract_sentence_detections(classifier_model, class_names, images)
    # visualize_results_list(extractions)
    answers = ocr_tuple_list(ocr_model, charset, extractions)
    print(answers)

    end = time.perf_counter()
    print(f"/sentence took {end - start:.4f} seconds")

    return answers