from typing import Union
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from time import strftime
import torch
import shutil
import os
import requests
from src.utils.init_path import init_path
from src.utils.preprocess import CropAndExtract
from src.generate_batch import get_data
from src.generate_facerender_batch import get_facerender_data
from src.test_audio2coeff import Audio2Coeff
from src.facerender.animate import AnimateFromCoeff

app = FastAPI()

RESULT_DIR = "./results"
os.makedirs(RESULT_DIR, exist_ok=True)

class Item(BaseModel):
    image_link: str
    audio_link: str


def download_file(url, save_path):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return save_path
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Error downloading file: {e}")

@app.post("/generate/")
async def generate_video(item: Item):
    save_dir = os.path.join(RESULT_DIR, strftime("%Y_%m_%d_%H.%M.%S"))
    os.makedirs(save_dir, exist_ok=True)
    
    pic_path = download_file(item.image_link, os.path.join(save_dir, "image.png"))
    audio_path = download_file(item.audio_link, os.path.join(save_dir, "audio.wav"))
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    preprocess_model = CropAndExtract(init_path("./checkpoints", "./src/config", 256, False, "full"), device)
    
    first_coeff_path, crop_pic_path, crop_info = preprocess_model.generate(pic_path, save_dir, "full", source_image_flag=True)
    if first_coeff_path is None:
        raise HTTPException(status_code=400, detail="Error extracting coefficients from the image")
    
    batch = get_data(first_coeff_path, audio_path, device, None, still=True)
    coeff_path = Audio2Coeff(init_path("./checkpoints", "./src/config", 256, False, "full"), device).generate(batch, save_dir, 0, None)
    
    data = get_facerender_data(coeff_path, crop_pic_path, first_coeff_path, audio_path, 2, None, None, None, expression_scale=1.0, still_mode=True, preprocess="full", size=256)
    
    result = AnimateFromCoeff(init_path("./checkpoints", "./src/config", 256, False, "full"), device).generate(data, save_dir, pic_path, crop_info, enhancer=None, background_enhancer=None, preprocess="full", img_size=256)
    
    video_path = save_dir + ".mp4"
    shutil.move(result, video_path)
    return {"video_url": f"http://localhost:8000/videos/{os.path.basename(video_path)}"}

@app.get("/videos/{filename}")
def get_video(filename: str):
    file_path = os.path.join(RESULT_DIR, filename)
    if os.path.exists(file_path):
        return file_path
    raise HTTPException(status_code=404, detail="File not found")
