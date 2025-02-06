import logging
from pathlib import Path
import subprocess
import sys
import time
import json
import multiprocessing
import queue
import os
import glob
import torch
import cv2

from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlShutdown
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize NVML for CUDA memory management
nvmlInit()

# Project setup
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root) + '/yolov9/')

from yolov9.models.common import DetectMultiBackend
from yolov9.utils.dataloaders import LoadImages
from yolov9.utils.general import check_img_size, increment_path, non_max_suppression
from yolov9.utils.torch_utils import select_device

# FastAPI app initialization
app = FastAPI()

# Global variables
image_queue = multiprocessing.Queue()
models = []

class ImageRequest(BaseModel):
    image_path: str
    weights_path: str = './yolov9/yolov9-e.pt'
    img_size: int = 640
    device: int = 0
    classes: list = [0]
    output_name: str = 'yolov9_detect'
    model_name: str = ""

def get_model_by_name(model_name):
    # Check if the model name exists in the dictionary
    if model_name in models:
        return models[model_name]
    else:
        logger.error(f"Model {model_name} not found!")
        return None
    
def get_cuda_memory(device_index=0):
    try:
        device_handle = nvmlDeviceGetHandleByIndex(device_index)
        mem_info = nvmlDeviceGetMemoryInfo(device_handle)
        return mem_info.used, mem_info.total
    except Exception as e:
        logger.error(f"Error checking CUDA memory: {e}")
        return 0, 1

def run_detection(image_data, model):
    logger.info(f"Starting detection for {image_data[0]}")
    image_path, weights_path, img_size, device, classes, output_name, model_name = image_data

    start_time = time.time()

    try:
        dataset = LoadImages(image_path, img_size=img_size)
        save_dir = increment_path(Path('runs/detect') / 'exp', exist_ok=True)
        save_dir.mkdir(parents=True, exist_ok=True)

        for path, im, im0s, _, _ in dataset:
            imos_resized = cv2.resize(im0s, (640,340))
            im = torch.from_numpy(im).to(device)
            im = im.half() if model.fp16 else im.float()
            im /= 255

            if len(im.shape) == 3:
                im = im[None]

            pred = model(im)
            pred = non_max_suppression(pred, 0.25, 0.45, classes=classes)

            models[model_name]['busy']= False

            for det in pred:
                if len(det):
                    for *xyxy, conf, cls in reversed(det):
                        label = f'{model.names[int(cls)]} {conf:.2f}'
                        cv2.rectangle(imos_resized, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255, 0, 0), 2)
                        cv2.putText(imos_resized, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            output_path = save_dir / f"{Path(path).stem}_detected.jpg"
            cv2.imwrite(str(output_path), imos_resized)

        logger.info(f"Detection completed for {image_path} | {time.time()-start_time} s")
        return {'message': f'Detection completed for {image_path} | {time.time()-start_time} s'}

    except Exception as e:
        logger.error(f"Error in run_detection: {e}")
        return {'error': str(e)}

import random

def process_image():

    def get_available_model():
        """Randomly selects an available (non-busy) model and marks it as busy."""
        try:
            available_models = [key for key, val in models.items() if not val["busy"]]

            if not available_models:
                logger.error("No available models!")
                queue_length = image_queue.qsize()
                logger.info(f"Queue Length: {queue_length}")
                return "", None
            
            chosen_model_name = random.choice(available_models)
            models[chosen_model_name]["busy"] = True  # Mark as busy
            return chosen_model_name, models[chosen_model_name]["model"]
        except Exception as e:
            logger.error(f"ERROR: "+str(e))
    
    while True:
        try:
            queue_length = image_queue.qsize()
            logger.info(f"Queue Length: {queue_length}")
            
            image_data = image_queue.get(timeout=60)
            logger.info(f"Processing image: {image_data[0]}")
            logger.info(f"Image data: {image_data}")  # Log image data for debugging

            chosen_model_name, model = get_available_model()
            image_data  += (chosen_model_name,)
            
            if model:
                logger.info(f"Using {chosen_model_name} in the pipeline.")
                used, total = get_cuda_memory()
                logger.info(f"CUDA memory usage: {used}/{total} bytes")

                if used / total < 0.9:
                    result = run_detection(image_data, model)
                    logger.info(f"Result: {result}")
                    break
                else:
                    logger.warning("High CUDA usage, re-queuing image.")
                    image_queue.put(image_data)
                    time.sleep(10)

        except queue.Empty:
            logger.info("No images in queue, waiting...")
            continue
        except Exception as e:
            logger.error(f"Unexpected error: {e}")

def start_worker_pool(num_workers=20):
    logger.info(f"Starting worker pool with {num_workers} workers.")
    workers = []
    for _ in range(num_workers):
        p = multiprocessing.Process(target=process_image)
        p.start()
        workers.append(p)
        logger.info(f"Started worker process {p.pid}")
    return workers

@app.on_event("startup")
async def on_startup():
    global models
    logger.info("Starting FastAPI server.")

    try:
        weights = ['./yolov9/yolov9-e.pt'] * 20
        device = select_device('0')
        
        # Create a dictionary with model names from "model_0" to "model_19"
        models = {f"model_{i}": {"model": DetectMultiBackend(w, device=device), "busy": False} for i, w in enumerate(weights)}


        logger.info(f"Models loaded: {len(models)}")  # Log model loading
        start_worker_pool(num_workers=20)

    except Exception as e:
        logger.error(f"Error during startup: {e}")


@app.post("/process_image/")
async def process_image_endpoint(data: ImageRequest):
    logger.info(f"Received image for processing: {data.image_path}")
    image_queue.put((data.image_path, data.weights_path, data.img_size, data.device, data.classes, data.output_name))
    logger.info(f"Image added to queue: {data.image_path}")
    process_image()
    return {'message': 'Image added to queue'}

@app.on_event("shutdown")
async def on_shutdown():
    logger.info("Shutting down FastAPI server.")
    for p in multiprocessing.active_children():
        logger.info(f"Terminating process {p.pid}")
        p.terminate()
        p.join()

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    uvicorn.run(app, host="0.0.0.0", port=8000)
