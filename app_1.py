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

from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root)+'/yolov9/')

from yolov9.models.common import DetectMultiBackend
from yolov9.utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from yolov9.utils.general import LOGGER, Profile, check_file, check_img_size, check_imshow, colorstr, cv2, increment_path, non_max_suppression
from yolov9.utils.torch_utils import select_device, smart_inference_mode

# Initialize NVML for GPU monitoring
nvmlInit()
device_handle = nvmlDeviceGetHandleByIndex(0)

# Queue for images when CUDA memory is full
image_queue = multiprocessing.Queue()

# FastAPI application
app = FastAPI()

models = {}
count =0
device='0'

class ImageRequest(BaseModel):
    image_path: str
    weights_path: str = './yolov9/yolov9-e.pt'
    img_size: int = 640
    device: int = 0
    classes: list = [0]
    output_name: str = 'yolov9_c_c_640_detect_all'

def get_cuda_memory():
    nvmlInit()
    device_handle = nvmlDeviceGetHandleByIndex(0)
    mem_info = nvmlDeviceGetMemoryInfo(device_handle)
    used = mem_info.used / (1024 ** 2)  # Convert to MB
    total = mem_info.total / (1024 ** 2)
    print(f"[LOG] CUDA Memory - Used: {used:.2f} MB / Total: {total:.2f} MB")
    return used, total

def run_detection(image_path, weights_path='./yolov9/yolov9-e.pt', img_size=640, device=0, classes=[0], output_name='yolov9_c_c_640_detect_all', model=1):
    
    FILE = Path(__file__).resolve()
    ROOT = FILE.parents[0]  # YOLO root directory
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))  # add ROOT to PATH
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
    project=ROOT / 'runs/detect'  # save results to project/name
    name='exp'  # save results to project/name
    save_txt=False

    conf_thres=0.25  # confidence threshold
    iou_thres=0.45  # NMS IOU threshold
    max_det=50
    agnostic_nms=False
    
    print(f"Running detection for {image_path}...")
    start_time = time.time()
    try:
        source = str(image_path)

        # Check if the source is a directory or a file
        if os.path.isdir(source):
            # If it's a folder, use glob to get all .jpg, .jpeg, and .png files recursively
            source_files = glob.glob(f"{source}/**/*.jpg", recursive=True) + \
                        glob.glob(f"{source}/**/*.jpeg", recursive=True) + \
                        glob.glob(f"{source}/**/*.png", recursive=True)
        elif os.path.isfile(source):
            # If it's a single file, wrap it in a list
            source_files = [source]
        else:
            # Handle the case where the source is neither a file nor a directory
            raise ValueError(f"Source {source} is neither a valid file nor a valid directory.")

        # Optionally save inference images
        save_img = [x for x in source_files]

        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=True)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        
        # Load model
        stride, names, pt = models[model][0].stride, models[model][0].names, models[model][0].pt
        imgsz = check_img_size(imgsz=(360,640), s=stride)  # check image size


        # Dataloader
        dataset = LoadImages(source_files, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)

        # Run inference on each image or video frame
        for path, im, im0s, _, _ in dataset:
            imos_resized = cv2.resize(im0s, (640,340))
            print(imos_resized.shape)
            im = torch.from_numpy(im).to(device)
            im = im.half() if models[model][0].fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # normalize image
            if len(im.shape) == 3:
                im = im[None]  # add batch dimension

            # Perform inference with each model
            pred_all = []
            start_time = time.time()
            for model in models[model]:  # Sequential inference on all models
                pred = model(im)
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
                pred_all.append(pred)

            # Process predictions from all models
            for i, det in enumerate(pred_all):  # per model
                if len(det):
                    for *xyxy, conf, cls in reversed(det[0]):  # process first model's detections
                        label = f'{names[int(cls)]} {conf:.2f}'
                        cv2.rectangle(imos_resized, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255, 0, 0), 2)
                        cv2.putText(imos_resized, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            LOGGER.info(f"Processed {path} in {time.time() - start_time:.3f}s")

            # Save results
            if save_img:
                cv2.imwrite(str(save_dir / path.split('/')[-1].replace('.jpg.jpg', '.jpg')), imos_resized)
            LOGGER.info(f'Results saved to {save_dir}')
            LOGGER.info(f"PROCESSED IN {time.time() - start_time} s")
    except subprocess.CalledProcessError as e:
        print(f"Error running detection: {e}")
        return {'error': str(e)}

    execution_time = time.time() - start_time
    print(f"Detection completed in {execution_time:.2f} seconds")  # Add logging
    return {'message': f'Detection completed in {execution_time:.2f} seconds'}

def process_image(image_data, model):
    image_path, weights_path, img_size, device, classes, output_name = image_data
    retry_count = 0
    while retry_count < 5:  # Retry up to 5 times
        used, total = get_cuda_memory()
        # Process the image if memory usage is under threshold
        if used / total < 0.9:
            result = run_detection(image_path, weights_path, img_size, device, classes, output_name, model)
            print(f"Processing {image_path}: {result['message']}")
            return result
        else:
            retry_count += 1
            print(f"Memory full, requeuing {image_path} attempt {retry_count}")
            image_queue.put(image_data)  # Put it back in the queue
            time.sleep(5)  # Delay before retrying

    print(f"Failed to process {image_path} after {retry_count} attempts")
    return None

def worker(args):
    while True:
        try:
            image_data = image_queue.get(timeout=60)  # Wait for task
            if image_data is None:  # Sentinel value to stop the worker
                break
            print(f"Worker processing image: {image_data[0]}")  # Log which image is being processed

            process_image(image_data, args)
        except queue.Empty:
            continue

def start_worker_pool(num_workers=4):
    workers = []
    count = 0
    for _ in range(num_workers):
        
        p = multiprocessing.Process(target=worker, args=(count,))
        p.start()
        
        workers.append(p)
        count += 1
    print(f"Started {num_workers} workers.")  # Log the number of workers started
    return workers

# Start worker pool inside the ASGI app
workers = []

@app.on_event("startup")
async def on_startup():

    try:
        # Initialize NVML
        nvmlInit()

        # Get handle for the first GPU (GPU 0)
        handle = nvmlDeviceGetHandleByIndex(0)

        # Get memory info for the selected GPU
        memory_info = nvmlDeviceGetMemoryInfo(handle)
        print(f"Total memory: {memory_info.total}")
        print(f"Free memory: {memory_info.free}")
        print(f"Used memory: {memory_info.used}")

    except NVMLError as e:
        print(f"NVML Error: {e}")

    global workers
    
    print("Starting worker pool...")
    workers = start_worker_pool(num_workers=1)

@app.post("/process_image/")
async def process_image_endpoint(data: ImageRequest):
    try:
        image_path = data.image_path
        weights_path = data.weights_path
        img_size = data.img_size
        device = data.device
        classes = data.classes
        output_name = data.output_name

        # Add image processing request to the queue
        image_queue.put((image_path, weights_path, img_size, device, classes, output_name))
        print(f"Added image {image_path} to queue")  # Log the image added

        return {'message': 'Image added to queue'}

    except Exception as e:
        return {'error': str(e)}

@app.on_event("shutdown")
async def on_shutdown():
    print("Shutting down workers...")
    for worker_process in workers:
        worker_process.terminate()  # Terminate the workers gracefully
        worker_process.join()  # Wait for workers to terminate
    print("All workers terminated.")

if __name__ == "__main__":
    # Run the app with Uvicorn

    weights = ['yolov9-e.pt' for w in range(20)]

    device = select_device('0')
    models_ = [DetectMultiBackend(w, device=device) for w in weights]  # Load all 8 models
    count = 0
    for model in models_:
        models[count] = models_
        count += 1

    uvicorn.run(app, host="0.0.0.0", port=8000)
