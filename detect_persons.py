import os
import subprocess
import time
import sys

START_TIME = time.time()

def run_detection_on_image(image_path, weights_path, img_size=640, device=0, classes=[0], output_name='yolov9_c_c_640_detect_all'):
    # Construct the command to run
    command = [
        'python', 'yolov9/detect.py',
        '--source', image_path,
        '--img', str(img_size),
        '--device', str(device),
        '--weights', weights_path,
        '--classes', ','.join(map(str, classes)),
        '--name', output_name,
    ]
    
    # Time the execution of the command
    start_time = time.time()
    print(f'Running detection on: {image_path}')
    subprocess.run(command)
    end_time = time.time()
    
    # Calculate and print the time taken for each command
    execution_time = end_time - start_time
    print(f'Detection completed for {image_path} in {execution_time:.2f} seconds')

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python script.py <image_path> <weights_path>")
        sys.exit(1)

    # Get the image path and weights path from command-line arguments
    image_path = sys.argv[1]
    weights_path = sys.argv[2]

    # Run the detection
    run_detection_on_image(image_path, weights_path)

    print(f'DETECTION COMPLETED IN: {time.time() - START_TIME:.2f} seconds')