import os
import subprocess

# Directory containing the images
base_directory = "/root/servers/IMV_SEMANTIC_SEARCH_YOLO_SERVER/INDEX/index_hetz4/output/INDEX/hetz1_index"

# Path to the YOLO weights
weights_path = "./yolov9/yolov9-e.pt"

# Function to process each image
def process_image(image_path, weights_path):
    command = [
        "curl", "-X", "POST", "http://localhost:8000/process_image/",
        "-H", "Content-Type: application/json",
        "-d", f'{{"image_path": "{image_path}", "weights_path": "{weights_path}"}}'
    ]
    
    print(command)
    try:
        result = subprocess.run(command, capture_output=True, text=True)
        
        # Print standard output and error
        print("Standard Output:", result.stdout)
        print("Standard Error:", result.stderr)
        
        if result.returncode != 0:
            print(f"Error running detection: {result.stderr}")
            return {'error': result.stderr}
        
    except subprocess.CalledProcessError as e:
        print(f"Error running detection: {e}")
        return {'error': str(e)}

# Recursively find all image files
for root, _, files in os.walk(base_directory):
    for file in files:
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(root, file)
            print(f"Processing {image_path}")
            process_image(image_path, weights_path)
