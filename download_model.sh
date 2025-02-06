#!/bin/bash

# Downloading the yolov9 model
echo "Starting download of yolov9-e-converted.pt model..."
wget https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-e-converted.pt -O yolov9-e-converted.pt

# Check if download was successful
if [ $? -eq 0 ]; then
  echo "Download completed successfully."
else
  echo "Download failed."
fi

