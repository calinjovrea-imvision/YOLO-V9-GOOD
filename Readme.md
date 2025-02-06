*** README ***

RUN THE COMMANDS IN THE FOLLOWING ORDER

* CLONE THE REPO *
https://github.com/calinjovrea-imvision/YOLO-V9-GOOD.git

* UNZIP YOLO *
unzip yolov9.zip

* DOWNLOAD THE MODEL *
./download_model.sh

* MOVE THE MODEL TO SUPPOSED FOLDER *
mv yolov9-e-converted.pt yolov9/yolov9-e.pt

* CREATE VENV *
python -m venv yolov9_good

* INSTALL PACKAGES *
python -m pip install -r requirements.txt 

* START THE PROGRAM *
python app__working.py

* RUN THE IMAGES *
python run.py
