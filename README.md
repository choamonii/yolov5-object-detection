# yolov5-object-detection
Object detection model trained with YOLOv5 in Python

An independent machine learning project I completed at UCLA: training a YOLOv5 model to detect [objects] in images.

## Data
- ~540 training images and 180 validation images of paper money bills, one class ("money")
- Rotated images so the model detects bills held at any angle

## Approach
- Model: YOLOv5s (pretrained by Ultralytics), fine-tuned in Python with PyTorch
- Training: 5 epochs, image size 640, batch size 16, CPU only (about 1.1 hours)

## Results
- mAP@0.5: 0.77 · Precision: 0.72 · Recall: 0.79
- Real-time webcam detection at about 4 frames per second on CPU

<img width="600" alt="Live webcam detection of a $2 bill" src="https://github.com/user-attachments/assets/67b12ca8-18e6-47da-a2d8-5bad83ff3b2d" />


## What I learned
- **Starting from a pretrained model goes a long way.** Fine-tuning YOLOv5s on only ~540 images reached 0.77 mAP@0.5 in 5 epochs on a regular CPU. Training from scratch would have needed far more data and time.
- **The data matters as much as the model.** Adding rotated images helped the model find bills held at different angles, as real people hold them in front of a camera.
- **There's still room to improve.** The model finds bills reliably (0.77 mAP@0.5), but its boxes are less precise (0.25 mAP@0.5:0.95). Next steps would be more varied photos (different lighting and backgrounds), longer training on a GPU, and separate classes for each denomination ($1, $5, $20).
- - **Debugging is part of the job.** My first live webcam run crashed with `'NoneType' object has no attribute 'shape'`, which meant the program wasn't getting frames from the camera. I traced the error back through the code, fixed it, and got real-time detection working.

## Credits
YOLOv5 by [Ultralytics](https://github.com/ultralytics/yolov5).
