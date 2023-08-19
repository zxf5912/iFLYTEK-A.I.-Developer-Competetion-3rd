# XF2023
XunFei2023,Competition,unmanned ship detected the barriers.

# 参数情况与结果
- yolov8x/epochs=200 batch=4 imgsz=640 lr=0.01/mAP0.5-0.95:0.557/lb:0.45592/未做数据增广
- yolov8x/epochs=200 batch=4 imgsz=640 lr=0.01/mAP0.5-0.95:0.561/lb:0.43762/针对数量少的类别ball，rubbish做了类别不平衡方式的增广
