from ultralytics import YOLO

'''
配置文件详见ultralytics/cfg/default.yaml
'''

model = YOLO('./models/pre.pt')
if __name__ == "__main__":
    results = model.train(data="./datasets/dataset.yaml", epochs=200, batch=8, device='0',amp=False)