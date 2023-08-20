import torch
from ultralytics.nn.tasks import attempt_load_weights

if __name__ == '__main__':

    x = torch.rand([8, 3, 640, 640])
    weights = ['./yolov8n.pt', './yolov8m.pt', './yolov8x.pt']
    device = torch.device('cpu')

    # 集成模型测试
    model = attempt_load_weights(weights)
    print("len(model(x)):", len(model(x)))
    print(model(x)[0].shape)

    # 单模型测试
    model = attempt_load_weights(weights[0])
    print("len(model(x)):", len(model(x)))
    print(model(x)[0].shape)

