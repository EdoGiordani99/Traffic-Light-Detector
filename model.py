import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def create_model(num_classes: int, name: str, pretrained: bool):

    if name == 'resnet50':
        weights = 'FasterRCNN_ResNet50_FPN_Weights.DEFAULT'
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights = weights, pretrained=pretrained)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    elif name == 'mobilenet':
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=pretrained)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    else:
        raise ValueError('Invalid name of the model. Please choose between: "resnet50" or "mobilenet"')

    return model

