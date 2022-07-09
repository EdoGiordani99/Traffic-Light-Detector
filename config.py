import torch

""" DATASET HYPER PARAMETERS """
TRAIN_DIR = 'data/Traffic-Light-Detector-2/train'
VALID_DIR = 'data/Traffic-Light-Detector-2/valid'
CLASSES = [
    '__background__', 'Unk',
    'Red Car', 'Red Ped',
    'Green Car', 'Green Ped',
    'Yellow Car', 'Yellow Ped']
NUM_CLASSES = len(CLASSES)

VISUALIZE_TRANSFORMED_IMAGES = True


""" MODEL HYPER PARAMETERS """
MODEL_NAME = 'resnet50'
PRETRAINED = True


""" TRAINING HYPER PARAMETERS """
BATCH_SIZE = 8
RESIZE_TO = 416
NUM_EPOCHS = 10
NUM_WORKERS = 4


""" DEVICE """
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')


""" SAVING DIRECTORY """
OUT_DIR = 'outputs'