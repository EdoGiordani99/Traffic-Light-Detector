import torch

""" DATASET HYPER PARAMETERS """
TRAIN_DIR = 'data/train'
VALID_DIR = 'data/valid'
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
BATCH_SIZE = 4
RESIZE_TO = 416
NUM_EPOCHS = 100
NUM_WORKERS = 4

# lr=0.001, momentum=0.9, weight_decay=0.0005
LEARNING_RATE = 0.001
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
OPTIM = 'Adam' #SGD or Adam


""" DEVICE """
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')


""" SAVING DIRECTORY """
OUT_DIR = 'outputs'

