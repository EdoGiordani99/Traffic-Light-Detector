
import cv2
import torch
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
from albumentations.pytorch import ToTensorV2

from config import DEVICE, CLASSES, OUT_DIR

plt.style.use('ggplot')


class Averager:
    """"
    With this class we can keeps track of the training and validation
    loss values making an average for each epoch
    """
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        """
        This function updates the value of the loss while increasing
        the iteration numbers
        :param value: loss value for the i-th iteration
        :return:
        """
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        """
        :return: returns the average loss value over iteration in a epoch
        """
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations

    def reset(self):
        """
        Reset the values to zero
        """
        self.current_total = 0.0
        self.iterations = 0.0


class SaveBestModel:
    """
    This class saves the best model configuration while training.
    At each epoch we save the model only if the validation loss
    of the epoch is lower of the previous training epoch
    """

    def __init__(self):
        # initializing the loss to +infinity
        self.best_valid_loss = float('inf')

    def __call__(self, current_valid_loss, epoch, model, optimizer):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch + 1}\n")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, 'outputs/best_model.pth')


def collate_fn(batch):
    """
    To handle the data loading as different images may have different number
    of objects and to handle varying size tensors as well.
    """
    return tuple(zip(*batch))


def train_augmentation():
    """
    :return: augmentation pipeline for training data augmentation
    """
    return A.Compose([
        #A.HorizontalFlip(0.5),                  # it doesn't make sense to have a vertical flip,
        #A.MedianBlur(blur_limit=1, p=0.1),      # we generally don't see upside down lights.
        ToTensorV2(p=1.0),
    ], bbox_params={
        'format': 'pascal_voc',
        'label_fields': ['labels']
    })


def valid_augmentation():
    """
    :return: augmentation pipeline for validation data augmentation
    """
    return A.Compose([
        A.HorizontalFlip(0.5),
        ToTensorV2(p=1.0),
    ], bbox_params={
        'format': 'pascal_voc',
        'label_fields': ['labels']
    })


def show_tranformed_image(train_loader):
    """
    To activate this function, set `VISUALIZE_TRANSFORMED_IMAGES = True` in config.py.
    In order to check whether the tranformed images along with the corresponding
    labels are correct or not, this function shows the transformed images from the `train_dataloader`.
    """
    if len(train_loader) > 0:
        for i in range(1):
            images, targets = next(iter(train_loader))
            images = list(image.to(DEVICE) for image in images)
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
            boxes = targets[i]['boxes'].cpu().numpy().astype(np.int32)
            labels = targets[i]['labels'].cpu().numpy().astype(np.int32)
            sample = images[i].permute(1, 2, 0).cpu().numpy()
            for box_num, box in enumerate(boxes):
                cv2.rectangle(sample,
                              (box[0], box[1]),
                              (box[2], box[3]),
                              (0, 0, 255), 2)
                cv2.putText(sample, CLASSES[labels[box_num]],
                            (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, (0, 0, 255), 2)
            cv2.imshow('Transformed image', sample)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


def save_model(epoch, model, optimizer):
    """
    Function to save the trained model till current epoch
    """
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, 'outputs/last_model.pth')


def save_loss_plot(OUT_DIR, train_loss, val_loss):
    figure_1, train_ax = plt.subplots()
    figure_2, valid_ax = plt.subplots()

    # Training Iteration Loss
    train_ax.plot(train_loss, color='tab:blue')
    train_ax.set_xlabel('iterations')
    train_ax.set_ylabel('train loss')

    # Validation Iteration Loss
    valid_ax.plot(val_loss, color='tab:red')
    valid_ax.set_xlabel('iterations')
    valid_ax.set_ylabel('validation loss')

    # Exporting
    figure_1.savefig(f"{OUT_DIR}/train_loss.png")
    figure_2.savefig(f"{OUT_DIR}/valid_loss.png")
    print('LOSS PLOTS HAVE BEEN SAVED!')

    plt.close('all')
