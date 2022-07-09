import os
import torch
import numpy as np
from cv2 import cv2
import glob as glob

from xml.etree import ElementTree as et
from torch.utils.data import Dataset, DataLoader

from utils import collate_fn, train_augmentation, valid_augmentation
from config import CLASSES, RESIZE_TO, TRAIN_DIR, VALID_DIR, BATCH_SIZE


class CustomDataset(Dataset):
    def __init__(self, data_path, width, height, classes, transforms=None):
        self.transforms = transforms
        self.data_path = data_path
        self.height = height
        self.width = width
        self.classes = classes

        # get all the image paths in sorted order
        self.image_paths = glob.glob(f"{self.data_path}/*.jpg")             # blob.blob finds all the path with the name
        self.images = [image_path.split(os.path.sep)[-1] for image_path in self.image_paths]
        self.images = sorted(self.images)

    def __getitem__(self, idx):
        """
        :param idx: index of the image
        :return: the image
        """
        # Reading, Resizing and Normalizing the images
        img_name = self.images[idx]
        img_path = os.path.join(self.data_path, img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img_resized = cv2.resize(img, (self.width, self.height))
        img_resized /= 255.0

        img_width = img.shape[1]
        img_height = img.shape[0]

        # Getting the annotations
        annot_filename = img_name[:-4] + '.xml'       # [:-4] is to remove the .jpg extension
        annot_path = os.path.join(self.data_path, annot_filename)

        boxes = []
        labels = []
        tree = et.parse(annot_path)
        root = tree.getroot()

        # box coordinates for xml files are extracted and corrected for image size given
        for member in root.findall('object'):
            # map the current object name to `classes` list to get...
            # ... the label index and append to `labels` list
            labels.append(self.classes.index(member.find('name').text))

            # X of bottom left corner
            xmin = int(member.find('bndbox').find('xmin').text)
            # X of upper right corner
            xmax = int(member.find('bndbox').find('xmax').text)
            # Y of bottom left corner
            ymin = int(member.find('bndbox').find('ymin').text)
            # Y of upper right corner
            ymax = int(member.find('bndbox').find('ymax').text)

            # scaling bounding boxes according to scaling
            xmin_scaled = (xmin / img_width) * self.width
            xmax_scaled = (xmax / img_width) * self.width
            ymin_scaled = (ymin / img_height) * self.height
            yamx_scaled = (ymax / img_height) * self.height

            boxes.append([xmin_scaled, ymin_scaled, xmax_scaled, yamx_scaled])

        # Converting into tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # How many boxes are in the image (as a zero vector)
        is_crowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = is_crowd
        image_id = torch.tensor([idx])
        target["image_id"] = image_id

        # Data Augmentation with albumentation pipeline
        if self.transforms:
            sample = self.transforms(image=img_resized,
                                     bboxes=target['boxes'],
                                     labels=labels)
            img_resized = sample['image']
            target['boxes'] = torch.Tensor(sample['bboxes'])

        return img_resized, target

    def __len__(self):
        return len(self.images)


def build_train_dataset():
    train_dataset = CustomDataset(TRAIN_DIR, RESIZE_TO, RESIZE_TO, CLASSES, train_augmentation())
    return train_dataset


def build_valid_dataset():
    valid_dataset = CustomDataset(VALID_DIR, RESIZE_TO, RESIZE_TO, CLASSES, valid_augmentation())
    return valid_dataset


def build_train_dataloader(train_dataset, num_workers=0):
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    return train_dataloader


def build_valid_dataloader(valid_dataset, num_workers=0):
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    return valid_dataloader


# execute datasets.py using Python command from Terminal...
# ... to visualize sample images
# USAGE: python datasets.py
if __name__ == '__main__':
    # sanity check of the Dataset pipeline with sample visualization
    dataset = CustomDataset(
        TRAIN_DIR, RESIZE_TO, RESIZE_TO, CLASSES
    )
    print(f"Number of training images: {len(dataset)}")

    # function to visualize a single sample
    def visualize_sample(image, target):
        for box_num in range(len(target['boxes'])):
            box = target['boxes'][box_num]
            label = CLASSES[target['labels'][box_num]]
            cv2.rectangle(
                image,
                (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                (0, 255, 0), 2
            )
            cv2.putText(
                image, label, (int(box[0]), int(box[1] - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
            )
        cv2.imshow('Image', image)
        cv2.waitKey(0)


    NUM_SAMPLES_TO_VISUALIZE = 5
    for i in range(NUM_SAMPLES_TO_VISUALIZE):
        image, target = dataset[i]
        visualize_sample(image, target)