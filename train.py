import time
import torch
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from model import create_model
from utils import Averager, SaveBestModel, save_model, save_loss_plot

from dataset import (
    build_train_dataset, build_valid_dataset,
    build_train_dataloader, build_valid_dataloader
)

from config import (
    DEVICE, NUM_CLASSES, NUM_EPOCHS, OUT_DIR, MODEL_NAME,
    NUM_WORKERS, PRETRAINED, 
    LEARNING_RATE, MOMENTUM, WEIGHT_DECAY, OPTIM
)

plt.style.use('ggplot')


def train(train_dataloader, model):
    """
    Training algorithm for 1 epoch. We take the batch, we pass it through the model and
    we get the loss for each element in the batch. We collect the losses from all training
    samples.
    :param train_dataloader: the training dataset
    :param model: the model
    :return: list of the losses of all the batches during the train_itr of training (this value
    will be averaged then to compute the final training_loss of the epoch).
    """
    # print('Training')
    global train_itr
    global train_iter_loss_list

    prog_bar = tqdm(train_dataloader, total=len(train_dataloader))

    # i = number of the batch element
    # data = the batch element
    # is like we are iterating for each batch in batches
    for i, data in enumerate(prog_bar):
        optimizer.zero_grad()
        images, targets = data

        # Converting the batch into a lists
        images = list(img.to(DEVICE) for img in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        # Getting the losses for all the images
        loss_dict = model(images, targets)

        # Summing the losses for all the element in the batch
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        train_iter_loss_list.append(loss_value)

        train_loss_hist.send(loss_value)

        # Backpropagation
        losses.backward()
        optimizer.step()

        train_itr += 1
    
        # update the loss value in the progress barr
        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")

    return train_iter_loss_list


def validate(valid_dataloader, model):
    """
        Similar to training algorithm for 1 epoch but without backpropagation of the loss.
        We get the loss for each element in the batch and collect the losses from all training
        samples.
        :param valid_dataloader: the validation dataset into batches
        :param model: the model
        :return: list of the losses of all the batches during the train_itr of training (this value
        will be averaged then to compute the final training_loss of the epoch).
        """

    print('Validating')
    global val_itr
    global val_iter_loss_list

    prog_bar = tqdm(valid_dataloader, total=len(valid_dataloader))
    
    for i, data in enumerate(prog_bar):
        images, targets = data
        
        images = list(img.to(DEVICE) for img in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        # this tells that we don't have to back propagate loss during validation
        with torch.no_grad():
            loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        val_iter_loss_list.append(loss_value)

        val_loss_hist.send(loss_value)

        val_itr += 1

        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")
    return val_iter_loss_list


if __name__ == '__main__':
    # Building the dataset
    train_dataset = build_train_dataset()
    valid_dataset = build_valid_dataset()

    # Creating the dataloaders for creating training / validation batches
    train_dataloader = build_train_dataloader(train_dataset, NUM_WORKERS)
    valid_dataloader = build_valid_dataloader(valid_dataset, NUM_WORKERS)

    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(valid_dataset)}\n")

    # Building the model
    model = create_model(num_classes=NUM_CLASSES, name=MODEL_NAME, pretrained=PRETRAINED)
    model = model.to(DEVICE)

    params = [p for p in model.parameters() if p.requires_grad]

    
    # Setting the optimizer
    if OPTIM == 'SGD': 
        optimizer = torch.optim.SGD(params, 
                                    lr=LEARNING_RATE, 
                                    momentum=MOMENTUM, 
                                    weight_decay=WEIGHT_DECAY)
        print('Setting the SDG optimizer')
    elif OPTIM == 'Adam': 
        optimizer = torch.optim.Adam(params, 
                                    lr=LEARNING_RATE,  
                                    weight_decay=WEIGHT_DECAY)
        print('Setting the Adam optimizer')
    else: 
        raise ValueError('Optimizer Name is not valid. Please use "SGD" or "Adam"')
        
    # Train / Val histories for epochs!
    train_loss_hist = Averager()
    val_loss_hist = Averager()

    train_itr = 1
    val_itr = 1

    # Iteration loss list for ALL THE TRAINING PROCEDURE
    train_iter_loss_list = []
    val_iter_loss_list = []

    # Save best model class
    save_best_model = SaveBestModel()

    # start the training epochs
    for epoch in range(NUM_EPOCHS):
        print(f"\nEPOCH {epoch+1} of {NUM_EPOCHS}")

        # reset the training and validation loss histories for the current epoch
        train_loss_hist.reset()
        val_loss_hist.reset()

        start = time.time()

        # Training Step
        train_loss = train(train_dataloader, model)

        # Validation Step
        val_loss = validate(valid_dataloader, model)

        # Printing the epoch resume
        print(f"Epoch #{epoch+1} train loss: {train_loss_hist.value:.3f}")   
        print(f"Epoch #{epoch+1} validation loss: {val_loss_hist.value:.3f}")   
        end = time.time()
        print(f"Took {((end - start) / 60):.3f} minutes for epoch {epoch}")

        # if the current model is the best --> save it
        save_best_model(val_loss_hist.value, epoch, model, optimizer)

        # save the current epoch model
        save_model(epoch, model, optimizer)

        # save loss plot
        save_loss_plot(OUT_DIR, train_loss, val_loss)
        
        # sleep for 5 seconds after each epoch
        time.sleep(5)
