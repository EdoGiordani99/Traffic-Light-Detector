# Traffic Light Detection - TLD
Autonomous driving is increasingly becoming a reality common to all of us thanks to the continuous and rapid development of Artificial Intelligence. A fundamental role underlying this technology is played by sensors, cameras and the digital processing of the captured signals. Obviously, the reliability of these systems must be the key element for optimal and safe operation of the device.

In this project a traffic light detector will be implemented. This will be also capable of classifying the traffic lights status: red, amber and green.

## Base Idea
The very first intuition is the one adopted in the FasterRCNN algorithm which is a lighter and faster version of Fast RCNN. This faster system should be great since it will allow us to use it in embedding systems. A possible approach to the algorithm I will implement is the following:
    1.  Feature extraction: a CNN will be used to extract the main features of the image. 
    2.  Region of Interest: Once we have our feature maps, we want to implement an algorithm which will find possible regions of interest (RoI).
    3.  RoI pooling: to bring all the regions of the same size.
    4.  Classification: we will finally use a fully connected network to classify the objects in the bounding boxes.

## Dataset
The dataset that will be used to train the Region Proposal Network is the Kaggle LISA Traffic Light Dataset. The database consists of continuous test and training video frames captured by a camera mounted on the roof of a vehicle. The images represent both night and daytime driving and include weather variations. The training dataset consists of 13 daytime clips and 5 nighttime clips. As further work, it would be interesting to compare a model which uses a pre-trained feature extractor with one trained by us on the COCO dataset (which also contains traffic light image category).
