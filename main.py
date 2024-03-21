import os
import cv2
from util import get_outputs

# define constants
model_cfg_path = os.path.join('.', 'model', 'cfg', 'yolov3.cfg')
model_weights_path = os.path.join('.', 'model', 'weights', 'yolov3.weights')
class_names_path = os.path.join('.', 'model', 'class.names')

img_path =  os.path.join('.', 'assets', 'messi_ball.jpg')

# open class.names, read each line and save it into list class_names
with open(class_names_path, 'r') as file:
  class_names = [ j[:-1] for j in file.readlines() if len(j) > 2 ]

# load model
net = cv2.dnn.readNetFromDarknet(model_cfg_path, model_weights_path)

# load image
img = cv2.imread(img_path)

# convert image for the Deep Neural Network(dnn)
blop_img = cv2.dnn.blobFromImage(img, 1 / 255, (320, 320), (0, 0, 0), False)

# get detections
net.setInput(blop_img) # input the blop image into the neural network
detections = get_outputs(net)

for detection in detections:
  print(detection)
  break