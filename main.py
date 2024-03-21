import os
import cv2
import numpy as np
from util import get_outputs, draw

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

H, W, _ = img.shape

# convert image for the Deep Neural Network(dnn)
blop_img = cv2.dnn.blobFromImage(img, 1 / 255, (320, 320), (0, 0, 0), False)

# get detections
net.setInput(blop_img) # input the blop image into the neural network
detections = get_outputs(net)

# create and save boundingbox, classid(name of the object detected), confidence(score) of every detection
# from yolov3
bboxes = []
class_ids = []
scores = []

for detection in detections:
  bbox = detection[:4]
  # converting float coordenates into integers
  xc, yc, w, h = bbox # (xc, yc) center of the bbox
  bbox = [int(xc * W), int(yc * H), int(w * W), int(h * H)]

  bbox_score = detection[4]
  class_id = np.argmax(detection[5:]) # save the INDEX of the heigher detection
  score = np.amax(detection[5:]) # save the VALUE of the heigher detection
  # append all the collected data to the global lists
  bboxes.append(bbox)
  class_ids.append(class_id)
  scores.append(score)
  
# plotting the imgage with the results 
for bbox in bboxes:
  img = draw(bbox, img)

cv2.imshow('img', img)
cv2.waitKey(0)

