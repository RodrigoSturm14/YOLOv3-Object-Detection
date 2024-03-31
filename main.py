import os
import cv2
import numpy as np
from util import get_outputs, draw, NMS

# define constants
model_cfg_path = os.path.join('.', 'model', 'cfg', 'yolov3.cfg')
model_weights_path = os.path.join('.', 'model', 'weights', 'yolov3.weights')
class_names_path = os.path.join('.', 'model', 'class.names')

img_path =  os.path.join('.', 'assets', 'messi_ball.jpg')

# open class.names, read each line and save it into list class_names
class_names = []
with open(class_names_path, 'r') as file:
  class_names = [ line.strip() for line in file.readlines() ]

# load model
net = cv2.dnn.readNet(model_cfg_path, model_weights_path)
layer_names = net.getLayerNames()
output_layers = [ layer_names[i - 1] for i in net.getUnconnectedOutLayers() ] # get output layers names
colors = np.random.uniform(0, 255, size=(len(class_names), 3))

# load and resize image
img = cv2.imread(img_path)
img = cv2.resize(img, None, 0.4, 0,4)
H, W, channels = img.shape

# convert image for the Deep Neural Network(dnn)
blop_img = cv2.dnn.blobFromImage(img, scalefactor=1/255, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)

# get detections
net.setInput(blop_img) # input the blop image into the neural network
detections = net.forward(output_layers)

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
  class_id = np.argmax(detection[5:]) # save the INDEX of the heigher detection; the index count of detection[5:] will start at 0, not at 5, so if the first detection is a person, it will be saved a 0, which is the index from 'detection[5:]'
  score = np.amax(detection[5:]) # save the VALUE of the heigher detection

  # append all the collected data to the general lists
  bboxes.append(bbox)
  class_ids.append(class_id)
  scores.append(score)

# aply NMS; subtract the unnecessary detections for the same object
bboxes, class_ids, scores = NMS(bboxes, class_ids, scores)

# plot the image with bbox and name object
for bbox_, bbox in enumerate(bboxes):
  name_position = class_ids[bbox_]
  name = class_names[name_position]
  print(name)

  xc, yc, w, h = bbox

  img = cv2.rectangle(img, 
                      (int(xc - (w / 2)),  (int(yc - (h / 2)))), 
                      (int(xc + (w / 2)),  (int(yc + (h / 2)))), 
                      (255, 0, 0), 3)
  
  img = cv2.putText(img, '{}'.format(name),  
                    (int(xc - (w / 2)),  int(yc - (h / 2)) - 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (58, 193, 242), 2)

# plotting the imgage with the results
# for bbox_, bbox in enumerate(bboxes):
  # getting the names of the objects detected
  # position_class_id = class_ids[bbox_]
  # object_name = class_names[position_class_id]

  # if(object_name =="sports ball"):
  #   print("yes")
  #   img = draw(bbox, img)

cv2.imshow('img', img)
cv2.waitKey(0)

