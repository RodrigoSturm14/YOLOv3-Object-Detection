import os
import cv2
import numpy as np
from util import get_outputs, draw, NMS

# define constants
model_cfg_path = os.path.join('.', 'model', 'cfg', 'yolov3.cfg')
model_weights_path = os.path.join('.', 'model', 'weights', 'yolov3.weights')
class_names_path = os.path.join('.', 'model', 'class.names')

img_path =  os.path.join('.', 'assets', 'partido_soccer_video.mp4')

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
#img = cv2.imread(img_path)
# img = cv2.resize(img, None, fx=0.4, fy=0.4)

# load video
img = cv2.VideoCapture(img_path)

ret = True
while ret:
  ret, frame = img.read()

  if ret:
    frame = cv2.resize(frame, None, fx=0.4, fy=0.4)
    H, W, channels = frame.shape

    # convert image for the Deep Neural Network(dnn)
    blop_img = cv2.dnn.blobFromImage(frame, scalefactor=1/255, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)

    # get detections
    net.setInput(blop_img) # input the blop image into the neural network
    detections = net.forward(output_layers)

    # create and save boundingbox, classid(name of the object detected), confidence(score) of every detection
    # from yolov3
    bboxes = []
    scores = []
    class_ids = []

    for detection in detections:
      for output in detection:
        confs = output[5:]
        class_id = np.argmax(confs)
        conf = confs[class_id]
        if conf > 0.6:
          center_x = int(output[0] * W)
          center_y = int(output[1] * H)
          w = int(output[2] * W)
          h = int(output[3] * H)
          x = int(center_x - (w/2))
          y = int(center_y - (h/2))
          # append all the collected data to the general lists
          bboxes.append([x, y, w, h])
          scores.append(float(conf))
          class_ids.append(class_id)

    # aply NMS; subtract the unnecessary detections for the same object
    indexes = cv2.dnn.NMSBoxes(bboxes=bboxes, scores=scores, score_threshold=0.5, nms_threshold=0.4)
    #print(indexes)
    # plot the image with bbox and name object
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(bboxes)):
      if i in indexes:
        x, y, w, h = bboxes[i]
        label = str(class_names[class_ids[i]])
        color = colors[i]
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 8), font, 1, color, 1)

    # plotting the imgage with the results
    # for bbox_, bbox in enumerate(bboxes):
      # getting the names of the objects detected
      # position_class_id = class_ids[bbox_]
      # object_name = class_names[position_class_id]

      # if(object_name =="sports ball"):
      #   print("yes")
      #   img = draw(bbox, img)

    cv2.imshow('frame', frame)
    if cv2.waitKey(40) & 0xFF == ord('q') :
      break

img.release()
cv2.destroyAllWindows()
