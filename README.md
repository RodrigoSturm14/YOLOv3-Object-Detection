# YOLOv3 and OpenCV Object Detection

<div id="header-image" aling="center">
  <img src="https://assets.website-files.com/5f6bc60e665f54db361e52a9/5f6bc60e665f546a6b1e5400_logo_yolo.png" width="300"/>
</div>

## Index

* [Description](#description)
* [Demo and overview](#demo-and-overview)
* [Improvement areas](#improvement-areas)
  * [To-Do list](#to-do-list)
* [Notes](#notes)

# Description

Object detection system mean to locate, track and classify the objects detected from a single image, using a pre-trained model from `YOLOv3` and `OpenCV` to plot the images.

# Demo and overview
Pre-trained YOLOv3 model detecting a person and a football ball

| Input image | Output image |
|         :---:            |          :---:         |
| <img src="https://github.com/RodrigoSturm14/YOLOv3-Object-Detection/assets/105557226/41d56b11-09ae-4b81-a04e-de04a030a466" width="300"/> | <img src="https://github.com/RodrigoSturm14/YOLOv3-Object-Detection/assets/105557226/d9dbe6f5-6e42-4fd2-bc88-784740db901e" width="300"/> |

When applying NMS (non maxima suppresion), the overlapping and unnecessary detections on a single object are discarded, leaving only one bbox for every object detected. That is the reason why in the third image only the football player and the football ball bbox is drawn. 

| Input image | Output image, not applying NMS | Output image, applying NMS |
|         :---:            |          :---:         |          :---:         |
| <img src="https://github.com/RodrigoSturm14/YOLOv3-Object-Detection/assets/105557226/605c4a2e-5cb2-4aa4-87b3-33238879b92e" width="300"/> | <img src="https://github.com/RodrigoSturm14/YOLOv3-Object-Detection/assets/105557226/fea1d127-7c82-4f52-a8bf-5729464b9727" width="300"/> | <img src="https://github.com/RodrigoSturm14/YOLOv3-Object-Detection/assets/105557226/338342db-2113-430a-a74f-d737a5495c3e" width="300"/> |

Futhermore this system is able to track and detect several objects of the same class. 

| Input image | Output image |
|         :---:            |          :---:         |
| <img src="https://github.com/RodrigoSturm14/YOLOv3-Object-Detection/assets/105557226/70685d59-cf10-4758-8686-aeca83d05332" width="500"/> | <img src="https://github.com/RodrigoSturm14/YOLOv3-Object-Detection/assets/105557226/3c06d8aa-1a4d-45fe-ae37-06ad22b03e4a" width="500"/> |

> [!NOTE]
> _The system can also use the same principles to detect objects but on videos. However, the pre-trained model takes quite a while to output every frame. The speed detection will depend how powerfull the CPU is._

# Improvement areas
### To-Do list
- [x] ~Improve the NMS. At the moment, the NMS function applied delete every overlapping bbox, when it should be deleting just the unnecesary detections for each object and be able to track two overlapped object.~
- [x] ~Implement the pre-trained model to a video.~
- [ ] Apply a fine tune. The next goal is to track only a specific object class that are meant to be tracked and avoid tracking and detecting other class objects by applying a fine tune to the pre-trained model.

# Notes
This repo is currently active and it will be updated with the lastest upgrades.

Leave a ⭐ if you are interested!
