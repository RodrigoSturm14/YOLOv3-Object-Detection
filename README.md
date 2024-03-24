# YOLOv3 and OpenCV Object Detection

<div id="header-image" aling="center">
  <img src="https://assets.website-files.com/5f6bc60e665f54db361e52a9/5f6bc60e665f546a6b1e5400_logo_yolo.png" width="300"/>
</div>

## Index

* [Description](#description)
* [Demo and overview](#demo-and-overview)
* [Notes](#notes)

# Description

Object detection system mean to locate, track and classify the objects detected from a single image, using a pre-trained model from `YOLOv3` and `OpenCV` to plot the images.

# Demo and overview
Pre-trained YOLOv3 model detecting a person and a football ball

| Input image | Output image |
|         :---:            |          :---:         |
| <img src="https://github.com/RodrigoSturm14/YOLOv3-Object-Detection/assets/105557226/41d56b11-09ae-4b81-a04e-de04a030a466" width="300"/> | <img src="https://github.com/RodrigoSturm14/YOLOv3-Object-Detection/assets/105557226/d9dbe6f5-6e42-4fd2-bc88-784740db901e" width="300"/> |

When applying NMS (non maxima suppresion), the overlapping and unnecessary detections on a single object are discarded, that is the reason why in the third image only the football ball bbox is drawn. 

| Input image | Output image, not applying NMS | Output image, applying NMS |
|         :---:            |          :---:         |          :---:         |
| <img src="https://github.com/RodrigoSturm14/YOLOv3-Object-Detection/assets/105557226/605c4a2e-5cb2-4aa4-87b3-33238879b92e" width="300"/> | <img src="https://github.com/RodrigoSturm14/YOLOv3-Object-Detection/assets/105557226/724816e2-952d-4d98-95e6-9cae3d5afb90" width="300"/> | <img src="https://github.com/RodrigoSturm14/YOLOv3-Object-Detection/assets/105557226/ed0279a7-1e50-47cb-a4f0-3d3003c0ab74" width="300"/> |

> _As you can see, this version of YOLO is not 100% accurate since at the third image the football ball was classified as a person._

>[!NOTE]
> However, the goal is to track both person and sports ball classes even though they are overlapped, and eliminate the unnecessary detections for both objects.

# Notes
This repo is currently active and it will be updated with the lastest upgrades.

Leave a ⭐ if you are interested!
