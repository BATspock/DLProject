from get_flipped_object import MoveObject
from neural_nexus_clipseg import getMask
import os
import cv2

class_list= ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light', \
                 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', \
                'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', \
                    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', \
                        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',\
                              'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa',\
                                  'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', \
                                    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', \
                                        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']


os.chdir('../SegmentationImagesReconstruction')
for imgs in os.listdir():
    print(imgs)
    img = cv2.imread(imgs)
    maskObj = getMask(class_list, imgs, ['left', 'right', 'up', 'down'], ["dining table", "tv monitor", "potted plant"])

    mask = maskObj.get_clip_mask()

    direction = None

    if 'left' in imgs:
        direction = 'left'
    elif 'right' in imgs:
        direction = 'right'
    elif 'above' in imgs:
        direction = 'above'
    else:
        direction = 'below'

    flip_obj = MoveObject(img, mask, direction=direction)
    translated_img = flip_obj.translate_object()

    cv2.imwrite('translated_' + imgs, translated_img)