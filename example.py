#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# MIT License
#
# Copyright (c) 2019 Iván de Paz Centeno
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import os
import cv2
from mtcnn import MTCNN
import glob

detector = MTCNN()

input_dir = "img"


if not os.path.exists(input_dir):
    os.makedirs(input_dir)

img_list =  glob.glob(input_dir + '/' + '*.png')
img_list += glob.glob(input_dir + '/' + '*.jpg')

for img_path in img_list:
    image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    result = detector.detect_faces(image)

    # Result is an array with all the bounding boxes detected. We know that for 'ivan.jpg' there is only one.
    bounding_box = result[0]['box']
    print(result)

    keypoints = result[0]['keypoints']

    # cv2.rectangle(image,
    #               (bounding_box[0], bounding_box[1]),
    #               (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
    #               (0,155,255),
    #               2)
    # cv2.circle(image,(keypoints['left_eye']), 2, (0,155,255), 2)
    # cv2.circle(image,(keypoints['right_eye']), 2, (0,155,255), 2)
    # cv2.circle(image,(keypoints['nose']), 2, (0,155,255), 2)
    # cv2.circle(image,(keypoints['mouth_left']), 2, (0,155,255), 2)
    # cv2.circle(image,(keypoints['mouth_right']), 2, (0,155,255), 2)
    # cv2.imwrite(img_path + "_land_mask.jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    land_mask_file = img_path.replace(".jpg",".txt")
    file_write = open(land_mask_file,"w")
    x, y = keypoints['left_eye']
    file_write.write(f"{x} {y}\n")
    x, y = keypoints['right_eye']
    file_write.write(f"{x} {y}\n")
    x, y = keypoints['nose']
    file_write.write(f"{x} {y}\n")
    x, y = keypoints['mouth_left']
    file_write.write(f"{x} {y}\n")
    x, y = keypoints['mouth_right']
    file_write.write(f"{x} {y}\n")


