import numpy as np
import pandas as pd
import ast

from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2

def plot_image_and_bboxes(img, bboxes):
    fig, ax = plt.subplots(1, figsize=(10, 8))
    ax.axis('off')
    ax.imshow(img)
    
    for bbox in bboxes:
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=2, edgecolor='r', facecolor="none")
        ax.add_patch(rect)
    
    plt.show()

def get_image(img_name):
    return np.array(Image.open(img_name))

def get_bbox(annots):
    bboxes = [list(annot.values()) for annot in annots]
    return bboxes


df = pd.read_csv('./slices_df.csv')
df['annotations'] = df['annotations'].apply(lambda x: ast.literal_eval(x))
df['bboxes'] = df.annotations.apply(get_bbox)

imglist = []
for i, row in df.iterrows():
    img = get_image(row["path"])
    bboxes = row["bboxes"]
    if bboxes != []:
        for bbox in bboxes:
            img = cv2.rectangle(img, (bbox[0],bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (255, 0, 0), 2)
    imglist.append(img)
    if (i+1) % 49 == 0:
        print(i, len(imglist))
        _, axs = plt.subplots(7, 7, figsize=(32, 32))
        axs = axs.flatten()
        for img, ax in zip(imglist, axs):
            ax.axis('off')
            ax.imshow(img)
        print(row)
        plt.show()
        imglist = []