import os
import ast
import random
import numpy as np
from tqdm.notebook import tqdm
tqdm.pandas()
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

import matplotlib.patches as patches
from matplotlib.pyplot import figure
from shapely.geometry import Polygon
import seaborn as sns

import cv2
import copy

###

def plot_image_and_bboxes(img, bboxes):
    fig, ax = plt.subplots(1, figsize=(10, 8))
    ax.axis('off')
    ax.imshow(img)
    
    for bbox in bboxes:
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=2, edgecolor='r', facecolor="none")
        ax.add_patch(rect)
    
    plt.show()

def get_image(img_name):
    out_image = cv2.imread(img_name)
    #out_image = cv2.cvtColor(out_image, cv2.COLOR_BGR2RGB)
    return out_image
    #return np.array(Image.open(img_name))
###


REMOVE_NOBBOX = True # remove images with no bbox
ROOT_DIR      = 'C:\train-area\images'
IMAGE_SPLIT   = 4 # Image split : (eg : 2, 4...)
WIDTH         = 1280
HEIGHT        = 720
# Blue color in BGR
color1 = (255, 0, 0)
color2 = (0, 255, 0)
# Line thickness of 2 px
thickness = 2

print('All image size will be : ('+str(HEIGHT//IMAGE_SPLIT)+', '+str(WIDTH//IMAGE_SPLIT)+')')

df = pd.read_csv('../../punk/reef/output/df_all.csv')

df['annotations'] = df['annotations'].progress_apply(lambda x: ast.literal_eval(x))

df['nb_bbox'] = df['annotations'].progress_apply(lambda x: len(x))
data = (df.nb_bbox>0).value_counts(normalize=True)*100
print(f"No BBOX: {data[0]:0.2f}% | With BBOX: {data[1]:0.2f}%")

#print(df.head())


def get_path(row):
    row['old_image_path'] = f'{ROOT_DIR}/{row.image_id}.jpg'
    return row

df = df.progress_apply(get_path, axis=1)

def get_bbox(annots):
    bboxes = [list(annot.values()) for annot in annots]
    return bboxes

df['bboxes'] = df.annotations.progress_apply(get_bbox)

#print(df.tail())

#######################################################################

## til 7 * 7 stride 0.5
# 此函数只输出稀疏表示，为求通用
def cal_tile_box(h, w, s_h, s_w, stride, bbox, IOU_THREAHOLD = 0.2, img_name=""):

    stride_h = int(s_h*stride)
    stride_w = int(s_w*stride)

    tileindexlist = []
    x1 = bbox[0]
    y1 = bbox[1]
    x2 = bbox[0] + bbox[2]
    y2 = bbox[1] + bbox[3]

    #  被影响的 tile index
    tileindex = x1//stride_w - 1 if x1//stride_w - 1 > 0 else 0, \
                y1//stride_h -1 if y1//stride_h - 1 > 0 else 0, \
                x2//stride_w if x2//stride_w < w/stride_w - 1 else int(w/stride_w) - 2,\
                y2//stride_h if y2//stride_h < h/stride_h - 1 else int(h/stride_h) - 2
    #print(tileindex)

    # 计算切分后的 box
    for x in range(tileindex[0],tileindex[2] + 1, 1):
        for y in range(tileindex[1],tileindex[3] + 1, 1):
            # hit border cut edge
            stridebbox = [
                x1 - x * stride_w if x1 - x * stride_w > 0 else 0, # "x1"
                y1 - y * stride_h if y1 - y * stride_h > 0 else 0, # "y1"
                x2 - x * stride_w if x2 - x * stride_w  < s_w else s_w, # "x2"
                y2 - y * stride_h if y2 - y * stride_h < s_h else s_h  # "y2"
            ]      
            # remove too small box  
            if ((stridebbox[2]-stridebbox[0])*(stridebbox[3]-stridebbox[1]) >= bbox[2] * bbox[3] * IOU_THREAHOLD):
                tileindexlist.append({
                    "x":x, 
                    "y":y,
                    "bbox": stridebbox,
                    "img_name": img_name,
                    "annotation": {
                        "x":stridebbox[0],
                        "y":stridebbox[1],
                        "width":stridebbox[2]-stridebbox[0],
                        "height":stridebbox[3]-stridebbox[1],
                    }
                })
    #print(tileindexlist)
    return tileindexlist

# just for visual test
def image_tiler(img, bboxes, s_h=180, s_w=320, img_name="", vis=False):
    stride_h = int(s_h/2)
    stride_w = int(s_w/2)
    tiles = [img[x:x+s_h,y:y+s_w] for x in range(0,img.shape[0]-stride_h,stride_h) for y in range(0,img.shape[1]-stride_w,stride_w)]
    bboxList = []

    y_lens = (img.shape[1]-stride_w) / stride_w            
    for bbox in bboxes:
        
        # 0.2 next
        tileindexlistwithbbox = cal_tile_box(h=img.shape[0], w=img.shape[1], s_h=180, s_w=320, stride=0.5, bbox=bbox, img_name=img_name)
        bboxList.append(tileindexlistwithbbox)
        if vis == True:
            # 0 first
            tileindexlistwithbboxAll = cal_tile_box(h=img.shape[0], w=img.shape[1], s_h=180, s_w=320, stride=0.5, bbox=bbox, IOU_THREAHOLD=0)        
            for tilebox in tileindexlistwithbboxAll:
                curImg = copy.deepcopy(tiles[int(tilebox["y"]*y_lens+tilebox["x"])])
                tiles[int(tilebox["y"]*y_lens+tilebox["x"])] = cv2.rectangle(curImg, (tilebox["bbox"][0], tilebox["bbox"][1]), (tilebox["bbox"][2], tilebox["bbox"][3]), color1, thickness)            
            # 0.2 next
            for tilebox in tileindexlistwithbbox:
                curImg = copy.deepcopy(tiles[int(tilebox["y"]*y_lens+tilebox["x"])])
                tiles[int(tilebox["y"]*y_lens+tilebox["x"])] = cv2.rectangle(curImg, (tilebox["bbox"][0], tilebox["bbox"][1]), (tilebox["bbox"][2], tilebox["bbox"][3]), color2, thickness)

            

    return tiles, bboxList


################################## test ######################################

"""
old_image_path = df['old_image_path'].tolist()
#img_name   = random.choice(old_image_path)
#print(img_name)

#img_name = '../../reef-data/out/all/images/2-5824.jpg'
img_name = '../../reef-data/out/all/images/0-9277.jpg'

img= get_image(img_name)
bboxes = df.loc[df["old_image_path"] == img_name]["bboxes"].values[0]

plot_image_and_bboxes(img, bboxes)

tiles, bboxes  = image_tiler(img, bboxes, s_h = HEIGHT//IMAGE_SPLIT, s_w = WIDTH//IMAGE_SPLIT, img_name=img_name, vis=True)

_, axs = plt.subplots(IMAGE_SPLIT*2-1, IMAGE_SPLIT*2-1, figsize=(20, 16))
axs = axs.flatten()
for img, ax in zip(tiles, axs):
    ax.axis('off')
    ax.imshow(img)
plt.show()

print(bboxes)

"""


############################### slice all images #######################

# Calling DataFrame constructor

#"""
slices_row_lens = IMAGE_SPLIT *2-1
OUTPUTPATH = "C:\train-area\slices"
_namelist = []
_outpathlist = []
_annoslist = []
for i, row in tqdm(df.iterrows(), total = len(df)):
    img_path = row['old_image_path']
    bboxes = row['bboxes']
    img= get_image(img_path)
    tiles, bboxes = image_tiler(img, bboxes, s_h = HEIGHT//IMAGE_SPLIT, s_w = WIDTH//IMAGE_SPLIT, img_name=row["image_id"])
    annos = []
    for x in range(len(tiles)):
        _name = row["image_id"] + "-" + str(x // slices_row_lens) + "-" + str(x % slices_row_lens) + ".jpg"
        _namelist.append(_name)
        _outpath = OUTPUTPATH + _name
        _outpathlist.append(_outpath)
        annos.append([])
        cv2.imwrite(_outpath, tiles[x])
        #_img = Image.fromarray(tiles[x])
        #_img.save(_outpath)
    for bbox in bboxes:
        for sliced_bbox in bbox:
            annos[sliced_bbox["y"]*slices_row_lens + sliced_bbox["x"]] = [sliced_bbox["annotation"]]
    _annoslist += annos
    #if i > 6:
    #    break

slices_df = pd.DataFrame({'name': _namelist,
                   'path': _outpathlist,
                   'annotations': _annoslist})

slices_df.to_csv("./slices_df.csv")
#"""
