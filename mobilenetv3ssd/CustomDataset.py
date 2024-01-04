import os, sys
from random import randrange
import numpy as np
import torch
from PIL import Image
import csv
from settings.demo7_conf import settings

class CustomDataset(torch.utils.data.Dataset):
    class_names = ["call", "dislike", "fist", "four", "like", "mute", "ok", "one", "palm", "peace", "peace_inverted", "rock", "stop", "stop_inverted", "three", "three2", "two_up", "two_up_inverted", "no_gesture"]

    def __init__(self, settings, mode="train", distance=None, transforms=None):
        self.settings = settings
        self.label_path = str()
        self.imgs_count = 0
        self.csv_file = None
        self.csv_reader = None
        
        if mode == "val" :
            self.label_path = os.path.join(self.settings.dataset, "demo7_rgb_val.csv")
            if settings.modality == "depth":
                self.label_path = os.path.join(self.settings.dataset, "demo7_depth_val.csv")
        elif mode == "test" :
            self.label_path = os.path.join(self.settings.dataset, "demo7_rgb_test.csv") 
            if distance is not None:
                self.label_path = os.path.join(self.settings.dataset, "demo7_rgb_test_{}.csv".format(distance))

            if settings.modality == "depth":
                self.label_path = os.path.join(self.settings.dataset, "demo7_depth_test.csv")
                if distance is not None:
                    self.label_path = os.path.join(self.settings.dataset, "demo7_depth_test_{}.csv".format(distance))
            
        else:
            self.label_path = os.path.join(self.settings.dataset, "demo7_rgb_train.csv")
            if settings.modality == "depth":
                self.label_path = os.path.join(self.settings.dataset, "demo7_depth_train.csv")

        self.transforms = transforms
        self.csv_file = open(self.label_path, 'r')
        self.csv_reader = list(csv.reader(self.csv_file, delimiter=';'))
        #self.imgs_count = len(self.imgs_list)

    def get_coord_bbox(self, arr):
        xmin = arr[0]
        ymin = arr[1]
        xmax = arr[2]
        ymax = arr[3]
        return (xmin, ymin, xmax, ymax)


    def __getitem__(self, idx):
        # load images
        idx += 1
        boxes = []
        labels = []
        num_objs = len(list(self.csv_reader)) - 1
        
        row = self.csv_reader[idx]
        img = Image.open(os.path.join(self.settings.dataset,  row[0]))
        
        raw_bbox = row[1][2:-2]
        #CHECK IF MORE THAN ONE BBOX
        if raw_bbox.find("], [") > 0:
            temp = raw_bbox.split("], [")
            for x in temp:
                boxes.append(self.get_coord_bbox(np.array(x.split(", ")).astype(np.float_)))
        else:
            boxes.append(self.get_coord_bbox(np.array(raw_bbox.split(", ")).astype(np.float_)))
        
        raw_label = row[2][1:-1].replace("'", "").split(", ")
        for x in raw_label:
            labels.append(int(x))

        #CODE TO CROP RECTANGULARE IMG IN SQUARED IMG (EX. FROM 640X480 TO 480X480)
        width, height = img.size
        crop_size = min(img.size)
        new_height = min(height, crop_size)
        new_width = min(width, crop_size)

        if new_height != height or new_width != width: #IF THE CROPPING IS NEEDED
            offset_height = max(height - crop_size, 0)
            offset_width = max(width - crop_size, 0)

            #GET THE TOP AND LEFT COORDINATES OF A CENTRAL CROP
            top = int(offset_height/2)
            left = int(offset_width/2)

            
            globalxmin = 100000
            globalymin = 100000
            globalxmax = -1
            globalymax = -1

            #GET THE GLOBAL COORDINATES OF ALL BBOXES, THAT IS TO CREATE A SINGLE BIGGER BBOX IF TWO BBOX ARE PRESENT
            for box in boxes:
                if box[0] < globalxmin:
                    globalxmin = box[0]
                if box[1] < globalymin:
                    globalymin = box[1]
                if box[2] > globalxmax:
                    globalxmax = box[2]
                if box[3] > globalymax:
                    globalymax = box[3]
            
            #IF THE BIGGER BBOX FOUNDED IS LARGER THAN THE SIZE OF CROPPING, THE CROP CAN'T CONTAIN BOTH BBOX. 
            if globalxmax - globalxmin > new_width or globalymax - globalymin > new_height:
                # print("skip image")
                # print(row[0])
                img = None
                #skipped_img += 1
 
            #DECIDE IF THE CROP CAN BE PERFORMED CENTRALLY OR MUST BE MOVED TO CONTAIN BOTH BBOXES
            if offset_width > 0:
                if globalxmin - left < 0:
                    left = globalxmin
                elif globalxmax - (left + new_width) > 0 :
                    left += globalxmax - (left + new_width)
            if offset_height > 0:
                if globalymin - top < 0:
                    top = globalymin
                elif globalymax - (top + new_height) > 0 :
                    top += globalymax - (top + new_height)

            #print("Final crop")
            #print(top, left, new_width, new_height)
            img_cropped = img.crop((left, top, left + new_height, top + new_width))
            #img.save("original_custom.png")
            #img_cropped.save("cropped_custom.png")

            cropped_box = []
            for box in boxes:
                cropped_box.append((box[0] - left, box[1] - top, box[2] - left, box[3] - top))
            boxes = cropped_box


        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([idx])

        if self.transforms is not None:
            img_cropped, target = self.transforms(img_cropped, target)

        return img_cropped,target


    def __len__(self):
      return len(list(self.csv_reader)) - 1


#"""
#dataset = CustomDataset(settings=settings, mode="train")
dataset = CustomDataset(settings=settings, mode="train", distance=None, transforms=None)
id = randrange(3072, 3286)
id = 65860
print(dataset[id][0]._size)
#"""
