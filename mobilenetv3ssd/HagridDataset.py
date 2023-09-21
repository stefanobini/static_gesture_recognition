import os, sys
import numpy as np
import torch
from PIL import Image
import csv

class HagridDataset(torch.utils.data.Dataset):

    root = "/mnt/sda1/rmarchesano/dataset/hagrid_dataset/"
    directory_dataset = ""
    imgs_count = 0
    label_path = ""
    class_names = ["call", "dislike", "fist", "four", "like", "mute", "ok", "one", "palm", "peace", "peace_inverted", "rock", "stop", "stop_inverted", "three", "three2", "two_up", "two_up_inverted", "no_gesture"]
    csv_file = None
    csv_reader = None

    def __init__(self, mode="train",transforms=None):
        
        if mode == "val" or mode == "test":
            self.root = "/mnt/sda1/rmarchesano/dataset/hagrid_test"
            self.directory_dataset = os.path.join(self.root, "final_preprocessed")
            self.label_path = os.path.join(self.directory_dataset, "dataset_hagrid_test_preprocessed.csv")
        else:
            self.directory_dataset = os.path.join(self.root, "final_preprocessed")
            self.label_path = os.path.join(self.directory_dataset, "dataset_hagrid_train_preprocessed.csv")
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
        img = Image.open(os.path.join(self.directory_dataset,  row[0]))
        
        raw_bbox = row[1][2:-2]

        #if row has more than one bbox
        if raw_bbox.find("], [") > 0:
            temp = raw_bbox.split("], [")
            for x in temp:
                boxes.append(self.get_coord_bbox(np.array(x.split(", ")).astype(np.float_)))
        else:
            boxes.append(self.get_coord_bbox(np.array(raw_bbox.split(", ")).astype(np.float_)))
        
        #get label/labels from row
        raw_label = row[2][1:-1].replace("'", "").split(", ")
        for x in raw_label:
            labels.append(int(x))


        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([idx])

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img,target


    def __len__(self):
      return len(list(self.csv_reader)) - 1



""" dataset = HagridDataset(None)
print(dataset[503394])
print(len(dataset)) """
