import math
import sys
import time
import torch
import utils
import cv2
import torchvision.transforms as T
import numpy as np
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchmetrics import Accuracy, F1Score, ConfusionMatrix
from torch.profiler import profile, record_function, ProfilerActivity
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from ensemble_boxes import weighted_boxes_fusion
from commands import DEMO7_GESTURES



@torch.inference_mode()
def calculate_and_save_preds(model, data_loader, device, dataset_name):
    show = False
    verbose = False
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"
    correct = 0
    total_detections = 0
    BBOX_THRESHOLD = 0.1
    MAX_BBOX_DETECTABLE = 2

    
    all_predictions = []

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        with torch.no_grad():
            outputs = model(images)
        # print("*******************")
        # print(outputs)
        # print("*******************")

        for output, target, imgs in zip(outputs, targets, images):

            boxes3 = target['boxes']

            boxes3 = boxes3.to(torch.float).tolist()

            pred_detections = torch.count_nonzero(output["scores"] > BBOX_THRESHOLD).item()

            if pred_detections > 0:
                output = reduce_size_tensor(output, pred_detections)
                index_argmin = torch.argmin(output["labels"]).item()  #check if a valid gesture is detected
                if  index_argmin != 13:
                    #If valid gesture is detected, we take the index of valid gesture [1,12] with higher scores
                    index_valid_gesture = torch.argmax((torch.where(output["labels"]< 13, output["scores"], 0.)))
                    output = {"boxes":output["boxes"][index_valid_gesture], "labels": output["labels"][index_valid_gesture], "scores": output["scores"][index_valid_gesture]}
                else:
                    output = reduce_size_tensor(output, MAX_BBOX_DETECTABLE) #take the no gesture with higher score

            else:
                all_predictions.append({"boxes" : [[0.,0.,0.,0.]], "labels" : 13, "scores" : 0}) 
                continue

     
            bbox = output["boxes"].tolist()
            #test_outputs.append(output)
            normalized_bbox = []
            normalized_bbox.append([bbox[0]/480, bbox[1]/480, bbox[2]/480, bbox[3]/480])
            

            all_predictions.append({"boxes" : normalized_bbox, "labels" : output["labels"].tolist(), "scores" : output["scores"].tolist()}) 

    torch.save({"depth" : all_predictions}, "saves/all_pred_for_fusion/pretrained/depth_5m.pt")
    #torch.save({"rgb" : all_predictions}, "saves/all_pred_for_fusion/pretrained/rgb_5m.pt")
    
    return {}


       

def reduce_size_tensor (tensor, size):
    return {k: v[:size] for k, v in tensor.items()}

def reduce_size_tensor_old (tensor, size):
    return {"boxes":tensor["boxes"][:size], "labels": tensor["labels"][:size], "scores": tensor["scores"][:size] }


def weighted_fusion (dataset):

    show = False
    show_original = False
    rgb = torch.load("saves/all_pred_for_fusion/pretrained/rgb_2m.pt")
    depth = torch.load("saves/all_pred_for_fusion/pretrained/depth_2m.pt")
    rgb = rgb['rgb']
    depth = depth['depth']
    count = 0
    MAX_BBOX_DETECTABLE = 1
    BBOX_THRESHOLD = 0.2
    accuracy = Accuracy(num_classes=len(DEMO7_GESTURES), average= "micro")
    accuracy_weighted = Accuracy(num_classes=len(DEMO7_GESTURES), average= "macro")
    accuracy_for_classes = Accuracy(num_classes=len(DEMO7_GESTURES), average= "none")
    confmat = ConfusionMatrix(num_classes=len(DEMO7_GESTURES))

    assert len(dataset) == len(rgb) == len(depth)

    all_pred_labels = []
    all_targets_labels = []

    for pred_rgb, pred_depth in zip(rgb, depth):

        boxes_list = []
        labels_list = []
        scores_list = []
        if show_original:
            print("rgb")
            print(pred_rgb['boxes'])
            print(pred_rgb['labels'])
            print(pred_rgb['scores'])

            print("depth")
            print(pred_depth['boxes'])
            print(pred_depth['labels'])
            print(pred_depth['scores'])

        boxes_list.append(pred_rgb['boxes'])
        boxes_list.append(pred_depth['boxes'])
        labels_list.append([pred_rgb['labels']])        
        labels_list.append([pred_depth['labels']])
        scores_list.append([pred_rgb['scores']])
        scores_list.append([pred_depth['scores']])
        if show_original:
            print(boxes_list)
            print(labels_list)
            print(scores_list)

        weights = [2, 1]
        #iou_thr = 0.43
        iou_thr = 0.50
        skip_box_thr = 0.0800
        sigma = 0.1

        boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
        #print(boxes, scores, labels)


        
        images, targets = dataset[count]

        #targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        
        if show:
            cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('image', 848, 640)
            color = (0, 0, 255)
            color2 = (255, 0, 0) 
            color3 = (0, 255 , 0) 
            transform = T.ToPILImage()
            img = transform(images)
            mat = np.array(img)
            mat = cv2.cvtColor(mat, cv2.COLOR_BGR2RGB)
            if show_original:
                for x in range(len(pred_rgb['boxes'])):
                    xmin_r = int(pred_rgb['boxes'][x][0]*480)
                    xmin_d = int(pred_depth['boxes'][x][0]*480)
                    
                    ymin_r = int(pred_rgb['boxes'][x][1]*480)
                    ymin_d = int(pred_depth['boxes'][x][1]*480)
                    
                    xmax_r = int(pred_rgb['boxes'][x][2]*480)
                    xmax_d = int(pred_depth['boxes'][x][2]*480)
                    
                    ymax_r = int(pred_rgb['boxes'][x][3]*480)
                    ymax_d = int(pred_depth['boxes'][x][3]*480)

                    cv2.rectangle(mat, (xmin_r, ymin_r), (xmax_r, ymax_r), color, 1)
                    cv2.rectangle(mat, (xmin_d, ymin_d), (xmax_d, ymax_d), color2, 1)
        
        converted_bbox = []
        for x in range(len(boxes)):
            xmin_f = int(boxes[x][0]*480)
            ymin_f = int(boxes[x][1]*480)
            xmax_f = int(boxes[x][2]*480)
            ymax_f = int(boxes[x][3]*480)
            converted_bbox.append([xmin_f, ymin_f, xmax_f, ymax_f])
            #if show:
                #cv2.rectangle(mat, (xmin_f, ymin_f), (xmax_f, ymax_f), color3, 1)
        #print((xmin_r/480, ymin_r/480,xmax_r/480, ymax_r/480))

        pred = {'boxes': torch.as_tensor(converted_bbox), 'labels' : torch.as_tensor(labels), 'scores' : torch.as_tensor(scores)}

        pred_detections = torch.count_nonzero(pred["scores"] > BBOX_THRESHOLD).item()

        output = None

        if len(targets["labels"]) > 1:
            #reduce target tensor to 1
            targets = reduce_size_tensor(targets, 1)
        targ_labels_for_classification = targets["labels"][0]
        
        all_targets_labels.append(targ_labels_for_classification.item())

        if pred_detections > 0:
                pred = reduce_size_tensor(pred, pred_detections)
                index_argmin = torch.argmin(pred["labels"]).item()  #check if a valid gesture is detected
                if  index_argmin != 13:
                    #If valid gesture is detected, we take the index of valid gesture [1,12] with higher scores
                    index_valid_gesture = torch.argmax((torch.where(pred["labels"]< 13, pred["scores"], 0.)))
                    output = {"boxes":pred["boxes"][index_valid_gesture], "labels": pred["labels"][index_valid_gesture], "scores": pred["scores"][index_valid_gesture]}
                else:
                    output = reduce_size_tensor(output, MAX_BBOX_DETECTABLE) #take the no gesture with higher score
        else:
            all_pred_labels.append(13)
            count+= 1
            #print(pred)
            #print(targets)
            continue

        #print("OUTPUT : ".format(output))

        all_pred_labels.append(int(output["labels"].item()))


        """ for label_pred, label_tar in zip(ordered_tensor['labels'], targets['labels']):
            all_pred_labels.append(int(label_pred.item()))
            all_targets_labels.append(label_tar.item()) """

        if show:        

            for x in range(len(targets['boxes'])):
                
                xmin_t = int(targets['boxes'][x][0])
                ymin_t = int(targets['boxes'][x][1])
                xmax_t = int(targets['boxes'][x][2])
                ymax_t = int(targets['boxes'][x][3])
                cv2.rectangle(mat, (xmin_t, ymin_t), (xmax_t, ymax_t), (255,0,255), 1)
                text = "GT: %s" % (int(targets['labels'].item()))
                cv2.putText(mat, text, (xmin_t , ymax_t-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 1)

                
                xmin_rid = int(output['boxes'][0])
                ymin_rid = int(output['boxes'][1])
                xmax_rid = int(output['boxes'][2])
                ymax_rid = int(output['boxes'][3])
                cv2.rectangle(mat, (xmin_rid, ymin_rid), (xmax_rid, ymax_rid), (0,255,255), 1)
                text = "%s" % (int(output["labels"].item()))
                cv2.putText(mat, text, (xmin_rid , ymin_rid - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)      


            
            cv2.imshow('image', mat)
            cv2.waitKey(0)

        
        #print("PREDETTO RIDOTTO: {}".format(pred_reduced))
        #print("RIFERIMENTO: {}".format(targets))

        #all_pred.append(pred_reduced)
        #all_targets.append(targets)

        #print(len(pred['boxes']))          #predicted bbox > than real bbox
        #print(len(targets['boxes']))
        #metric.update([pred], [targets])
        if count % 100 == 0:
            print("{}/{}".format(count, len(rgb)))
        count+= 1

    
    #print(metric.compute())
    all_pred_labels = torch.as_tensor(all_pred_labels)
    all_targets_labels = torch.as_tensor(all_targets_labels)
    print("ACCURACY: {}".format(accuracy(all_pred_labels, all_targets_labels)))
    print("ACCURACY WEIGHTED: {}".format(accuracy_weighted(all_pred_labels, all_targets_labels)))
    print("ACCURACY FOR EACH CLASS: {}".format(accuracy_for_classes(all_pred_labels, all_targets_labels)))
    #conf_matrix = confmat(all_pred_labels, all_targets_labels)
    #torch.save({"confusion_matrix": conf_matrix}, "saves/confusion_matrix/conf_matrix_NoHagrid_rgb_1bbox_rgb_plus_depth.pt")
    #metric2.update(all_pred, all_targets)
    #print(metric2.compute())



