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

class_name = ["Background", "Come here", "Go", "Start", "Stop", "Move To R", "Move To L", "Move Up", "Move Up", "Move Down", "Move Down", "Move Forward", "Move Backward", "NoGesture"]


@torch.inference_mode()
def test(model, data_loader, device, dataset_name):
    show = False
    verbose = False
    BBOX_THRESHOLD = 0.2
    MAX_BBOX_DETECTABLE = 1
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"
       
    accuracy = Accuracy(num_classes=14, average= "micro")
    accuracy_weighted = Accuracy(num_classes=14, average= "macro")
    accuracy_for_classes = Accuracy(num_classes=14, average= "none")
    confmat = ConfusionMatrix(num_classes=14)


    
    all_pred = []
    all_targ = []

    count = 0

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        # print("TARGETS")

        if count == 900:
            break

       
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        with torch.no_grad():
            outputs = model(images)

        
        if show:
            #cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            #cv2.resizeWindow('image', 848, 640)
            color = (255, 0, 255)
            color2 = (0, 255, 0)   


        for output, target, imgs in zip(outputs, targets, images):

            boxes3 = target['boxes']

            boxes3 = boxes3.to(torch.float).tolist()

            pred_detections = torch.count_nonzero(output["scores"] > BBOX_THRESHOLD).item()

            if show:
                transform = T.ToPILImage()
                img = transform(imgs.cpu())
                mat = np.array(img)
                mat = cv2.cvtColor(mat, cv2.COLOR_BGR2RGB)

            if len(target["labels"]) > 1:
                target = reduce_size_tensor(target, MAX_BBOX_DETECTABLE)
            targ_labels_for_classification = target["labels"][0]
            all_targ.append(targ_labels_for_classification.item())
            
            #check if at least one prediction is > threshold
            if pred_detections > 0:
                output = reduce_size_tensor(output, pred_detections)
                index_argmin = torch.argmin(output["labels"]).item()  #check if a valid gesture is detected
                if  index_argmin != 13:
                    #If valid gesture is detected, we take the index of valid gesture [1,12] with higher scores
                    index_valid_gesture = torch.argmax((torch.where(output["labels"]< 13, output["scores"], 0.)))
                    output = {"boxes":output["boxes"][index_valid_gesture], "labels": output["labels"][index_valid_gesture], "scores": output["scores"][index_valid_gesture]}
                #if argmin == 13, only no gesture are detected
                else:
                    output = reduce_size_tensor(output, MAX_BBOX_DETECTABLE) #take the no gesture with higher score

            #no prediction, skip frame e set "no-gesture"
            else:
                all_pred.append(13)
                continue

            out_labels_for_classification = output["labels"]
            out_bbox_for_classification = output["boxes"]
            all_pred.append(out_labels_for_classification.item())
            

            '''
            ########## DETECTION METRICS MANIPULATION ##########

            #if have the same lenght, set only the labels to 1 (hand)
            if len(target["labels"]) == len(output["labels"]):
                labels_output = []
                target_output = []
                for x in range(len(target["labels"])):
                    labels_output.append(1)
                    target_output.append(1)
                output["labels"] = torch.as_tensor(labels_output).to(device)
                target["labels"] = torch.as_tensor(target_output).to(device)
            
            else:
                
                #increase the size of output tensor [0] to equal the lenght of target [1]
                if pred_detections == 0:
                    labels_new = []
                    scores_new = []                    
                    output["boxes"] = torch.cat((output["boxes"], torch.as_tensor([[0,0,0,0]]).to(device)))
                    labels_new.append(0)
                    scores_new.append(0.)
                    output["labels"] = torch.as_tensor(labels_new).to(device)
                    output["scores"] = torch.as_tensor(scores_new).to(device)


                #increase the size of target to equal the lenght of output
                # ONE CASE POSSIBLE: only if the gt has 1 detection instead of two, the max possible
                if len(target["labels"]) < len(output["labels"]):
                    temp_box = []
                    temp_labels = []
                    temp_box.append(target["boxes"][0])
                    temp_box.append(torch.as_tensor([0,0,0,0]).to(device))
                    temp_labels.append(1)
                    temp_labels.append(0)
                    target = {"boxes": torch.stack(temp_box), "labels" : torch.as_tensor(temp_labels).to(device)}

                    output["labels"] = torch.as_tensor([1,1]).to(device)
                    '''

            if verbose:
                print("output : {}".format(out_labels_for_classification))
                print("gt     : {}".format(targ_labels_for_classification))

            if show:   
                #display the predicted label 
                if pred_detections > 0:
                    label = class_name[out_labels_for_classification.item()]
                    xmin = int(out_bbox_for_classification[0].item())
                    ymin = int(out_bbox_for_classification[1].item())
                    xmax = int(out_bbox_for_classification[2].item())
                    ymax = int(out_bbox_for_classification[3].item())
                    
                    
                    text = "%s (%s)" % (label, round(output["scores"].item()*100, 2))
                    cv2.putText(mat, text, (xmin, ymin - 3), cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 1)
                    cv2.rectangle(mat, (xmin, ymin), (xmax, ymax), color, 1)
                
                #Display GT 
                #xmingt = int(target["boxes"][0])
                #ymingt = int(target["boxes"][1])
                #xmaxgt = int(target["boxes"][2])
                #ymaxgt = int(target["boxes"][3])
                #cv2.rectangle(mat, (xmingt, ymingt),(xmaxgt, ymaxgt), color2, 1)
                text = "GT: %s" % (class_name[target["labels"].item()])
                #cv2.putText(mat, text, (xmin, ymin - 20), cv2.FONT_HERSHEY_DUPLEX, 0.6, color2, 1)
                
                
                cv2.imwrite("imgs_thesis/test_rgb{}.jpg".format(count), mat)
                #cv2.imwrite("imgs_thesis_depth/test_depth{}.jpg".format(count), mat)
                #cv2.imshow('image', mat)
                #cv2.waitKey(70)    

            count += 1
    cv2.destroyAllWindows()


    all_pred = torch.as_tensor(all_pred)
    all_targ = torch.as_tensor(all_targ)

    print("ACCURACY: {}".format(accuracy(all_pred, all_targ)))
    print("ACCURACY WEIGHTED: {}".format(accuracy_weighted(all_pred, all_targ)))
    print("ACCURACY FOR EACH CLASS: {}".format(accuracy_for_classes(all_pred, all_targ)))
    #conf_matrix = confmat(all_pred, all_targ)
    #torch.save({"confusion_matrix": conf_matrix}, "saves/confusion_matrix/conf_matrix_pretrained_rgb.pt")
    
    
    return {}


def reduce_size_tensor (tensor, size):
    return {k: v[:size] for k, v in tensor.items()}


