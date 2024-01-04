import datetime, sys
import argparse
from torchvision.models.detection.ssdlite import ssdlite320_mobilenet_v3_large as mobilenetv3ssd
import transforms_custom as T
import matplotlib.pyplot as plt
from engine_fusion import calculate_and_save_preds, weighted_fusion
import torch
from HagridDataset import HagridDataset
from CustomDataset import CustomDataset

def collate_fn(batch): 
    return tuple(zip(*batch)) 


def main():

    transform = T.Compose([T.ToTensor(), T.ResizeTo320()])
    #dataset_test = HagridDataset("test", transform)
    device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')


    #BLOCK of CODE TO SAVE RGB SSD RESULT PRETRAINED OR NOTPRETRAINED
    # BEFORE RUN THE CODE, 3 CHANGES NEED TO BE DONE: CHOOSE THE DISTANCE OF DATASET, CHOOSE WHICH WEIGHTS
    # TO LOAD AND CHANGE THE LOCATION AND FILENAME WHERE SAVE THE PREDICITION IN THE LAST ROWS OF "calculate_and_save_preds"
    """ dataset_test = CustomDataset("test","rgb", "5m", transform)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=8, shuffle=False, num_workers=1, collate_fn=collate_fn)
    model = mobilenetv3ssd(weights=None, progress = True,   num_classes = 14)   #12 + 1 no-gesture + background
    #checkpoint = torch.load("saves/models/last_dance/15ep_2022_09_16_17_55_finetuning_hagrid_model.pt")
    #PRETRAINED
    #RGB FREEZED
    checkpoint = torch.load("saves/models/pretrainedHagridFinetuningCustom/freezed/10ep_2022_09_22_13_27_finetuning_from_hagrid_to_custom_BB_block_model.pt")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    result = calculate_and_save_preds(model, data_loader_test, device, "custom") """


    #BLOCK of CODE TO SAVE DEPTH SSD RESULT PRETRAINED OR NOTPRETRAINED
    # BEFORE RUN THE CODE, 3 CHANGES NEED TO BE DONE: CHOOSE THE DISTANCE OF DATASET, CHOOSE WHICH WEIGHTS
    # TO LOAD AND CHANGE THE LOCATION AND FILENAME WHERE SAVE THE PREDICITION IN THE LAST ROWS OF "calculate_and_save_preds"
    """ dataset_test_depth = CustomDataset("test","depth", "5m", transform)
    data_loader_test_depth = torch.utils.data.DataLoader(dataset_test_depth, batch_size=8, shuffle=False, num_workers=1, collate_fn=collate_fn)
    model_depth = mobilenetv3ssd(weights=None, progress = True,   num_classes = 14)   #12 + 1 no-gesture + background
    #ONLY CUSTOM
    #checkpoint_depth = torch.load("saves/models/depth_noHagrid/15ep_2022_09_19_12_55_finetuning_custom_depth_model.pt")
    #DEPTH PREEETRAINING
    #PRETRAINED DEPTH NO FREEZ 5 epoch
    checkpoint_depth = torch.load("saves/models/pretrainedHagridFinetuningCustom/depth/no_freezed/5ep_2022_09_28_10_10_depth_finetuning_from_hagrid_to_custom_model.pt")
    model_depth.load_state_dict(checkpoint_depth["model_state_dict"])
    model_depth.to(device)
    result = calculate_and_save_preds(model_depth,  data_loader_test_depth, device, "custom") """


    #BLOCK OF CODE TO PERFORM FUSION AND CALCULATE METRICS
    # 2 CHANGES NEEDED: CHOOSE THE TYPE & DISTANCES OF DATASET AND CHOOSE WHICH RESULT PREDICTION LOAD IN "weighted_fusion"
    dataset_test = CustomDataset("test","rgb", "2m", transform)
    #data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=1, collate_fn=collate_fn)
    weighted_fusion(dataset_test)

if __name__ == "__main__":
    main()