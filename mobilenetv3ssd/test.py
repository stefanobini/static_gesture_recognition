import datetime, sys
import argparse
from torchvision.models.detection.ssdlite import ssdlite320_mobilenet_v3_large as mobilenetv3ssd
import transforms_custom as T
import matplotlib.pyplot as plt
from engine_with_manipulation_reintegration import test
import torch
from HagridDataset import HagridDataset
from CustomDataset import CustomDataset

def collate_fn(batch): 
    return tuple(zip(*batch)) 


def main():

    transform = T.Compose([T.ToTensor(), T.ResizeTo320()])
    #dataset_test = HagridDataset("test", transform)
    dataset_test = CustomDataset("test","rgb", "1m", transform)
    data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=1, shuffle=False, num_workers=1,
    collate_fn=collate_fn
    )

    #model = mobilenetv3ssd(weights=None, progress = True,   num_classes = 19)   #18 + 1 no-gesture + background
    model = mobilenetv3ssd(weights=None, progress = True,   num_classes = 14)   #12 + 1 no-gesture + background

    device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
    
    #best checkpoint SSD only custom epoch 15 : "saves/models/last_dance/15ep_2022_09_16_17_55_finetuning_hagrid_model.pt"
    #best checkpoint DEPTH SSD only custom epoch 15 : "saves/models/depth_noHagrid/15ep_2022_09_19_12_55_finetuning_custom_depth_model.pt"
    #checkpoint = torch.load("saves/models/last_dance/15ep_2022_09_16_17_55_finetuning_hagrid_model.pt")
    #checkpoint = torch.load("saves/models/depth_noHagrid/15ep_2022_09_19_12_55_finetuning_custom_depth_model.pt")

    checkpoint = torch.load("static_gesture_recognition/mobilenetv3ssd/experiments/FELICE/demo7/rgb.pt")
    
    
    #PRETRAINED


    #RGB FREEZED
    #checkpoint = torch.load("saves/models/pretrainedHagridFinetuningCustom/freezed/10ep_2022_09_22_13_27_finetuning_from_hagrid_to_custom_BB_block_model.pt")
    
    #DEPTH PREEETRAINING
    #PRETRAINED DEPTH NO FREEZ 5 epoch
    #checkpoint = torch.load("saves/models/pretrainedHagridFinetuningCustom/depth/no_freezed/5ep_2022_09_28_10_10_depth_finetuning_from_hagrid_to_custom_model.pt")
    #print(checkpoint.keys())
    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(device)

    result = test(model, data_loader_test, device, "custom")

    print(result)

if __name__ == "__main__":
    main()