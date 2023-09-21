import datetime, sys
import argparse
from torchvision.models.detection.ssdlite import ssdlite320_mobilenet_v3_large as mobilenetv3ssd
import vision.references.detection.transforms as T
import matplotlib.pyplot as plt
from engine import benchmark, test
import torch
from HagridDataset import HagridDataset
from CustomDataset import CustomDataset

def collate_fn(batch): 
    return tuple(zip(*batch)) 


def main():

    transform = T.Compose([T.ToTensor(), T.ResizeTo320()])
    dataset_test = CustomDataset("test","rgb", transform)
    data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=1, shuffle=False, num_workers=1,
    collate_fn=collate_fn
    )

    #model = mobilenetv3ssd(weights=None, progress = True,   num_classes = 19)   #18 + 1 no-gesture + background
    model = mobilenetv3ssd(weights=None, progress = True,   num_classes = 14)   #12 + 1 no-gesture + background

    device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
    
    checkpoint = torch.load("saves/models/last_dance/20ep_2022_09_16_17_55_finetuning_hagrid_model.pt")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    result = benchmark(model, data_loader_test, device)

    #print(result)

if __name__ == "__main__":
    main()