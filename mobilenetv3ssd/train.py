"""
python3 train.py --configuration demo7_conf
"""

import datetime, sys, os, argparse, colorama
colorama.init(autoreset=True)
from colorama import Back, Fore
from torchvision.models.detection.ssdlite import ssdlite320_mobilenet_v3_large as mobilenetv3ssd
import transforms_custom as T
import matplotlib.pyplot as plt
from engine import train_one_epoch, evaluate
import torch
from HagridDataset import HagridDataset
from CustomDataset import CustomDataset
import matplotlib.patches as patches
from torch.utils.tensorboard import SummaryWriter

########################
# Acquiring parameters #
########################
parser = argparse.ArgumentParser()
parser.add_argument("--configuration", type=str, dest="configuration", required=True, help="Configuration file (e.g., 'conf_1')")
args = parser.parse_args()
args.configuration = "settings.{}".format(args.configuration)
settings = getattr(__import__(args.configuration, fromlist=["settings"]), "settings")
print(Back.CYAN + "Loaded <{}> as configuration file.".format(settings.name))

ROOT = settings.dataset
e = datetime.datetime.now()
writer_ts = SummaryWriter(log_dir=settings.experiment.folder)

def collate_fn(batch): 
    return tuple(zip(*batch)) 

def main():
    #transform = T.Compose([T.ToTensor()])
    #transform = T.Compose([T.FixedCenterSizeCrop(size = (480, 480)),T.ToTensor(), T.ResizeTo320()])
    transform = T.Compose([T.ToTensor(), T.ResizeTo320()])
    #transform_base = T.Compose([Tra.Resize(320)])
    """ dataset = HandsDataset('hands_dataset', "train",  transform)
    dataset_val = HandsDataset('hands_dataset', "val",  transform) """
    #dataset = HagridDataset("train", transform)
    #dataset_val = HagridDataset("val", transform)
    dataset = CustomDataset(settings=settings, mode="train", distance=None, transforms=transform)
    dataset_val = CustomDataset(settings=settings, mode="val", distance=None, transforms=transform)


    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=settings.training.batch_size, shuffle=True, num_workers=4,
        collate_fn=collate_fn
        )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=settings.validation.batch_size, shuffle=False, num_workers=4,
        collate_fn=collate_fn
        )

    #model = mobilenetv3ssd(weights= None, progress = True,  trainable_backbone_layers = 0, num_classes=19)   #29 + 1 background
    #model = mobilenetv3ssd(weights= None, progress = True, weights_backbone="MobileNet_V3_Large_Weights.IMAGENET1K_V2",   num_classes = 20)   #29 + 1 background
    #backbone = torch.load("saves/bakcbone_hagrid/backbone.pt")
    model = mobilenetv3ssd(weights= None, progress = True, num_classes=14)
    device = torch.device(settings.device) if torch.cuda.is_available() else torch.device('cpu')
    
    #model.backbone.load_state_dict(backbone["model.backbone.weights"])
    # DEVO SALVARMI l'ID ORIGINALE DELL'IMMAGINE?


    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=settings.training.lr, momentum=settings.training.momentum, weight_decay=settings.training.weight_decay)
    # and a learning rate scheduler
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    #checkpoint = torch.load("saves/models/last_change_bbox 35_25/10ep_2022_09_13_17_35_finetuning_hagrid_model.pt")
    #model.load_state_dict(checkpoint['model_state_dict'])
    #torch.save({'model.backbone': model.backbone, 'model.backbone.weights': model.backbone.state_dict()} , 'saves/bakcbone_hagrid/backbone.pt')
    #last_layer_custom = torch.load("saves/last_layer_custom/lastlayer.pt")
    #model.load_state_dict(checkpoint['model_state_dict'])
    

    #optimizer = torch.optim.SGD(checkpoint['optimizer_state_dict'], lr=0.01, momentum=0.9)
    model.to(device)

    # let's train it for 10 epochs
    num_epochs = settings.training.epochs

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        losses, loss_dict, lr_value= train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=100)
        # update the learning rate
        writer_ts.add_scalar("Total Loss/train", losses, epoch)
        writer_ts.add_scalar("Loss_detector/train", loss_dict['bbox_regression'], epoch)
        writer_ts.add_scalar("Loss_class/train", loss_dict["classification"], epoch)
        writer_ts.add_scalar("Learning_rate", lr_value, epoch)
        lr_scheduler.step(losses)
        # evaluate on the test dataset 
        val_losses = evaluate(model, data_loader_val, device=device)
        total_val_loss = sum(loss for loss in val_losses.values())
        writer_ts.add_scalar("Total Loss/Val", total_val_loss, epoch)
        writer_ts.add_scalar("Loss_detector/val", val_losses["bbox_regression"].item(), epoch)
        writer_ts.add_scalar("Loss_class/val", val_losses["classification"].item(), epoch)
        
        # save the best model
        prev_val_loss = total_val_loss if epoch==0 else prev_val_loss
        if total_val_loss < prev_val_loss:
            torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),}, settings.experiment.checkpoint.path)
            prev_val_loss = total_val_loss

    print("That's it!")
    weight_name = e.strftime(settings.experiment.checkpoint.path)
    torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, settings.experiment.checkpoint.path.replace(".pt", "{}epoch.pt".format(epoch)))



if __name__ == "__main__":
    main()


'''
   ##################  STAMPARE BBOX
    images,targets = next(iter(data_loader))
    images = list(image for image in images)
    targets = [{k: v for k, v in t.items()} for t in targets]
    #dataset = HandsDataset('TESTMOBILENET/hands/hands/images/100k/train', transform)
    #dataset_normal = HandsDataset('TESTMOBILENET/hands/hands/images/100k/train', None)
    #plt.imshow(dataset[0][0].permute(1, 2, 0))

    fig3, ax3 = plt.subplots(1)
    ax3.imshow(images[0].permute(1, 2, 0))
    boxes3 = targets[0]['boxes']
    #print(boxes3)
    boxes3 = boxes3.to(torch.float).tolist()
    id = targets[0]['image_id'].item()
    print(id)
    #print(boxes3)
    for bbox in boxes3:
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=1, edgecolor='r', facecolor='none')
        ax3.add_patch(rect)

    id-=1
    fig3.savefig("fig3.png")
    #print(dataset_simple[id][1]['image_id'])
    #print(dataset_simple[id][1]['boxes'])
    fig4, ax4 = plt.subplots(1)
    ax4.imshow(dataset[id][0].permute(1, 2, 0))
    print(id)
    boxes4 = dataset[id][1]['boxes']
    boxes4 = boxes4.to(torch.float).tolist()
    #print(boxes4)
    for bbox in boxes4:
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=1, edgecolor='r', facecolor='none')
        ax4.add_patch(rect)
    fig4.savefig("fig4.png")
    ################## 
'''