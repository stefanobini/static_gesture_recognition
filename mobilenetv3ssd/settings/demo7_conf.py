import os
from typing import List
from dotmap import DotMap

settings = DotMap()

settings.name:str = __file__
settings.mode:str = "train"
settings.experimentation:str = "FELICE"
settings.demo:str = "demo7"

settings.input.modality:list = "rgb"

settings.dataset:str = os.path.join("datasets", "MIVIA_HGR")
settings.device:str = 'cuda:0'

settings.training.batch_size = 32
settings.validation.batch_size = 32
settings.training.epochs:int = 100
settings.training.lr = 0.001
settings.training.momentum = 0.9
settings.training.weight_decay = 0.0005

settings.experiment.folder:str = os.path.join("experiments", settings.experimentation, settings.demo)
os.makedirs(name=settings.experiment.folder, exist_ok=True)
settings.experiment.checkpoint.path:str = os.path.join(settings.experiment.folder, "{}.pt".format(settings.input.modality))