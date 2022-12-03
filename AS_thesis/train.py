# -*- coding: utf-8 -*-
from network import Network
from get_config import get_config
from dataloader.as_dataloader import get_as_dataloader
from get_model import get_model
import os
from utils import validation_constructive

import wandb

if __name__ == "__main__":
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    config = get_config()
    
    if config['use_wandb']:
        run = wandb.init(project="Elec532-AS", entity="armin-saadat", config=config, name='CrossEntropy_MSELoss_AVA')
    
    model = get_model(config)
    net = Network(model, config)
    dataloader_tr = get_as_dataloader(config, split='train', mode='train')
    dataloader_ssl = get_as_dataloader(config, split='train_all', mode='ssl')
    dataloader_va = get_as_dataloader(config, split='val', mode='val')
    dataloader_te = get_as_dataloader(config, split='test', mode='test')
    
    if config['mode']=="train":
        net.train(dataloader_tr, dataloader_va)
        #net.test_comprehensive(dataloader_te, mode="test")
    if config['mode']=="ssl":
        net.train(dataloader_ssl, dataloader_va)
        #net.test_comprehensive(dataloader_te, mode="test")
    if config['mode']=="test":
        net.test_comprehensive(dataloader_va, mode="test", record_embeddings=True)
    if config['use_wandb']:
        wandb.finish()
