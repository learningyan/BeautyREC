import os
import pdb
import time
import numpy as np
import cv2
import torch
from torch import optim
from torch.autograd import Variable

import makeuploader.dataloaders as dl
from network import get_model

from options import opt

from utils import init_log, load_meta, save_meta
from mscv.summary import create_summary_writer, write_meters_loss,write_image

import misc_utils as utils
from mscv.image import tensor2im

from network.REC.Model import Model 
test_loader = dl.testdataloader

save_root = 'results/res'
utils.try_make_dir(save_root)
save_root_ori = 'results/ori'
utils.try_make_dir(save_root_ori)

model = Model(opt)
load_epoch = model.load(opt.load)
model.eval()


with torch.no_grad():

    for j, sample in enumerate(test_loader):
        print(f'{j}/{len(test_loader)}')
        try:
            pred = model(sample)

            no_image = sample['nonmakeup_img'][0].detach().cpu().numpy().transpose([1,2,0])/2+0.5
            no_image = (no_image.copy()*255).astype(np.uint8)

            image = sample['makeup_img'][0].detach().cpu().numpy().transpose([1,2,0])/2+0.5
            image = (image.copy()*255).astype(np.uint8)

            pred = pred[0].detach().cpu().numpy().transpose([1,2,0])/2+0.5
            pred = (pred.copy()*255).astype(np.uint8)

            cv2.imwrite(f'{save_root}/{j}_nomakeup.png', no_image[:,:,::-1])
            cv2.imwrite(f'{save_root}/{j}_makeup.png', image[:,:,::-1])
            cv2.imwrite(f'{save_root}/{j}_pred.png', pred[:,:,::-1])
        except Exception as e:
            print(e)
