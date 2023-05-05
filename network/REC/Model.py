import pdb

import numpy as np
import torch
import os
import torch.nn as nn
from .REC import BeautyREC

from options import opt

from optimizer import get_optimizer
from scheduler import get_scheduler

from network.base_model import BaseModel
from mscv import ExponentialMovingAverage, print_network, load_checkpoint, save_checkpoint
from loss import  GANLoss,HistogramLoss,ColorLoss
from .dis import Discriminator_VGG, SCDis, UNetDiscriminatorSN
from torch.autograd import Variable

import misc_utils as utils
from .vgg import VGG

params={'dim':48,
        'style_dim':48,
        'activ': 'relu',
        'n_downsample':2,
        'n_res':3,
        'pad_type':'reflect'
}


class Model(BaseModel):
    def __init__(self, opt):
        super(Model, self).__init__()
        self.opt = opt
        self.cleaner = BeautyREC(params).to(device=opt.device)
        self.discriminator_lip = UNetDiscriminatorSN().to(device=opt.device)
        self.discriminator_skin = UNetDiscriminatorSN().to(device=opt.device)
        self.discriminator_eye_left= UNetDiscriminatorSN().to(device=opt.device)
        self.discriminator_eye_right= UNetDiscriminatorSN().to(device=opt.device)
        self.discriminator_all = UNetDiscriminatorSN().to(device=opt.device)

        self.vgg = VGG().to(device=opt.device)
        self.vgg.load_state_dict(torch.load('./vgg_conv.pth'))

        #####################


        # print_network(self.cleaner)
        model_params = filter(lambda p:p.requires_grad, self.cleaner.parameters())
        res = sum(np.prod(p.size()) for p in model_params)/1000000
        print(f'****   Model_Params: {res}M !!!!!!!')
        
        self.g_optimizer = get_optimizer(opt, self.cleaner)
        self.scheduler = get_scheduler(opt, self.g_optimizer)

        self.d_optimizer_lip = get_optimizer(opt, self.discriminator_lip)
        self.d_optimizer_skin = get_optimizer(opt, self.discriminator_skin)
        self.d_optimizer_eye_left = get_optimizer(opt, self.discriminator_eye_left)
        self.d_optimizer_eye_right = get_optimizer(opt, self.discriminator_eye_right)
        self.d_optimizer_all= get_optimizer(opt, self.discriminator_all)

        self.avg_meters = ExponentialMovingAverage(0.95)
        self.save_dir = os.path.join(opt.checkpoint_dir, opt.tag)

        self.criterionGAN = GANLoss(use_lsgan=True, tensor=torch.cuda.FloatTensor)
        self.criterionL1 = nn.L1Loss()
        self.criterionHis = HistogramLoss()
        self.criterionL2 = torch.nn.MSELoss()
        self.criterionColor = ColorLoss()

    def to_var(self, x, requires_grad=False):
        if isinstance(x, list):
            return x
        if torch.cuda.is_available():
            x = x.cuda()
        if not requires_grad:
            return Variable(x, requires_grad=requires_grad)
        else:
            return Variable(x)    

    def update(self, sample):

        nonmakeup = self.to_var(sample['nonmakeup_img'])
        makeup = self.to_var(sample['makeup_img'])

        makeup_seg = self.to_var(sample['makeup_seg'])
        nonmakeup_seg = self.to_var(sample['nonmakeup_seg'])

        self.mask_A = sample['mask_A']
        self.mask_B = sample['mask_B']
        
        self.makeup_unchanged=sample['makeup_unchanged']
        self.nonmakeup_unchanged=sample['nonmakeup_unchanged']

        mask_nonmakeup = {key: self.to_var(self.mask_A[key]) for key in self.mask_A}
        mask_makeup = {key: self.to_var(self.mask_B[key]) for key in self.mask_B}
        makeup_unchanged=self.to_var(self.makeup_unchanged)
        nonmakeup_unchanged=self.to_var(self.nonmakeup_unchanged)

        fake_makeup = self.forward(sample)

        content_x, = self.cleaner.infercontent(nonmakeup)
        content_x_hat = self.cleaner.infercontent(fake_makeup)

        style_y = self.cleaner.inferstyle(makeup, makeup_seg)
        style_x_hat = self.cleaner.inferstyle(fake_makeup, nonmakeup_seg)


        _,c,_,_ = style_y.shape
        style_y_lip = style_y[:,:c//3,:,:]
        style_x_hat_lip = style_x_hat[:,:c//3,:,:]

        style_y_skin = style_y[:,c//3:c//3*2,:,:]
        style_x_hat_skin = style_x_hat[:,c//3:c//3*2,:,:]

        style_y_eye = style_y[:,c//3*2:,:,:]
        style_x_hat_eye = style_x_hat[:,c//3*2:,:,:]

        loss_content = self.criterionL1(content_x, content_x_hat)

        # pdb.set_trace()
        ### four local discriminators 
        ##########################################################
        # pdb.set_trace()
        a,b,c,d = torch.min(self.mask_B['index_B_lip'][0]), torch.max(self.mask_B['index_B_lip'][0]), torch.min(self.mask_B['index_B_lip'][1]), torch.max(self.mask_B['index_B_lip'][1])
        y_lip = style_y_lip[:,:,a:b+1,c:d+1]
        out = self.discriminator_lip(y_lip)
        d_loss_real = self.criterionGAN(out, True)

        a,b,c,d = torch.min(self.mask_A['index_A_lip'][0]), torch.max(self.mask_A['index_A_lip'][0]), torch.min(self.mask_A['index_A_lip'][1]), torch.max(self.mask_A['index_A_lip'][1])
        x_hat_lip = style_x_hat_lip[:,:,a:b+1,c:d+1]

        out = self.discriminator_lip(x_hat_lip)
        d_loss_fake = self.criterionGAN(out, False)

        loss_d_lip = (d_loss_real.mean() + d_loss_fake.mean()) * 0.5

        #---------------------------------------------------------

        a,b,c,d = torch.min(self.mask_B['index_B_skin'][0]), torch.max(self.mask_B['index_B_skin'][0]), torch.min(self.mask_B['index_B_skin'][1]), torch.max(self.mask_B['index_B_skin'][1])
        y_skin = style_y_skin[:,:,a:b+1,c:d+1]
        out = self.discriminator_skin(y_skin)
        d_loss_real = self.criterionGAN(out, True)

        a,b,c,d = torch.min(self.mask_A['index_A_skin'][0]), torch.max(self.mask_A['index_A_skin'][0]), torch.min(self.mask_A['index_A_skin'][1]), torch.max(self.mask_A['index_A_skin'][1])
        x_hat_skin = style_x_hat_skin[:,:,a:b+1,c:d+1]

        out = self.discriminator_skin(x_hat_skin)
        d_loss_fake = self.criterionGAN(out, False)

        loss_d_skin = (d_loss_real.mean() + d_loss_fake.mean()) 

        #---------------------------------------------------------
        a,b,c,d = torch.min(self.mask_B['index_B_eye_left'][0]), torch.max(self.mask_B['index_B_eye_left'][0]), torch.min(self.mask_B['index_B_eye_left'][1]), torch.max(self.mask_B['index_B_eye_left'][1])
        y_eye_left = style_y_eye[:,:,a:b+1,c:d+1]

        out = self.discriminator_eye_left(y_eye_left)
        d_loss_real = self.criterionGAN(out, True)

        a,b,c,d = torch.min(self.mask_A['index_A_eye_left'][0]), torch.max(self.mask_A['index_A_eye_left'][0]), torch.min(self.mask_A['index_A_eye_left'][1]), torch.max(self.mask_A['index_A_eye_left'][1])
        x_hat_eye_left = style_x_hat_eye[:,:,a:b+1,c:d+1]

        out = self.discriminator_eye_left(x_hat_eye_left)
        d_loss_fake = self.criterionGAN(out, False)

        loss_d_eye_left = (d_loss_real.mean() + d_loss_fake.mean()) * 0.5

        #---------------------------------------------------------
        a,b,c,d = torch.min(self.mask_B['index_B_eye_right'][0]), torch.max(self.mask_B['index_B_eye_right'][0]), torch.min(self.mask_B['index_B_eye_right'][1]), torch.max(self.mask_B['index_B_eye_right'][1])
        y_eye_right = style_y_eye[:,:,a:b+1,c:d+1]
        # pdb.set_trace()
        out = self.discriminator_eye_right(y_eye_right)
        d_loss_real = self.criterionGAN(out, True)

        a,b,c,d = torch.min(self.mask_A['index_A_eye_right'][0]), torch.max(self.mask_A['index_A_eye_right'][0]), torch.min(self.mask_A['index_A_eye_right'][1]), torch.max(self.mask_A['index_A_eye_right'][1])
        x_hat_eye_right = style_x_hat_eye[:,:,a:b+1,c:d+1]

        out = self.discriminator_eye_right(x_hat_eye_right)
        d_loss_fake = self.criterionGAN(out, False)

        loss_d_eye_right = (d_loss_real.mean() + d_loss_fake.mean()) * 0.5
        #---------------------------------------------------------

        out = self.discriminator_all(makeup)
        d_loss_real = self.criterionGAN(out, True)

        out = self.discriminator_all(fake_makeup)
        d_loss_fake = self.criterionGAN(out, False)

        loss_d_global= (d_loss_real.mean() + d_loss_fake.mean()) * 0.5

        #############################################################

        loss_d = (loss_d_lip  + loss_d_skin + loss_d_eye_left + loss_d_eye_right + loss_d_global).mean()



        pred_fake_lip = self.discriminator_lip(x_hat_lip)
        g_loss_adv_lip = self.criterionGAN(pred_fake_lip, True)

        pred_fake_skin = self.discriminator_skin(x_hat_skin)
        g_loss_adv_skin = self.criterionGAN(pred_fake_skin, True)

        pred_fake_eye_left = self.discriminator_eye_left(x_hat_eye_left)
        g_loss_adv_eye_left = self.criterionGAN(pred_fake_eye_left, True)

        pred_fake_eye_right = self.discriminator_eye_right(x_hat_eye_right)
        g_loss_adv_eye_right = self.criterionGAN(pred_fake_eye_right, True)

        pred_fake_global = self.discriminator_all(fake_makeup)
        g_loss_adv_global = self.criterionGAN(pred_fake_global, True)

        g_loss_adv = (g_loss_adv_lip + g_loss_adv_skin + g_loss_adv_eye_left +  g_loss_adv_eye_right + g_loss_adv_global)
        # pdb.set_trace()

        g_lip_loss_his = self.criterionHis(fake_makeup, makeup, mask_nonmakeup["mask_A_lip"],
                                                mask_makeup['mask_B_lip'],
                                                mask_nonmakeup["index_A_lip"],
                                                nonmakeup)

        g_skin_loss_his = self.criterionHis(fake_makeup, makeup, mask_nonmakeup["mask_A_skin"],
                                                mask_makeup['mask_B_skin'],
                                                mask_nonmakeup["index_A_skin"],
                                                nonmakeup)

        g_eye_left_loss_his = self.criterionHis(fake_makeup, makeup,
                                                                  mask_nonmakeup["mask_A_eye_left"],
                                                                  mask_makeup["mask_B_eye_left"],
                                                                  mask_nonmakeup["index_A_eye_left"],
                                                                  nonmakeup)

        g_eye_right_loss_his = self.criterionHis(fake_makeup, makeup,
                                                                   mask_nonmakeup["mask_A_eye_right"],
                                                                   mask_makeup["mask_B_eye_right"],
                                                                   mask_nonmakeup["index_A_eye_right"],
                                                                   nonmakeup) 

        
        g_loss_wuguan = g_lip_loss_his  + g_skin_loss_his*0.1 + g_eye_left_loss_his + g_eye_right_loss_his
        # g_loss_wuguan = g_skin_loss_his*0.1 + g_eye_left_loss_his + g_eye_right_loss_his

        vgg_s = self.vgg(nonmakeup, ['r41'])[0]
        vgg_s = Variable(vgg_s.data).detach()
        vgg_fake_makeup = self.vgg(fake_makeup, ['r41'])[0]
        g_loss_vgg = self.criterionL2(vgg_fake_makeup, vgg_s)*0.005


        vgg_fake_makeup_unchanged=self.vgg(fake_makeup*nonmakeup_unchanged,['r41'])[0]
        vgg_nonmakeup_masked=self.vgg(nonmakeup*nonmakeup_unchanged,['r41'])[0]
        g_loss_unchanged=self.criterionL2(vgg_fake_makeup_unchanged, vgg_nonmakeup_masked)*0.05

        loss_G = (loss_content + g_loss_adv+ g_loss_wuguan + g_loss_vgg + g_loss_unchanged).mean()
        # loss_G = (loss_content + g_loss_adv+ g_loss_wuguan + g_loss_vgg ).mean()

        self.avg_meters.update({'G_all': loss_G.item(),
            'g_adv': g_loss_adv.item(),
            'wuguan':g_loss_wuguan.item(),
            'vgg': g_loss_vgg.item(),
            'unchanged':g_loss_unchanged.item(),
            'D':loss_d.item()})

        self.d_optimizer_lip.zero_grad()
        self.d_optimizer_skin.zero_grad()
        self.d_optimizer_eye_left.zero_grad()
        self.d_optimizer_eye_right.zero_grad()
        self.d_optimizer_all.zero_grad()

        (loss_d).backward(retain_graph=True)

        self.g_optimizer.zero_grad()
        loss_G.backward()

        self.d_optimizer_lip.step() 
        self.d_optimizer_skin.step() 
        self.d_optimizer_eye_left.step() 
        self.d_optimizer_eye_right.step() 
        self.d_optimizer_all.step()

        self.g_optimizer.step()

        return {'output': fake_makeup}

    def forward(self, sample):

        x = sample['nonmakeup_img'].to(opt.device)
        y = sample['makeup_img'].to(opt.device)
        y_seg = sample['makeup_seg'].to(opt.device)
        x_seg = sample['nonmakeup_seg'].to(opt.device)

        out = self.cleaner(x, y, y_seg, x_seg)
        return out

    def write_train_summary(self, update_return):
        pass

    def step_scheduler(self):
        self.scheduler.step()

    def get_lr(self):
        return self.scheduler.get_lr()[0]

    def load(self, ckpt_path):
        load_dict = {
            'cleaner': self.cleaner,
        }

        if opt.resume:
            load_dict.update({
                'optimizer': self.g_optimizer,
                'scheduler': self.scheduler,
            })
            utils.color_print('Load checkpoint from %s, resume training.' % ckpt_path, 3)
        else:
            utils.color_print('Load checkpoint from %s.' % ckpt_path, 3)

        ckpt_info = load_checkpoint(load_dict, ckpt_path, map_location=opt.device)
        epoch = ckpt_info.get('epoch', 0)

        return epoch

    def save(self, which_epoch):
        save_filename = f'{which_epoch}_{opt.model}.pt'
        save_path = os.path.join(self.save_dir, save_filename)
        save_dict = {
            'cleaner': self.cleaner,
            'optimizer': self.g_optimizer,
            'scheduler': self.scheduler,
            'epoch': which_epoch
        }

        save_checkpoint(save_dict, save_path)
        utils.color_print(f'Save checkpoint "{save_path}".', 3)



