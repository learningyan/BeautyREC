#!/usr/bin/python
# -*- encoding: utf-8 -*-
import os.path as osp

import numpy as np
import cv2
from PIL import Image
import torch
import torchvision.transforms as transforms

from .model import BiSeNet


class FaceParser:
    def __init__(self, device="cuda"):
        mapper = [0, 1, 2, 3, 4, 5, 0, 11, 12, 0, 6, 8, 7, 9, 13, 0, 0, 10, 0]
        self.device = device
        self.dic = torch.tensor(mapper, device=device)
        save_pth = osp.split(osp.realpath(__file__))[0] + '/resnet.pth'

        net = BiSeNet(n_classes=19)
        net.load_state_dict(torch.load(save_pth, map_location=device))
        self.net = net.to(device).eval()
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])


    def parse(self, image: Image):
        assert image.shape[:2] == (512, 512)
        with torch.no_grad():
            image = self.to_tensor(image).to(self.device)
            image = torch.unsqueeze(image, 0)
            out = self.net(image)[0]
            parsing = out.squeeze(0).argmax(0)
        parsing = torch.nn.functional.embedding(parsing, self.dic)
        return parsing.float()


if __name__ == '__main__':
    import glob
    import misc_utils as utils
    import pdb
    import os
    import torch.nn.functional as F

    images = glob.glob('/home/raid/yanqixin/makeup/beautyREC/wilddataset/images/makeup/*.jpg')
    savepath = '/home/raid/yanqixin/makeup/beautyREC/wilddataset/parsing2/makeup'
    utils.try_make_dir(savepath)
    parse = FaceParser()
    for j, img in enumerate (images):
        print(f'{j}/{len(images)}')
        image = Image.open(img).convert('RGB')
        np_image = np.array(image)
        np_image = cv2.resize(np_image, (512, 512))
        mask = parse.parse(np_image)
        mask = F.interpolate(
                mask.view(1, 1, 512, 512),
                (256,256),
                mode="nearest")
        # lip_class   = [7,9]
        # face_class  = [1,6]
        # eyes_class = [4,5]

        # mask_lip = (mask == lip_class[0]).float() + (mask == lip_class[1]).float()
        # mask_face = (mask == face_class[0]).float() + (mask == face_class[1]).float()
        # mask_eyes = (mask == eyes_class[0]).float() + (mask == eyes_class[1]).float()

        # mask_list = [mask_lip, mask_face, mask_eyes]
        # mask_aug = torch.cat(mask_list, 0) 


        # image = mask_aug[:,0,:,:].detach().cpu().numpy().transpose([1,2,0])
        # image = (image.copy()*255).astype(np.uint8)

        # pdb.set_trace()

        name = os.path.join(savepath, os.path.basename(img))
        cv2.imwrite(name , mask[0,0,:,:].detach().cpu().numpy().astype(np.uint8))
        # pdb.set_trace()
