import os.path
import pdb
import torchvision.transforms as transforms
import os.path as osp

from PIL import Image
import PIL
import numpy as np
import torch
from torch.autograd import Variable
import cv2
pwd = osp.split(osp.realpath(__file__))[0]
import sys
sys.path.append(pwd + '/..')
import faceutils as futils
import torch.nn.functional as F



def ToTensor(pic):
    # handle PIL Image
    if pic.mode == 'I':
        img = torch.from_numpy(np.array(pic, np.int32, copy=False))
    elif pic.mode == 'I;16':
        img = torch.from_numpy(np.array(pic, np.int16, copy=False))
    else:
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
    # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
    if pic.mode == 'YCbCr':
        nchannel = 3
    elif pic.mode == 'I;16':
        nchannel = 1
    else:
        nchannel = len(pic.mode)
    img = img.view(pic.size[1], pic.size[0], nchannel)
    # put it from HWC to CHW format
    # yikes, this transpose takes 80% of the loading time/CPU
    img = img.transpose(0, 1).transpose(0, 2).contiguous()
    if isinstance(img, torch.ByteTensor):
        return img.float()
    else:
        return img

def to_var(x, requires_grad=True):
    if requires_grad:
        return Variable(x).float()
    else:
        return Variable(x, requires_grad=requires_grad).float()

dataroot = 'wilddataset/images'
n_componets = 3
img_size = 256

class WildDataset():
    def __init__(self, phase):
        self.random = None
        self.phase = phase
        self.root = dataroot
        self.dir_makeup = dataroot
        self.dir_nonmakeup = dataroot
        self.n_componets = n_componets
        self.makeup_names = []
        self.non_makeup_names = []
        if self.phase == 'train':
            self.makeup_names = [name.strip() for name in
                                 open(os.path.join('wilddataset', 'makeup.txt'), "rt").readlines()]
            self.non_makeup_names = [name.strip() for name in
                                     open(os.path.join('wilddataset', 'nonmakeup.txt'), "rt").readlines()]
        if self.phase == 'test':

            self.makeup_names = [name.strip() for name in
                                 open(os.path.join('wilddataset', 'make_draw.txt'), "rt").readlines()]
            self.non_makeup_names = [name.strip() for name in
                                     open(os.path.join('wilddataset', 'nomake_draw.txt'), "rt").readlines()]
            # import random
            # random.seed(6)
            # random.shuffle(self.makeup_names)
            # random.shuffle(self.non_makeup_names)

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])

        self.transform_mask = transforms.Compose([
            # transforms.Resize((img_size, img_size), interpolation=PIL.Image.NEAREST),
            ToTensor])

    def __getitem__(self, index):
        if self.phase == 'test':
            makeup_name = self.makeup_names[index]
            nonmakeup_name = self.non_makeup_names[index]
        if self.phase == 'train':
            index = self.pick()
            makeup_name = self.makeup_names[index[0]]
            nonmakeup_name = self.non_makeup_names[index[1]]

        nonmakeup_path = os.path.join(self.dir_nonmakeup, nonmakeup_name)


        makeup_path = os.path.join(self.dir_makeup, makeup_name)

        
        makeup_img = Image.open(makeup_path).convert('RGB')
        nonmakeup_img = Image.open(nonmakeup_path).convert('RGB')


        self.device = 'cpu'
        self.face_parse = futils.mask.FaceParser(device=self.device)

        self.up_ratio = 0.6 / 0.85
        self.down_ratio = 0.2 / 0.85
        self.width_ratio = 0.2 / 0.85
        self.img_size = 256
        self.lip_class   = [7,9]
        self.face_class  = [1,6]
        self.eyes_class = [4,5]

        face = futils.dlib.detect(makeup_img)
        try:
            face_on_image = face[0]
        except:
            return {}


        image_makeup, face, crop_face = futils.dlib.crop(
        makeup_img, face_on_image, self.up_ratio, self.down_ratio, self.width_ratio)
        np_image_makeup = np.array(image_makeup)
        # pdb.set_trace()
        mask = self.face_parse.parse(cv2.resize(np_image_makeup, (512, 512)))
 
        mask = F.interpolate(
            mask.view(1, 1, 512, 512),
            (self.img_size, self.img_size),
            mode="nearest")
        mask = mask.type(torch.uint8)
        mask = to_var(mask, requires_grad=False).to(self.device)

        # pdb.set_trace()
        mask_B_lip = (mask == self.lip_class[0]).float() + (mask == self.lip_class[1]).float()
        mask_B_face = (mask == self.face_class[0]).float() + (mask == self.face_class[1]).float()
        # mask_eyes = (mask == self.eyes_class[0]).float() + (mask == self.eyes_class[1]).float()
        mask_B_eye_left = (mask == self.eyes_class[0]).float()
        mask_B_eye_right= (mask == self.eyes_class[1]).float()

        if not ((mask_B_eye_left > 0).any() and \
                (mask_B_eye_right > 0).any()):
            return {}
        mask_B_eye_left, mask_B_eye_right = self.rebound_box(mask_B_eye_left[0], mask_B_eye_right[0], mask_B_face[0])

        if (mask_B_eye_left + mask_B_eye_right).max()>1.5:
            # print('error')
            return {}
        mask_eyes = mask_B_eye_left + mask_B_eye_right
        mask_list = [mask_B_lip, mask_B_face, mask_eyes]
        makeup_seg = torch.cat(mask_list, 0) 
        makeup_seg = makeup_seg[:,0,:,:]

        makeup_unchanged = (mask == 0).float().squeeze(0)

##--------------------------------------------------------------------------------no make up 
        nonmakeface = futils.dlib.detect(nonmakeup_img)
        try:

            face_on_image = nonmakeface[0]  
        except:
            return{}          

        image_nonmakeup, face, crop_face = futils.dlib.crop(
        nonmakeup_img, face_on_image, self.up_ratio, self.down_ratio, self.width_ratio)
        np_image_nomakeup = np.array(image_nonmakeup)
        mask = self.face_parse.parse(cv2.resize(np_image_nomakeup, (512, 512)))

        mask = F.interpolate(
            mask.view(1, 1, 512, 512),
            (self.img_size, self.img_size),
            mode="nearest")
        mask = mask.type(torch.uint8)
        mask = to_var(mask, requires_grad=False).to(self.device)

        
        mask_A_lip = (mask == self.lip_class[0]).float() + (mask == self.lip_class[1]).float()
        mask_A_face = (mask == self.face_class[0]).float() + (mask == self.face_class[1]).float()
        mask_A_eye_left = (mask == self.eyes_class[0]).float()
        mask_A_eye_right= (mask == self.eyes_class[1]).float()


        if not ((mask_A_eye_left > 0).any() and \
                (mask_A_eye_right > 0).any()):
            return {}
    
        # mask_eyes = (mask == self.eyes_class[0]).float() + (mask == self.eyes_class[1]).float()
        mask_A_eye_left, mask_A_eye_right = self.rebound_box(mask_A_eye_left[0], mask_A_eye_right[0], mask_A_face[0])
        if (mask_A_eye_left + mask_A_eye_right).max()>1.5:
            # print('error')
            return {}

        mask_eyes = mask_A_eye_left + mask_A_eye_right
        mask_list = [mask_A_lip, mask_A_face, mask_eyes]
        nonmakeup_seg = torch.cat(mask_list, 0) 
        nonmakeup_seg = nonmakeup_seg[:,0,:,:]

        nonmakeup_unchanged = (mask == 0).float().squeeze(0)


        mask_A_face, mask_B_face, index_A_skin, index_B_skin = self.mask_preprocess(mask_A_face, mask_B_face) 
        mask_A_lip, mask_B_lip, index_A_lip, index_B_lip = self.mask_preprocess(mask_A_lip, mask_B_lip)
        mask_A_eye_left, mask_B_eye_left, index_A_eye_left, index_B_eye_left = self.mask_preprocess(mask_A_eye_left, mask_B_eye_left)
        mask_A_eye_right, mask_B_eye_right, index_A_eye_right, index_B_eye_right = self.mask_preprocess(mask_A_eye_right, mask_B_eye_right)


        mask_A = {}
        mask_A["mask_A_eye_left"] = mask_A_eye_left
        mask_A["mask_A_eye_right"] = mask_A_eye_right
        mask_A["index_A_eye_left"] = index_A_eye_left
        mask_A["index_A_eye_right"] = index_A_eye_right
        mask_A["mask_A_skin"] = mask_A_face
        mask_A["index_A_skin"] = index_A_skin
        mask_A["mask_A_lip"] = mask_A_lip
        mask_A["index_A_lip"] = index_A_lip

        mask_B = {}
        mask_B["mask_B_eye_left"] = mask_B_eye_left
        mask_B["mask_B_eye_right"] = mask_B_eye_right
        mask_B["index_B_eye_left"] = index_B_eye_left
        mask_B["index_B_eye_right"] = index_B_eye_right
        mask_B["mask_B_skin"] = mask_B_face
        mask_B["index_B_skin"] = index_B_skin
        mask_B["mask_B_lip"] = mask_B_lip
        mask_B["index_B_lip"] = index_B_lip
        

        makeup_img = self.transform(image_makeup)
        nonmakeup_img = self.transform(image_nonmakeup)
        return {
                'nonmakeup_seg': nonmakeup_seg,
                'makeup_seg': makeup_seg, 
                'nonmakeup_img': nonmakeup_img,
                'makeup_img': makeup_img,
                'mask_A': mask_A, 
                'mask_B': mask_B,
                'makeup_unchanged': makeup_unchanged,
                'nonmakeup_unchanged': nonmakeup_unchanged,
                'nonmakeup_name':nonmakeup_name,
                'makeup_name':makeup_name
                }


    def pick(self):
        if self.random is None:
            self.random = np.random.RandomState(np.random.seed())
        a_index = self.random.randint(0, len(self.makeup_names))
        another_index = self.random.randint(0, len(self.non_makeup_names))
        return [a_index, another_index]

    def __len__(self):
        if self.phase == 'train':
            return len(self.non_makeup_names)
        elif self.phase == 'test':
            return len(self.non_makeup_names)

    def name(self):
        return 'WildDataset'



    def rebound_box(self, mask_A, mask_B, mask_A_face):
        mask_A = mask_A.unsqueeze(0)
        mask_B = mask_B.unsqueeze(0)
        mask_A_face = mask_A_face.unsqueeze(0)

        index_tmp = torch.nonzero(mask_A, as_tuple=False)
        # pdb.set_trace()
        x_A_index = index_tmp[:, 2]
        y_A_index = index_tmp[:, 3]
        index_tmp = torch.nonzero(mask_B, as_tuple=False)
        x_B_index = index_tmp[:, 2]
        y_B_index = index_tmp[:, 3]
        mask_A_temp = mask_A.copy_(mask_A)
        mask_B_temp = mask_B.copy_(mask_B)
        mask_A_temp[:, :, min(x_A_index) - 5:max(x_A_index) + 6, min(y_A_index) - 5:max(y_A_index) + 6] = \
            mask_A_face[:, :, min(x_A_index) - 5:max(x_A_index) + 6, min(y_A_index) - 5:max(y_A_index) + 6]
        mask_B_temp[:, :, min(x_B_index) - 5:max(x_B_index) + 6, min(y_B_index) - 5:max(y_B_index) + 6] = \
            mask_A_face[:, :, min(x_B_index) - 5:max(x_B_index) + 6, min(y_B_index) - 5:max(y_B_index) + 6]



        # mask_A_temp = mask_A_temp.squeeze(0)

        mask_A = mask_A.squeeze(0)
        mask_B = mask_B.squeeze(0)
        
        mask_A_face = mask_A_face.squeeze(0)
        # mask_B_temp = mask_B_temp.squeeze(0)

        return mask_A_temp, mask_B_temp

    def mask_preprocess(self, mask_A, mask_B):
        # pdb.set_trace()
        # mask_A = mask_A.unsqueeze(0)
        # mask_B = mask_B.unsqueeze(0)
        index_tmp = torch.nonzero(mask_A, as_tuple=False)
        x_A_index = index_tmp[:, 2]

        y_A_index = index_tmp[:, 3]
        index_tmp = torch.nonzero(mask_B, as_tuple=False)
        x_B_index = index_tmp[:, 2]
        y_B_index = index_tmp[:, 3]
        index = [x_A_index, y_A_index, x_B_index, y_B_index]
        index_2 = [x_B_index, y_B_index, x_A_index, y_A_index]
        mask_A = mask_A.squeeze(0)
        mask_B = mask_B.squeeze(0)
        return mask_A, mask_B, index, index_2

    def to_var(self, x, requires_grad=True):
        if torch.cuda.is_available():
            x = x.cuda()
        if not requires_grad:
            return Variable(x, requires_grad=requires_grad)
        else:
            return Variable(x)


