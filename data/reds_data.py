import torch
import numpy as np
from scipy.spatial.transform import Rotation as ROT
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
import csv
import random
from PIL import Image

class REDSDataLoader(Dataset):
    """ Dataset for loading the RealEstate10K. In this case, images are 
    chosen within a video.
    """

    def __init__(self, train_data_src,train_data_tgt,train_transform_src,train_transform_tgt):
        super(REDSDataLoader, self).__init__()

        self.initialize(train_data_src,train_data_tgt,train_transform_src,train_transform_tgt)

    def initialize(self, train_data_src,train_data_tgt,train_transform_src,train_transform_tgt):
        self.dataroot_src = train_data_src
        self.dataroot_tgt = train_data_tgt
        self.transform_src = train_transform_src
        self.transform_tgt = train_transform_tgt
        self.dataset_size = 240
        self.frame_size = 100        

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()
 
        #frame_no
        #frm = random.randint(0, self.frame_size-1)

        idx = int(idx)

        A, B = None, None
        while A is None or B is None:
            idx = random.randint(0, (self.dataset_size*self.frame_size)-1)
            B = self.load_image(self.dataroot_tgt, idx%self.dataset_size, idx//self.dataset_size, self.transform_tgt) 
            A = self.load_image(self.dataroot_src, idx%self.dataset_size, idx//self.dataset_size, self.transform_src)  

        identity = torch.eye(4)        

        return {'images' : [A, B], 'cameras' : [{'Pinv' : identity, 'P' : identity, 'K' : identity, 'Kinv' : identity},
                                                {'Pinv' : identity, 'P' : identity, 'K' : identity, 'Kinv' : identity}]
        }

    def load_image(self, path, id, frm, transform):
        vid_template = '%%0%dd' % (3)
        vid_path = vid_template % (id)
        img_template = '%%0%dd.png' % (8)
        img_path = img_template % (frm)
        

        try:
            image_path = os.path.join(path, vid_path + '\\' + img_path)
            image = transform(Image.open(image_path))
            return image
        except:
            return None

    def __len__(self):
        return self.dataset_size*self.frame_size 

    def toval(self, epoch):
        pass

    def totrain(self, epoch):
        pass

# train_data_src = "train_blur\\train\\train_blur\\"
# train_data_tgt = "train_sharp\\train\\train_sharp\\"
# train_transform_src = transforms.Compose([
#     transforms.Resize((256,256)),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# train_transform_tgt = transforms.Compose([
#     transforms.Resize((256,256)),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# description_set= REDSDataLoader(train_data_src,train_data_tgt,train_transform_src,train_transform_tgt)
# dataloader = DataLoader(description_set, batch_size=16, shuffle=True, num_workers=0)
# dataiter = iter(dataloader)
# data = dataiter.next()
# print(data['images'][0].shape)