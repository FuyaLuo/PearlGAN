import os.path, glob
import numpy as np
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform, night_train_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import torch

class EBGCDataset(BaseDataset):
    def __init__(self, opt):
        super(EBGCDataset, self).__init__()
        self.opt = opt
        self.transform = get_transform(opt)
        self.night_edge_transform = night_train_transform(opt)

        datapath = os.path.join(opt.dataroot, opt.phase + '*')
        self.dirs = sorted(glob.glob(datapath))
        if self.opt.isTrain:
            self.IR_edge_paths = opt.IR_edge_path
            self.Vis_edge_paths = opt.Vis_edge_path

        self.paths = [sorted(make_dataset(d)) for d in self.dirs] 
        self.sizes = [len(p) for p in self.paths] 

    def load_image(self, dom, idx):
        path = self.paths[dom][idx]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img, path

    def load_image_train(self, dom, idx):
        path = self.paths[dom][idx]
        img = np.array(Image.open(path).convert('RGB'))
        path_split = path.split('/')
        img_name = path_split[-1]
        if dom == 0:
            edge_map_file = self.Vis_edge_paths + img_name
        else:
            edge_map_file = self.IR_edge_paths + img_name
        edge_map = np.array(Image.open(edge_map_file).convert('L'))

        for func in self.night_edge_transform:
            img, edge_map = func(img, edge_map)
        # img = img.transpose(2, 0, 1) / 255.0
        transform_list_torch = [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        transform_new = transforms.Compose(transform_list_torch)
        # img = torch.from_numpy(img_file.transpose((2, 0, 1)))
        img_new = Image.fromarray(np.uint8(img))
        img_res = transform_new(img_new)
        edge_map = edge_map / 255.0

        return img_res, path, torch.tensor(edge_map)

    def __getitem__(self, index):
        if not self.opt.isTrain:
            if self.opt.serial_test:
                for d,s in enumerate(self.sizes):
                    if index < s:
                        DA = d; break
                    index -= s
                index_A = index
            else:
                DA = index % len(self.dirs)
                index_A = random.randint(0, self.sizes[DA] - 1)
        else:
            DA, DB = 0, 1
            index_A = random.randint(0, self.sizes[DA] - 1)
        
        if self.opt.isTrain:
            A_img, A_path, edge_map_A = self.load_image_train(DA, index_A)
            bundle = {'A': A_img, 'DA': DA, 'path': A_path, 'EMA':edge_map_A}
            index_B = random.randint(0, self.sizes[DB] - 1)
            B_img, _, edge_map_B = self.load_image_train(DB, index_B)
            bundle.update( {'B': B_img, 'DB': DB, 'EMB':edge_map_B} )
        else:
            A_img, A_path = self.load_image(DA, index_A)
            bundle = {'A': A_img, 'DA': DA, 'path': A_path}

        return bundle


    def __len__(self):
        if self.opt.isTrain:
            return max(self.sizes)
        return sum(self.sizes)

    def name(self):
        return 'EBGCDataset'
