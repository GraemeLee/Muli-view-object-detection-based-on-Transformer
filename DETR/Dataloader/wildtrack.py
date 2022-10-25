'''
datasets/wildrack.py

this script generates a simple dataloader for the wildrack dataset
'''

import json
import os
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset

class Wildtrack(Dataset):
    def __init__(self, cfg):
        self.fps = 10
        self.gt_fps = 2
        #get the annotations from gt_dir
        self.gt_dir = cfg['gt_dir']

        #get imges from different 'Image_subsets'
        self.im_dir = cfg['base_path'] + '/Image_subsets'

        self.files = sorted(os.listdir(self.gt_dir))

    def __getitem__(self, index):

        #get annotation of index
        file = open(self.gt_dir + self.files[index], 'r')

        #load the annotation of index
        frame = json.load(file)
        anns = {}
        for person in frame:
            person_id = person['personID']
            for view in person['views']:
                if view['xmin'] == -1 and view['ymin'] == -1: # BB doesn't exist
                    continue
                cam_id = view['viewNum'] + 1
                xmin = max(view['xmin'], 0)
                ymin = max(view['ymin'], 0)
                xmax = min(view['xmax'], 1920)
                ymax = min(view['ymax'], 1080)
                w = xmax - xmin
                h = ymax - ymin
                bb_data = [xmin, ymin, w, h, person_id]
                if cam_id not in anns:
                    #
                    anns[cam_id] = [bb_data]
                else: anns[cam_id].append(bb_data)
        file.close()

        """anns: an dictionary 
        
         generate the annotations of different person based on camera id """
        #imgs annotations index
        imgs = anns.copy()
        for cam_id in imgs.keys():
            anns[cam_id] = torch.tensor(anns[cam_id])
            fname = self.im_dir + '/C%d/%08d.png' % (cam_id, index * self.fps // self.gt_fps)
            img = plt.imread(fname)
            imgs[cam_id] = img.transpose((2, 0, 1))
        return imgs, anns


    def __len__(self):
        return len(self.files)