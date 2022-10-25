from torch.utils.data import DataLoader

from config import cfg
from datasets.wildtrack import Wildtrack

if __name__ == '__main__':
    wild = Wildtrack(cfg)
    wild = DataLoader(wild)
    for i, (img, anns) in enumerate(wild):
        print(img[1].shape, img[1].dtype, anns[1])