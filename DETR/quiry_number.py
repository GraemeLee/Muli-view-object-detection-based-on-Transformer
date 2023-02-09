from sahi.utils.coco import Coco, CocoCategory, CocoImage, CocoAnnotation
from sahi.utils.file import save_json
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

gt_dir = "F:\Multicamera_detection\DETR\Wildtrack_dataset/annotations_positions/"
save_path = "F:/Multicamera_detection/DETR/Wildtrack_dataset/Image_subsets/dataset1/test/"
files = os.listdir(gt_dir)
print(len(files))

number_p = np.zeros((400,7),dtype = int)


 #if cam_id ==0 :
for i,annot  in enumerate(files):
    file = open(gt_dir+'/'+annot,'r')
    file = json.load(file)
    for person in file:
            for view in person['views']:
                for cam_id in range(7):
                    if view['viewNum'] == cam_id:
                        if view['xmin'] == -1 and view['ymin'] == -1:  # BB doesn't exist
                            continue
                        number_p[i][cam_id] += 1
print(number_p,number_p.shape, np.max(number_p))
m=number_p.T
for i in range(7):
    plt.plot(m[i])
plt.legend()
plt.show()

