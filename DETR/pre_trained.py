import torch
model1  = torch.load('F:/Multicamera_detection/DETR/detr-r50-e632da11.pth')

num_class = 2 #我只需要检测一个物体，所以是2（检测个数+background）
model1["model"]["class_embed.weight"].resize_(num_class+1, 256)
model1["model"]["class_embed.bias"].resize_(num_class+1)
#model1["model"].query_embed.num_embeddings = 100
torch.save(model1, "F:\Multicamera_detection\DETR\detr-r50_test_%d.pth"%num_class)
