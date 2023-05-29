import torch
import torch.nn as nn
import torchvision
from torchvision import models, transforms, utils
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
from PIL import Image
import json
#%matplotlib inline


transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),])
    
image = Image.open(r'C:\Users\CVPR\Desktop\demo\Featuremap\011.jpg')
#plt.imshow(image)
#image.show()
# image = image.convert('L') 
from model import CSRNet
model = CSRNet()
print(model)

# we will save the conv layer weights in this list
model_weights =[]
#we will save the 49 conv layers in this list
conv_layers = []
# get all the model children as list
model_children = list(model.children())
#counter to keep count of the conv layers
counter = 0
#append all the conv layers and their respective wights to the list
for i in range(len(model_children)):
    if type(model_children[i]) == nn.Conv2d:
        counter+=1
        model_weights.append(model_children[i].weight)
        conv_layers.append(model_children[i])
    elif type(model_children[i]) == nn.Sequential:
        # for j in range(len(model_children[i])):
        for child in model_children[i].children():
            if type(child) == nn.Conv2d:
                counter+=1
                model_weights.append(child.weight)
                conv_layers.append(child)
# print(f"Total convolution layers: {counter}")
# print(conv_layers)
# print (model_weights)


# take a look at the conv layers and the respective weights
for weight, conv in zip(model_weights, conv_layers):
    # print(f"WEIGHT: {weight} \nSHAPE: {weight.shape}")
    print(f"CONV: {conv} ====> SHAPE: {weight.shape} ===> LAYERS {len(conv_layers)}")



# plt.figure(figsize=(120, 126))
# for i, filter in enumerate(model_weights[0]):
#     # print(i, filter.size())
#     plt.subplot(9, 9, i+1) # (8, 8) because in conv0 we have 7x7 filters and total of 64 (see printed shapes)
#     plt.imshow(filter[0, :, :].detach(), cmap='gray')
#     plt.axis('off')
#     plt.savefig('C:/Users/CVPR/source/repos/Detection/CityCam/feature_map_kernels.jpg')


img = np.array(image)
# print(img)
# apply the transforms
img = transform(img)
print(img.size())
# # unsqueeze to add a batch dimension
img = img.unsqueeze(0)
print(img.size())

# pass the image through all the layers
results = [conv_layers[0](img)]
for i in range(1, len(conv_layers)):
    # pass the result from the last layer to the next layer
    results.append(conv_layers[i](results[-1]))
# make a copy of the `results`
outputs = results
s=0
# visualize 64 features from each layer 
# (although there are more feature maps in the upper layers)
for num_layer in range(len(outputs)):
    # plt.figure(figsize=(240, 352))
    layer_viz = outputs[num_layer][0, :, :, :]
    layer_viz = layer_viz.data
    print(layer_viz.size())
    for i, filter in enumerate(layer_viz):
        if i == len(outputs): # we will visualize only 8x8 blocks from each layer
            # break
            # print (i)
            s=s+1
            # plt.subplot(1, 1, i + 1)
            plt.imshow(filter, cmap='jet')
            plt.axis("off")
            plt.show()
            # plt.imsave('D:/featuremap/feature_maps_'+str(num_layer)+str(s)+'.jpg',filter)
    # plt.savefig('D:/featuremap/feature_maps_'+str(num_layer)+'.jpg',filter)