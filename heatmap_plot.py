import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

img=Image.open(r'D:\Soumi\License plate detection\images\Cars75.png')
map=np.load(r'D:\Soumi\License plate detection\1\1\Cars75_dm.npy')
plt.imshow(img)
plt.imshow(map,alpha=0.9,cmap='jet')
plt.show()
