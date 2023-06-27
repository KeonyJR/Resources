# import the necessary packages
import numpy as np
import cv2
import os
import sys
import tifffile # Read .tif , with one f.  Opencv only read .tiff, with two f.
import flyr
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def get_current_directory():
    return os.path.dirname(os.path.abspath(__file__))

# Example usage
current_directory = get_current_directory()
directory_root = os.path.join(current_directory, 'Data/Test')
root_dir = os.listdir(directory_root)  # Obtener una lista de los directorios en el directorio raíz
print('----------------')
print(root_dir)
print('----------------')


imagePath =  directory_root +  '/'  + root_dir[1]
print('----------------')
print(imagePath)
print('----------------')

img = tifffile.imread(imagePath)
thermogram = flyr.unpack(directory_root+ "/" + root_dir[3])
thermogramREN=thermogram.render(palette='grayscale')
print('----------------')
print(img.shape)
print('THERMAL METADATA\n')
for key, value in thermogram.metadata.items():
    print(f'{key}: {value}')
print('----------------')

# cv2.imshow('tif', thermogramREN.celsius)
# cv2.waitKey(0) # Wait for a key press and keep the window open
# cv2.destroyAllWindows() # Close all open windows



_, ax = plt.subplots(1, 1, figsize=(10,5))
im =ax.imshow(thermogram.celsius)
#cax = make_axes_locatable(ax[0]).append_axes("right", size="5%", pad=0.05)
plt.colorbar(im,  values=np.unique(thermogram.celsius), label='Temperature (ºC)')
ax.set_title('Thermography image')
ax.set_axis_off()
# ax[1].imshow(thermogram.celsius, cmap='inferno')
# ax[2].imshow(thermogram.celsius, cmap='jet')

plt.show()









# img = tifffile.imread(imagePath)
# cv2.imshow('tif', img)
# cv2.waitKey(0) # Wait for a key press and keep the window open
# cv2.destroyAllWindows() # Close all open windows



# imagePath =  directory_root +  '/'  + root_dir[0]

# image = cv2.imread(imagePath,-1)
# print('----------------')
# print(image)
# print('----------------')
# # print(f'dtype: {image.dtype}, shape: {image.shape}, min: {np.min(image)}, max: {np.max(image)}')


# img_scaled = cv2.normalize(image, dst=None, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX)

# smallest = np.amin(image)
# biggest = np.amax(image)

# print('----------------')
# print('Min: {} - Max: {}'.format(smallest, biggest))
# print('----------------')

# cv2.imshow('tiff', img_scaled)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# print('----------------')
# print(image)
# print('----------------')

