import matplotlib.pyplot as plt
import pydicom
import numpy as np

# Load images
image1 = pydicom.dcmread('/home/setup/Desktop/CSCcode/BRAIN_MRI_1.dcm', force=True).pixel_array
image2 = pydicom.dcmread('/home/setup/Desktop/CSCcode/BRAIN_MRI_2.dcm', force=True).pixel_array

# Display images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image1, cmap='gray')
plt.title('Image 1')
plt.subplot(1, 2, 2)
plt.imshow(image2, cmap='gray')
plt.title('Image 2')
plt.show()


# Fuse the images by averaging pixel values
fused_image = (image1 + image2) / 2

# Display the fused image
plt.imshow(fused_image, cmap='gray')
plt.title('Fused Image')
plt.show()


# Fourier Transform
f_transform_image1 = np.fft.fft2(image1)
f_transform_image2 = np.fft.fft2(image2)

# Display Fourier Transformed Images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(np.log(np.abs(f_transform_image1)), cmap='gray')
plt.title('Fourier Transform - Image 1')
plt.subplot(1, 2, 2)
plt.imshow(np.log(np.abs(f_transform_image2)), cmap='gray')
plt.title('Fourier Transform - Image 2')
plt.show()


from scipy import ndimage

# Edge enhancement using Sobel filter
edges1 = ndimage.sobel(image1)
edges2 = ndimage.sobel(image2)

# Display edge-enhanced images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(edges1, cmap='gray')
plt.title('Edge Enhanced - Image 1')
plt.subplot(1, 2, 2)
plt.imshow(edges2, cmap='gray')
plt.title('Edge Enhanced - Image 2')
plt.show()


from scipy.ndimage import rotate

# Rotate both images by 45 degrees
rotated_image1 = rotate(image1, 45)
rotated_image2 = rotate(image2, 45)

# Display rotated images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(rotated_image1, cmap='gray')
plt.title('Rotated Image 1')
plt.subplot(1, 2, 2)
plt.imshow(rotated_image2, cmap='gray')
plt.title('Rotated Image 2')
plt.show()
