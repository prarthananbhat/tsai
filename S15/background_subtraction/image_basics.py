from skimage import io
import matplotlib.pyplot as plt

img_name = "D:/Projects/theschoolofai/datasets/custom_dataset/train/train_beach1_boat1_0_0.jpg"
image = io.imread(img_name)
plt.imshow(image)
plt.show()
print(image.shape)

new_image = image.transpose(2,0,1)
print(new_image.shape)
plt.imshow(new_image.transpose(1,2,0))
plt.show()

print(image)
print(new_image)