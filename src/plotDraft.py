from keras.applications.vgg16 import VGG16
from matplotlib import pyplot
# Load the VGG16 model
model = VGG16()
# Extract filters from the first convolutional layer
filters, biases = model.layers[1].get_weights()
filters = (filters - filters.min()) / (filters.max() - filters.min()) # Normalize
# Visualize the first 6 filters
n_filters, ix = 6, 1
for i in range(n_filters):
   f = filters[:, :, :, i]
   for j in range(3): # Visualize each channel
       ax = pyplot.subplot(n_filters, 3, ix)
       ax.set_xticks([])
       ax.set_yticks([])
       pyplot.imshow(f[:, :, j], cmap='gray')
       ix += 1
pyplot.show()