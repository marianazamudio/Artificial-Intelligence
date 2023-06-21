import numpy as np
import matplotlib.pyplot as plt
import dataset

train_lb, train_im, test_lb, test_im = dataset.get_MNIST_dataset("archive")
digits_images_tr, idx_digits_change_tr = dataset.obtain_images(train_lb, train_im, 3)
digits_images_te, idx_digits_change_te = dataset.obtain_images(test_lb, test_im, 3)

print(idx_digits_change_tr)
print(idx_digits_change_te)
# Tu lista de 784 elementos que representan la imagen en escala de grises
image_data = digits_images_tr[59000]

# Convertir la lista en una matriz de 28x28
image_matrix = np.array(image_data).reshape(28, 28)

# Graficar la imagen
plt.imshow(image_matrix, cmap='gray')
plt.axis('off')  # Para ocultar los ejes x e y
plt.show()