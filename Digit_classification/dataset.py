import idx2numpy
import numpy as np
import random

# ----------------------------------------------------------------------- #
# Función that fetchs MNIST data set and labels as numpy arrays. 
# Inputs:
#   path: path where the MNIST data set is located
# Outputs:
#   train_images: numpy matrix of size (num_train_images, 28, 28)
#   train_labels: numpy matrix of size (num_train_images)
#   test_images:  numpy matrix of size (num_test_images, 28, 28)
#   test_labels:  numpy matriz of size (num_test_images)
#
# ----------------------------------------------------------------------- #

def get_MNIST_dataset(path):
    # Path of IDX documents
    train_labels_path = path + '/train-labels.idx1-ubyte'
    train_images_path = path + '/train-images.idx3-ubyte'
    test_labels_path =  path + '/t10k-labels.idx1-ubyte'
    test_images_path =  path + '/t10k-images.idx3-ubyte'

    # Load train and test data set
    train_labels = idx2numpy.convert_from_file(train_labels_path)
    train_images = idx2numpy.convert_from_file(train_images_path)
    test_labels = idx2numpy.convert_from_file(test_labels_path)
    test_images = idx2numpy.convert_from_file(test_images_path)

    # Show shape of data structures
    print("Forma de los datos de entrenamiento:")
    print("Etiquetas de entrenamiento:", train_labels.shape)
    print("Imágenes de entrenamiento:", train_images.shape)
    print()
    print("Forma de los datos de prueba:")
    print("Etiquetas de prueba:", test_labels.shape)
    print("Imágenes de prueba:", test_images.shape)
    print()

    return train_labels, train_images, test_labels, test_images



# ----------------------------------------------------------------------- #
# Function that obtains a given number of images from the MNIST data set 
# and converts it to a unidimentional format (list). It fetchs images
# with uniform distribution of digits. 
# Examples: num_dig -> 100. output -> 10 images per digit. 
# Inputs:
#   labels: numpy arrays with labels, label[0] indicates the digit in the images[0]
#   images: numpy arrays with image data
#   num_dig: number of images to obtain
#   mode: 1 - random
#         2 - first occurrences for each digit.
#         3 - include all data in data set
# Outputs:
#   digits_images: list of lists with image data in unidimentional format. 
#   idx_digit_change: list of indexes that idicate where a set of images
#                     of a different digit start.
# ------------------------------------------------------------------------
def obtain_images(labels, images, mode, num_dig=None):
    if mode != 3:
        # Find out how many images per digit should be obtained
        images_per_digit = num_dig//10
        module = num_dig%10
        # List that indicates images per digit to collect. idx = digit.
        images_per_digit = [images_per_digit for x in range(10)]
        # Find random digits to acquire the module images. 
        indices_aleatorios = random.sample(range(10), module)
        for i in indices_aleatorios:
            images_per_digit[i] += 1
        print(images_per_digit, "images per digit")
        # Generate list where it is stored the index where new digit information start 
        idx_digit_change = []
        for i in range(1,len(images_per_digit)):
            idx_digit_change.append(sum(images_per_digit[:i]))
    
    else: 
        # Generate empty lists
        images_per_digit = [0 for i in range(10)]
        idx_digit_change = []
        

    digits_images = []
    # Iterate between digits and number of images to obtained per digit. 
    for i, n in zip(range(10), images_per_digit):
        if mode == 1:
            # Obtener las primeras n imagenes del digito i 
            idx = np.where(labels == i)[0][:n]

        if mode == 2:
            # Obtener cantidad de imagenes existentes para el digito i
            idx = np.where(labels == i)[0]
            max_imag = images.shape[0]
            # Obtener indices random no repetidos que correspondan a una imagen del dígito
            idx = random.sample(range(max_imag), n)
        
        if mode == 3:
            idx = np.where(labels == i)[0]
            num_img = len(idx)
            images_per_digit[i] = num_img
            idx_digit_change.append(sum(images_per_digit))

        i_images = images[idx]
        for image in i_images:
            image = image.reshape(-1).tolist()
            digits_images.append(image)
    if mode != 3:
        idx_digit_change.append(num_dig)
    return digits_images, idx_digit_change


def get_num_images_per_digit(labels, images):
    for i in range(10):
        # Obtener las primeras n imagenes del digito i 
        idx = np.where(labels == i)[0]
        size = len(idx)
        print(f"digit: {i}, num_imag: {size}")



# test
if __name__ == "__main__":
    train_labels, train_images, test_labels, test_images = get_MNIST_dataset("archive")
    digits_images, idx_digit_change = obtain_images(train_labels, train_images, 3)
    print(len(digits_images))
    print(len(digits_images[0]))
    
    get_num_images_per_digit(train_labels, train_images)
    print(idx_digit_change)

    import numpy as np


    