# --------------------------------------------------- #
# File: main.py
# Author: Mariana Zamudio Ayala
# Date: 14/06/2023
# Description:
# Program that trains a multi layer
# perceptron to classify digits from 0-9. 
# --------------------------------------------------- #
from interconnected_neuronal_network import InterconNeuralNet
import dataset
import datetime
import numpy as np
import pickle


def save_results(alpha, a, b, eta, num_epochs, num_train_data, num_test_data, num_layers, num_neurons,\
                 initial_weights, final_weights, MSE_values, error_per_tr, error_per_te):
    archivo = open("results.txt", "a")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")  # Generar el marcador de tiempo
    archivo.write(f"---------------------------------------------------------\n")
    archivo.write(f"weights_{timestamp}.npy")
    archivo.write(f"alpha = {alpha}, eta = {eta}\n")
    archivo.write(f"a = {a}, b = {b}\n")
    archivo.write(f"#_layers = {num_layers} -> {num_neurons}\n")
    archivo.write(f"num_epochs = {num_epochs}\n")
    archivo.write(f"#_tr_data = {num_train_data}\n")
    archivo.write(f"#_te_data = {num_test_data}\n")
    #archivo.write(f"initial_weights = {initial_weights}\n")
    archivo.write(f"MSE_values = {MSE_values}\n")
    archivo.write(f"ERROR TRAINNING  = {error_per_tr}\n")
    archivo.write(f"ERROR TEST  = {error_per_te}\n")
    archivo.write(f"final_weights = {final_weights}\n")
    archivo.write("\n")  # Agregar un salto de línea para separar los resultados de cada ejecución
    # Cerrar el archivo
    archivo.close()
    
    # Crear archivo .pkl para guardar los pesos de la red, despues del entrenamiento
    file_name = f"weights_{timestamp}.pkl"  # Nombre del archivo con el marcador de tiempo

    # Guardar todas las matrices en un solo archivo
    with open(file_name, "wb") as archivo:
        pickle.dump(final_weights, archivo)


def load_weight_matrix(file_name):
    # Cargar las matrices desde el archivo en otro programa
    with open(file_name, "rb") as archivo:
        matrices_recuperadas = pickle.load(archivo)
    return matrices_recuperadas

    
if __name__ == "__main__":
    pass
    #save_results(alpha, a, b, eta, num_epochs, num_train_data, num_test_data, num_layers, num_neurons,\
    #                 initial_weights, final_weights, MSE_values, error_per_tr, error_per_te)