import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np


#Array for all the models to load
models = []
#len(models)


for i in range(10):
    models.append(tf.keras.models.load_model(f'models{i+1}.h5'))

for i in range(len(models):
    loss, acc = models[i].evaluate(input_test, target_test, verbose=0)
    #print(f"Loss {i+1} ={loss}  ACC {i+1} = {acc} ")
    with open("results.txt","a") as archive:
        archive.write(f"\n Model{i+1} = Loss: {loss}  Acc: {acc} \n")

