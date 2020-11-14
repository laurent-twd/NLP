import tensorflow as tf
import pandas as pd
from model import MyELECTRA
from utilities.utils import prepare_text_training
import os

path_data = r"/Users/laurentthanwerdas/Documents/Documents/Etudes/NY/Personal/PROJECTS/Deep_Embedded_Clustering/severeinjury.csv"
data = pd.read_csv(path_data, encoding = 'latin9')
corpus = prepare_text_training(data['text'])

parameters = {
    'd_model' : 128,
    'dff' : 512,
    'pe_input' : 150,
    'num_layers' : 12,
    'fitted' : False
}

path = os.getcwd()
model = MyELECTRA(parameters, path_model = os.path.join(path, 'model'))
model.fit(corpus[0], batch_size = 32, epochs = 1)


