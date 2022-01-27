import tensorflow as tf
import pandas as pd

import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Rescaling, Flatten, BatchNormalization, Dropout
from keras.layers import Conv2D, MaxPooling2D


class BaseConvNet():
    
    def train(self, epochs, dataset_dir, validation_size, export_data=False, gpu=True, verbose=True):
        '''
        Entrena la red construida con el conjunto de datos especificado

        Parámetros:
            epochs (int): Número de épocas del entrenamiento
            dataset_dir (str): Dirección del conjunto de datos
            validation_size (float): Tamaño del conjunto de datos de validación
            gpu (bool): Uso de GPU para el entrenamiento
            export_data (bool): Escribe en un fichero de texto el resultado del entrenamiento y guarda el plot
            verbose (bool): mostrar resultados del entrenamiento en tiempo real

        Retorno:
            No retorna nada
        ''' 
        
        self.__split_datasets(dataset_dir, validation_size)

        if not gpu:
            print('Using CPU')
            tf.device('/cpu:0')
        
        else:
            device_name = tf.test.gpu_device_name()
            if device_name != '':
                print('Using GPU: ' + device_name)
                tf.device(device_name)
            
            else:
                print('GPU not found. Using CPU')
            
        train_result = self.model.fit(self.train_data, validation_data=self.validation_data, epochs=epochs, verbose=verbose)
        
        if export_data:
            self.__write_data(train_result)

    def show_model(self):
        ''' Imprime el modelo construido '''
        self.model.summary()

    def __split_datasets(self, dataset_dir, validation_size):
        self.train_data = tf.keras.utils.image_dataset_from_directory(
            dataset_dir,
            validation_split=validation_size,
            subset="training",
            seed=123,
            image_size=(self.layer_image_shape[0], self.layer_image_shape[1])
        )

        self.validation_data = tf.keras.utils.image_dataset_from_directory(
            dataset_dir,
            validation_split=1-validation_size,
            subset="validation",
            seed=123,
            image_size=(self.layer_image_shape[0], self.layer_image_shape[1])
        )
    
    def __write_data(self, train_result):
        layers = self.__get_model_layers()

        # Escribe los datos en CSV
        df = pd.DataFrame(data={'loss': train_result.history['loss'], 
                                'val_loss': train_result.history['val_loss'],
                                'accuracy': train_result.history['accuracy'],
                                'val_accuracy': train_result.history['val_accuracy']
                                })
        
        df.to_csv('train_results' + str(layers['conv']) + '.csv', sep=',', index=False)

        # Guarda las gráficas de los resultados de entrenamiento
        plt.plot(train_result.history['accuracy'])
        plt.plot(train_result.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')

        plt.savefig('accuracy_' + str(layers['conv']))
            
        plt.plot(train_result.history['loss'])
        plt.plot(train_result.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.savefig('loss_' + str(layers['conv']))

    def __get_model_layers(self):
        layers_dict = {
            'conv': 0,
            'max_pooling': 0
        }

        for layer in self.model.layers:
            if isinstance(layer, Conv2D):
                layers_dict['conv'] = layers_dict['conv'] + 1
            
            elif isinstance(layer, MaxPooling2D):
                layers_dict['max_pooling'] = layers_dict['max_pooling'] + 1
        
        return layers_dict


class StandarCNN(BaseConvNet):

    def __init__(self, classes, layer_image_shape=(180,180,3), convolution_layers=1):
        '''
        Construye una red neuronal con número de capas convolucionales pasadas como parámetro

        Atributos:
            classes (int): número de clases a clasificar
            layer_image_shape (tuple): dimensión de la primera capa que será igual 
                                       que la dimensión de la imágen
            convolution_layers (int): número de capas convolucionales de la red
        '''

        self.convolution_layers = convolution_layers
        self.layer_image_shape = layer_image_shape
        
        # Se crea el modelo a partir de los parámetros introducidos
        self.model = Sequential()
        self.model.add(Rescaling(scale=1./255, input_shape=layer_image_shape))

        for i in range(1, convolution_layers+1):
            self.model.add(Conv2D(i*32, (3,3), activation='relu'))
            self.model.add(MaxPooling2D(pool_size=(2,2)))
        
        self.model.add(Flatten())
        self.model.add(Dense(1000, activation='relu'))
        self.model.add(Dense(classes, activation='softmax'))
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='Adam',metrics=['accuracy'])
        self.show_model()


class ResNet50CNN(BaseConvNet):

    def __init__(self, classes, layer_image_shape=(180,180,3)):
        '''
        Construye una red neuronal resnet50 preentrenada con imagenet

        Atributos:
            layer_image_shape (tuple): dimensión de la primera capa que será igual 
                                       que la dimensión de la imágen
            classes (int): número de clases a clasificar
        '''
        
        self.layer_image_shape = layer_image_shape
        
        # Creación de RN50
        rs_50 = tf.keras.applications.resnet50.ResNet50(weights='imagenet', include_top=False, input_shape=layer_image_shape)
        self.model = Sequential()
        self.model.add(rs_50)
        self.model.add(Flatten())
        self.model.add(BatchNormalization())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(BatchNormalization())
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(BatchNormalization())
        self.model.add(Dense(classes, activation='softmax'))
        self.show_model()
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='Adam',metrics=['accuracy'])


