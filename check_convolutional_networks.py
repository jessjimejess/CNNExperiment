'''
    Programa para comprobar el comportamiento al aumentar el número de capas 
    convolucionales en redes sencillas 
    
'''

from cnn_models import StandarCNN, ResNet50CNN


if __name__ == "__main__":

    for i in range(2, 5):
        cnn = StandarCNN(5, (180,180,3), i)
        cnn.train(10, 'Fish_Dataset/Fish_Dataset', 0.2, export_data=True)

    cnn50 = ResNet50CNN(9, (180,180,3))
    cnn50.train(1, 'Fish_Dataset/Fish_Dataset', 0.2, export_data=True)