"""my-app: A Flower / TensorFlow app."""

import os

import keras
from keras import layers

from flwr_datasets.partitioner import IidPartitioner

from keras.api.models import *
from keras.api.layers import RandomFlip, RandomRotation

from my_app.delper import *
from my_app.delper2 import * 

from typing import Dict, Any, Iterator
from datasets import Dataset, Features, Array4D, Array2D

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

def load_model():
    """ Defines which model to use HMDB51 or UCF101 specific one. Comment out the other """
    cnxsmall = ConvNeXtSmall()
    cwd = pathlib.Path.cwd()
    cnxsmall.load_weights(f'{cwd.parent}/convnextv1small.h5')
    mm = cnxsmall

    kcv_bone1 = Model(mm.inputs, mm.layers[-2].output)

    h = 0
    for layer in kcv_bone1.layers[:-1]:
        h=h+1

    trn = round(h*.9)
    for layer in kcv_bone1.layers[:trn]:
        layer.trainable = False
        
    kcv_bone1.compile(optimizer = keras.optimizers.Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

    dmv  = 224

    # UCF101 DATASET
    #input_img = tf.keras.layers.Input(shape=(None,dmv,dmv,3))
    #y1 = tf.keras.layers.TimeDistributed( RandomFlip("horizontal_and_vertical"))(input_img)  

    #y2 = tf.keras.layers.TimeDistributed( RandomRotation(0.5))(y1)  

    #yd = layers.SpatialDropout3D(0.5)(input_img)

    #y3 = layers.concatenate([input_img,y1,y2,yd],1)

    #y0 = tf.keras.layers.TimeDistributed(kcv_bone1)(y3)
    
    #y0 = tf.keras.layers.GlobalAveragePooling1D()(y0)

    #y0 = layers.Dropout(0.9)(y0)
    #y0= layers.Dense(1024 , activation='gelu')(y0)
    #y0 = layers.Dropout(0.5)(y0)


    #y= layers.Dense(101 , activation='softmax')(y0)


    # HMDB51 DATASET
    input_img = tf.keras.layers.Input(shape=(None,dmv,dmv,3))
    y1 = tf.keras.layers.TimeDistributed( RandomFlip("horizontal_and_vertical"))(input_img)  
    
    y2 = tf.keras.layers.TimeDistributed( RandomRotation(0.2))(y1)  
    
    yd = SpatialDropout3D(0.5)(input_img)
    
    y3 = concatenate([input_img,y1,y2,yd],1)
    
    y0 = tf.keras.layers.TimeDistributed(kcv_bone1)(y3)
     
    y0 = tf.keras.layers.GlobalAveragePooling1D()(y0)
    
    y0= Dense(1024 , activation='gelu')(y0)
    y0 = Dropout(0.5)(y0)
    
    
    y= Dense(51 , activation='softmax')(y0)
    
    cls_bone = Model(input_img, y)
    cls_bone.compile(optimizer = 'adam',
                  #loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
                  loss = tf.keras.losses.CategoricalCrossentropy(from_logits = True),
                  metrics=['accuracy'])

    return cls_bone


class DataGenerator:
    def __init__(self, images_array: np.ndarray, labels_array: np.ndarray):
        """
        Initialize the data generator with image and label arrays.
        
        Args:
            images_array: Numpy array of images with shape (n_samples, height, width, channels)
            labels_array: Numpy array of labels with shape (n_samples,)
        """
        self.images_array = images_array
        self.labels_array = labels_array
        self.length = len(images_array)
    
    def __call__(self) -> Iterator[Dict[str, Any]]:
        """
        Generate dataset items one at a time to save memory.
        
        Yields:
            Dictionary containing an image and its corresponding label
        """
        for idx in range(self.length):
            yield {
                'img': self.images_array[idx].tolist(),  # Directly yield the numpy array
                #'label': self.labels_array[idx].astype('float32').reshape(1, 101).tolist()
                'label': self.labels_array[idx].astype('float32').reshape(1, 51).tolist()
            }


fds = None  # Cache FederatedDataset
def load_data(partition_id, num_partitions):
    # Download and partition dataset
    global fds 
    if fds is None:
        cwd = pathlib.Path.cwd().parent
        fds = IidPartitioner(num_partitions=num_partitions)

        images_array = np.load(cwd / 'my-app/data/hmdb/distilled.npy')
        labels_array = np.load(cwd / 'my-app/data/hmdb/labels.npy')
        
        data_gen = DataGenerator(images_array, labels_array)

        features = Features({
            'img': Array4D(shape=(1, 224, 224, 3), dtype='float32'),
            #'label': Array2D(shape=(1,101), dtype='float32')
            'label': Array2D(shape=(1,51), dtype='float32')
        })
        
        dataset = Dataset.from_generator(
            generator=data_gen,
            features=features
            )
        
        fds.dataset = dataset
    
    partition = fds.load_partition(partition_id)
    partition.set_format("numpy")

    # Divide data on each node: 80% train, 20% test
    partition = partition.train_test_split(test_size=0.2)
    x_train, y_train = partition["train"]["img"], partition["train"]["label"]
    x_test, y_test = partition["test"]["img"], partition["test"]["label"]

    return x_train, y_train.squeeze(), x_test, y_test.squeeze()

# testing purposes
if __name__ == "__main__":
    data = load_data(partition_id=0, num_partitions=10)
    print(data[1].shape)
