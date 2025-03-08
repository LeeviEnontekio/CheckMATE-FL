import tensorflow as tf
import numpy as np
# import pandas as pd
import cv2
import os
import tqdm
import heapq
import datetime
import glob
import random,time
# from einops import*


from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.callbacks import *
from tensorflow.keras import backend as K
from tensorflow.keras.utils import *
from tensorflow.keras.regularizers import * 

# from sklearn.preprocessing import LabelBinarizer
# from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
# import einops

import pathlib
import itertools
# import skvideo.io

from operator import itemgetter 

import random

def agy(parameters, gradients ):
    new_grads = []
    for (params, grads) in zip(parameters, gradients):
 
        ap = .8
        gt = (grads)*(1-ap)  +(params)*ap
        new_grads.append( (gt)  )
    return new_grads  

def msse(y_true, y_pred):
        
    return tf.reduce_mean(tf.keras.losses.mse(y_true, y_pred)) 
    
kld = tf.keras.losses.KLDivergence()    
    
    
def shfl_pair (test_list1, test_list2):
    
    temp = list(zip(test_list1, test_list2))
    random.shuffle(temp)
    res1, res2 = zip(*temp)
    # res1 and res2 come out as tuples, and so must be converted to lists.
    res1, res2 = list(res1), list(res2)
    
    return np.array(res1), np.array(res2)





class CreprocessInput:
    """`rescale_mode` `torch` means `(image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]`, `tf` means `(image - 0.5) / 0.5`"""

    def __init__(self, input_shape=(224, 224, 3), rescale_mode="torch"):
        self.rescale_mode = rescale_mode
        self.input_shape = input_shape[1:-1] if len(input_shape) == 4 else input_shape[:2]

    def __call__(self, image, resize_method="bilinear", resize_antialias=False, input_shape=None):
        input_shape = self.input_shape if input_shape is None else input_shape[:2]
        image = tf.convert_to_tensor(image)
        if tf.reduce_max(image) < 2:
            image *= 255
        image = tf.image.resize(image, input_shape, method=resize_method, antialias=resize_antialias)
        if len(image.shape) == 3:
            image = tf.expand_dims(image, 0)

        if self.rescale_mode == "raw":
            return image
        elif self.rescale_mode == "raw01":
            return image / 255.0
        else:
            return tf.keras.applications.imagenet_utils.preprocess_input(image, mode=self.rescale_mode)

        
        
tproc = CreprocessInput((224, 224, 3), rescale_mode='torch')   

def ran_crop(vido):
    n1,n2,n3,n4 =tf.shape(vido).numpy() 
    dcrop =  tf.image.random_crop(vido, size=(n1,n2//2, n2//2 ,3))
    return  np.array (tf.image.resize(dcrop,[n2,n3]))

def B_data_generatora(data1, data2, batch_size, frame_count, n_class, shuffle=True, toption = True ):              

        num_samples = len(data1)
        
        pairs = list(zip(data1, data2))
        cairs = list(zip(data1, data2))
        
        if shuffle:
            random.shuffle(pairs)
            random.shuffle(cairs)
            
        beta = 0.2
        l_param  = np.random.beta(beta, beta)
    
        for offset in range(0, num_samples, batch_size):
                #print ('startring index: ', offset) 
                # Get the samples you'll use in this batch
                
                
                v_batch = np.array(pairs[offset:offset+batch_size]) [:,0]
                lb_batch = np.array(pairs[offset:offset+batch_size]) [:,1]
                
                cv_batch = np.array(cairs[offset:offset+batch_size]) [:,0]
                clb_batch = np.array(cairs[offset:offset+batch_size]) [:,1]
                
                
                # Initialise X_train and y_train arrays for this batch
                X_train = []
                y_train = []
                # For each example
                for (sample1,sample2, cg1,cg2)  in zip(v_batch, lb_batch,cv_batch, clb_batch):
                   
                    jvd = frames_from_video_file(sample1, frame_count )
                    cvd = frames_from_video_file(cg1, frame_count )
                    
                    if np.random.randint(10)>5:
                        jvd = ran_crop(jvd)
                        cvd = ran_crop(cvd)
                    
                    
                    
                    id1 = to_categorical(sample2, n_class)*(l_param)
                    id2 = to_categorical(cg2, n_class)*(1-l_param)
                    
                    nvd =   np.array([sutmix (vido, l_param) for vido in zip(jvd, cvd)   ])
                    
                    if toption: 
                        X_train.append( tf.image.per_image_standardization(vid_aug (jvd) ) )
                    else:
                        X_train.append( tproc(vid_aug (jvd) ) )
                        
                        
                    y_train.append(to_categorical(sample2, n_class)  )
                    
                    
#                     if np.random.randint(10)>8:
#                         X_train.append( tf.image.per_image_standardization( nvd ) )
#                         y_train.append( id1+id2 )
                    
                   
                    
        
                # Make sure they're numpy arrays (as opposed to lists)
                X_train = np.array(X_train)
          
                y_train = np.array(y_train)
      
                yield X_train, y_train   
        
        
        

def agy(parameters, gradients ):
    new_grads = []
    for (params, grads) in zip(parameters, gradients):
 
        ap = .5
        gt = (grads)*(1-ap)  +(params)*ap
        new_grads.append( (gt)  )
    return new_grads  





def B_data_generator(data1, data2, batch_size, frame_count, n_class, shuffle=True, toption = True ):                 

        num_samples = len(data1)
        
        pairs = list(zip(data1, data2))
        
        if shuffle:
            random.shuffle(pairs)
            
  
        for offset in range(0, num_samples, batch_size):
                #print ('startring index: ', offset) 
                # Get the samples you'll use in this batch
                
                
                v_batch = np.array(pairs[offset:offset+batch_size]) [:,0]
                lb_batch = np.array(pairs[offset:offset+batch_size]) [:,1]
                
                
                
                # Initialise X_train and y_train arrays for this batch
                X_train = []
                y_train = []
                # For each example
                for (sample1,sample2)  in zip(v_batch, lb_batch):
                    
                    
                    if toption: 
                                            
                        s1 =  tf.image.per_image_standardization(frames_from_video_file(sample1, frame_count,frame_step = 10 ))
                        s2 =  tf.image.per_image_standardization(frames_from_video_file(sample1, frame_count,frame_step = 5 ))
                    else:
                                            
                       s1 =  tproc(frames_from_video_file(sample1, frame_count,frame_step = 10 ))
                       s2 =  tproc(frames_from_video_file(sample1, frame_count,frame_step = 5 ))
                    
                    
                    if np.random.randint(10)>5:
                        s1 = ran_crop(s1)
                        s2 = ran_crop(s2)
                        
                    
#                     if np.random.randint(10)>6:
#                         if toption: 
#                              s2 =  tf.image.per_image_standardization(tf.random.shuffle(s2))
#                         else:
#                             s2 =  tproc(tf.random.shuffle(s2))
                    
                    
                    
                    
                    z1 = to_categorical(sample2, n_class) 
    
                    X_train.append( s1 )
                    X_train.append( s2 )
                    y_train.append( z1 )
                    y_train.append( z1 )
        
                # Make sure they're numpy arrays (as opposed to lists)
                X_train = np.array(X_train)
          
                y_train = np.array(y_train)
      
                yield X_train, y_train   


        
        

def B_data_generatoro(data1, data2, batch_size, frame_count, n_class, shuffle=True, toption = True ):             

        num_samples = len(data1)
        
        pairs = list(zip(data1, data2))
        
        if shuffle:
            random.shuffle(pairs)
            
  
        for offset in range(0, num_samples, batch_size):
                #print ('startring index: ', offset) 
                # Get the samples you'll use in this batch
                
                
                v_batch = np.array(pairs[offset:offset+batch_size]) [:,0]
                lb_batch = np.array(pairs[offset:offset+batch_size]) [:,1]
                
                
                
                # Initialise X_train and y_train arrays for this batch
                X_train = []
                y_train = []
                # For each example
                for (sample1,sample2)  in zip(v_batch, lb_batch):

                    #print(sample1, sample2)
                    if toption: 
                         X_train.append(  tf.image.per_image_standardization(frames_from_video_file(sample1, frame_count ))  )
                    else:
                         X_train.append(  tproc(frames_from_video_file(sample1, frame_count ))  )
                    
                    y_train.append(to_categorical(sample2, n_class)  )
        
                # Make sure they're numpy arrays (as opposed to lists)
                X_train = np.array(X_train)
          
                y_train = np.array(y_train)
      
                yield X_train, y_train   

        
                
def is_channel_last(image):
    channel = image.shape[2]
    assert len(image.shape) == 3
    assert channel == 3 or channel == 1
    assert image_data_format() == "channels_last"

def get_rand_bbox(image, l):
    # Note image is channel last
    #is_channel_last(image)
    width = image.shape[0]
    height = image.shape[1]
    r_x = np.random.randint(width)
    r_y = np.random.randint(height)
    r_l = np.sqrt(1 - l)
    r_w = np.int32(width * r_l)
    r_h = np.int32(height * r_l)
    bb_x_1 = np.int32(np.clip(r_x - r_w, 0, width))
    bb_y_1 = np.int32(np.clip(r_y - r_h, 0, height))
    bb_x_2 = np.int32(np.clip(r_x + r_w, 0, width))
    bb_y_2 = np.int32(np.clip(r_y + r_h, 0, height))
    return bb_x_1, bb_y_1, bb_x_2, bb_y_2



def sutmix(imh, lmda):
    vsl1 = imh[0]
    vsl2 = imh[1]
    bx1, by1, bx2, by2  = get_rand_bbox(vsl2, lmda)
    vsl1[bx1:bx2, by1:by2, :] = vsl2[bx1:bx2, by1:by2, :]
    return vsl1
          
    
    

def format_frames(frame, output_size):
  """
    Pad and resize an image from a video.
    
    Args:
      frame: Image that needs to resized and padded. 
      output_size: Pixel size of the output frame image.

    Return:
      Formatted frame with padding of specified output size.
  """
  frame = tf.image.convert_image_dtype(frame, tf.float32)
  frame = tf.image.resize_with_pad(frame, *output_size)
  return frame
class FrameGenerator:
  def __init__(self, path, n_frames,n_cls, training = False):
    """ Returns a set of frames with their associated label. 

      Args:
        path: Video file paths.
        n_frames: Number of frames. 
        training: Boolean to determine if training dataset is being created.
    """
    self.path = path
    self.n_frames = n_frames
    self.frame_step = frame_step
    self.training = training
    self.class_names = sorted(set(p.name for p in self.path.iterdir() if p.is_dir()))
    self.class_ids_for_name = dict((name, idx) for idx, name in enumerate(self.class_names))

  def get_files_and_class_names(self):
    video_paths = list(self.path.glob('*/*.avi'))
    classes = [p.parent.name for p in video_paths] 
    return video_paths, classes

  def __call__(self):
    video_paths, classes = self.get_files_and_class_names()

    pairs = list(zip(video_paths, classes))

    if self.training:
      random.shuffle(pairs)

    for path, idx in pairs:
        video_frames = frames_from_video_file(path, self.n_frames, self.frame_step) 
        #video_frames = decorder(path) 
        #label = (self.class_ids_for_name[name]) # Encode labels
        label = to_categorical (self.class_ids_for_name[idx],n_cls) # Encode labels
        yield video_frames, (label)
        
        
        
        
def frames_from_video_file(video_path, n_frames, output_size = (224,224), frame_step = 6):
  """
    Creates frames from each video file present for each category.

    Args:
      video_path: File path to the video.
      n_frames: Number of frames to be created per video file.
      output_size: Pixel size of the output frame image.

    Return:
      An NumPy array of frames in the shape of (n_frames, height, width, channels).
  """
  # Read each video frame by frame
  result = []
  src = cv2.VideoCapture(str(video_path))  
    
#   pmdm = 300  
#   output_size = (pmdm,pmdm)

  video_length = src.get(cv2.CAP_PROP_FRAME_COUNT)

  need_length = 1 + (n_frames - 1) * frame_step

  if need_length > video_length:
    start = 0
  else:
    max_start = video_length - need_length
    start = random.randint(0, max_start + 1)

  src.set(cv2.CAP_PROP_POS_FRAMES, start)
  # ret is a boolean indicating whether read was successful, frame is the image itself
  ret, frame = src.read()
  result.append(format_frames(frame, output_size))

  for _ in range(n_frames - 1):
    for _ in range(frame_step):
      ret, frame = src.read()
    if ret:
      frame = format_frames(frame, output_size)
      result.append(frame)
    else:
      result.append((result[-1]))
  src.release()
  result = np.array(result)[..., [2, 1, 0]]

  return result       




                
def vid_aug(vio):
    ff1 = np.random.rand(1) 
    
    
    if 0.0 <= ff1 <= .4:
        vio = tf.image.flip_left_right(vio)
        vio = tf.image.random_brightness(vio, 0.2)
        vio = tf.image.random_saturation(vio, 5, 10)
        
    elif 0.41 <= ff1 <= .8:
        vio = tf.image.flip_up_down(vio)
        vio =  tf.image.random_hue(vio, 0.2)
        vio = tf.image.random_contrast(vio, 0.2, 0.5)
    else:
        vio = vio
    return K.clip(vio,0,1)
        
def vid_aug0(vio):
    ff1 = np.random.rand(1) 
    
    
    if 0.0 <= ff1 <= .4:
        vio = tf.image.random_flip_left_right(vio)
        vio = tf.image.random_brightness(vio, 0.2)
        vio = tf.image.random_saturation(vio, 5, 10)
        
    elif 0.41 <= ff1 <= .8:
        vio = tf.image.flip_up_down(vio)
        vio =  tf.image.random_hue(vio, 0.2)
        vio = tf.image.random_contrast(vio, 0.2, 0.5)
    else:
        vio = vio
    return K.clip(vio,0,1)

 
    

def low_hig(vido):
    jim = 224
    e1 = tf.image.resize(vido,[32,32])
    return  tf.image.resize(e1,[jim,jim])    
    
 
        
def B_data_generatora1(data1, data2, batch_size, frame_count, n_class, shuffle=True, toption = True ):              

        num_samples = len(data1)
        
        pairs = list(zip(data1, data2))
        cairs = list(zip(data1, data2))
        
        if shuffle:
            random.shuffle(pairs)
            random.shuffle(cairs)
            
        beta = 0.2
        l_param  = np.random.beta(beta, beta)
    
        for offset in range(0, num_samples, batch_size):
                #print ('startring index: ', offset) 
                # Get the samples you'll use in this batch
                
                
                v_batch = np.array(pairs[offset:offset+batch_size]) [:,0]
                lb_batch = np.array(pairs[offset:offset+batch_size]) [:,1]
                
                cv_batch = np.array(cairs[offset:offset+batch_size]) [:,0]
                clb_batch = np.array(cairs[offset:offset+batch_size]) [:,1]
                
                
                # Initialise X_train and y_train arrays for this batch
                X_train = []
                y_train = []
                # For each example
                for (sample1,sample2, cg1,cg2)  in zip(v_batch, lb_batch,cv_batch, clb_batch):
                   
                    rfrm = np.random.randint(4) + 6
                    
                    jvd = frames_from_video_file(sample1, frame_count,frame_step = rfrm  )
                    cvd = frames_from_video_file(cg1, frame_count,frame_step = rfrm  )
                    
                    if np.random.randint(10)>5:
                        jvd = ran_crop(jvd)
                        cvd = ran_crop(cvd)
                        
                    if np.random.randint(10)>5:
                        jvd = vid_aug (jvd)
                        
#                     if np.random.randint(10)>7:
#                         jvd =  low_hig (jvd)
                    
  
                    
                    if toption: 
                        X_train.append( tf.image.per_image_standardization(  (jvd) ) )
                    else:
                        X_train.append( tproc( (jvd) ) )
                        
                        
                    y_train.append(to_categorical(sample2, n_class)  )
                    
                    
#                     if np.random.randint(10)>8:
#                         X_train.append( tf.image.per_image_standardization( nvd ) )
#                         y_train.append( id1+id2 )
                    
                   
                    
        
                # Make sure they're numpy arrays (as opposed to lists)
                X_train = np.array(X_train)
          
                y_train = np.array(y_train)
      
                yield X_train, y_train   
        
        
        
        
def B_data_generatora2(data1, data2, batch_size, frame_count, n_class, shuffle=True, toption = True ):              

        num_samples = len(data1)
        
        pairs = list(zip(data1, data2))
        cairs = list(zip(data1, data2))
        
        if shuffle:
            random.shuffle(pairs)
            random.shuffle(cairs)
            
        beta = 0.2
        l_param  = np.random.beta(beta, beta)
    
        for offset in range(0, num_samples, batch_size):
                #print ('startring index: ', offset) 
                # Get the samples you'll use in this batch
                
                
                v_batch = np.array(pairs[offset:offset+batch_size]) [:,0]
                lb_batch = np.array(pairs[offset:offset+batch_size]) [:,1]
                
                cv_batch = np.array(cairs[offset:offset+batch_size]) [:,0]
                clb_batch = np.array(cairs[offset:offset+batch_size]) [:,1]
                
                
                # Initialise X_train and y_train arrays for this batch
                X_train = []
                y_train = []
                # For each example
                for (sample1,sample2, cg1,cg2)  in zip(v_batch, lb_batch,cv_batch, clb_batch):
                   
                    rfrm = np.random.randint(4) + 6
                    
                    jvd = frames_from_video_file(sample1, frame_count,frame_step = rfrm  )
                    cvd = frames_from_video_file(cg1, frame_count,frame_step = rfrm  )
                    
                    if np.random.randint(10)>5:
                        jvd = ran_crop(jvd)
                        cvd = ran_crop(cvd)
                        
                    if np.random.randint(10)>5:
                        jvd = vid_aug (jvd)
                    
                    if np.random.randint(10)>5:
                        jvd = np.concatenate([jvd, low_hig (jvd) ],0)
                    else:
                        jvd = np.concatenate([low_hig (cvd), low_hig (jvd) ],0)
                    
  
                    if toption: 
                        X_train.append( tf.image.per_image_standardization(  (jvd) ) )
              
                    else:
                        X_train.append( tproc( (jvd) ) )   
                
                        
                    y_train.append(to_categorical(sample2, n_class)  )
                    
                    
#                     if np.random.randint(10)>8:
#                         X_train.append( tf.image.per_image_standardization( nvd ) )
#                         y_train.append( id1+id2 )
                    
                   
                    
        
                # Make sure they're numpy arrays (as opposed to lists)
                X_train = np.array(X_train)
          
                y_train = np.array(y_train)
      
                yield X_train, y_train    
        
        
        
        
        
        
def B_data_generatora3(data1, data2, batch_size, frame_count, n_class, shuffle=True, toption = True, dimn = True ):              

        num_samples = len(data1)
        
        pairs = list(zip(data1, data2))
        cairs = list(zip(data1, data2))
        
        if shuffle:
            random.shuffle(pairs)
            random.shuffle(cairs)
            
        beta = 0.2
        l_param  = np.random.beta(beta, beta)
    
        for offset in range(0, num_samples, batch_size):
                #print ('startring index: ', offset) 
                # Get the samples you'll use in this batch
                
                
                v_batch = np.array(pairs[offset:offset+batch_size]) [:,0]
                lb_batch = np.array(pairs[offset:offset+batch_size]) [:,1]
                
                cv_batch = np.array(cairs[offset:offset+batch_size]) [:,0]
                clb_batch = np.array(cairs[offset:offset+batch_size]) [:,1]
                
                
                # Initialise X_train and y_train arrays for this batch
                X_train = []
                y_train = []
                # For each example
                for (sample1,sample2, cg1,cg2)  in zip(v_batch, lb_batch,cv_batch, clb_batch):
                   
                    rfrm = np.random.randint(4) + 6
                    
                    jvd = frames_from_video_file(sample1, frame_count,frame_step = rfrm  )
                    cvd = frames_from_video_file(cg1, frame_count,frame_step = rfrm  )
                    
                    if np.random.randint(10)>5:
                        jvd = ran_crop(jvd)
                        cvd = ran_crop(cvd)
                        
                    if np.random.randint(10)>5:
                        jvd = vid_aug (jvd)
                    
                    if np.random.randint(10)>5:
                        jvd = np.concatenate([jvd, low_hig (jvd) ],0)
                    else:
                        jvd = np.concatenate([low_hig (cvd), low_hig (jvd) ],0)
                    
                    
                    if dimn:
                        jvd = jvd
                    else:
                        jvd = tf.image.resize(jvd,[32,32])
                        jvd = tf.image.resize(jvd,[224,224])
                    
  
                    if toption: 
                        X_train.append( tf.image.per_image_standardization(  (jvd) ) )
              
                    else:
                        X_train.append( tproc( (jvd) ) )   
                
                        
                    y_train.append(to_categorical(sample2, n_class)  )
                    
                    
#                     if np.random.randint(10)>8:
#                         X_train.append( tf.image.per_image_standardization( nvd ) )
#                         y_train.append( id1+id2 )
                    
                   
                    
        
                # Make sure they're numpy arrays (as opposed to lists)
                X_train = np.array(X_train)
          
                y_train = np.array(y_train)
      
                yield X_train, y_train                  
        
        
        
        

def B_data_generatorlow(data1, data2, batch_size, frame_count, n_class, f_dim, s_dim, shuffle=True, toption = True ):             

        num_samples = len(data1)
        
        pairs = list(zip(data1, data2))
        
        if shuffle:
            random.shuffle(pairs)
            
  
        for offset in range(0, num_samples, batch_size):
                #print ('startring index: ', offset) 
                # Get the samples you'll use in this batch
                
                
                v_batch = np.array(pairs[offset:offset+batch_size]) [:,0]
                lb_batch = np.array(pairs[offset:offset+batch_size]) [:,1]
                
                
                
                # Initialise X_train and y_train arrays for this batch
                X_train = []
                y_train = []
                # For each example
                for (sample1,sample2)  in zip(v_batch, lb_batch):
                    
                    
                    jvd = frames_from_video_file(sample1, frame_count )
                    ldm = 28
                    jvd = tf.image.resize(jvd,[f_dim, s_dim])
                    jvd = tf.image.resize(jvd,[224,224])
                
                
                   
                    if toption: 
                         X_train.append(  tf.image.per_image_standardization(jvd)  )
                    else:
                         X_train.append(  tproc(jvd)  )
                    
                    y_train.append(to_categorical(sample2, n_class)  )
        
                # Make sure they're numpy arrays (as opposed to lists)
                X_train = np.array(X_train)
          
                y_train = np.array(y_train)
      
                yield X_train, y_train   
        
        
        
def B_data_base_low(data1, data2, batch_size, frame_count, n_class, shuffle=True):             

        num_samples = len(data1)
        
        pairs = list(zip(data1, data2))
        
        if shuffle:
            random.shuffle(pairs)
            
  
        for offset in range(0, num_samples, batch_size):
                #print ('startring index: ', offset) 
                # Get the samples you'll use in this batch
                
                
                v_batch = np.array(pairs[offset:offset+batch_size]) [:,0]
                lb_batch = np.array(pairs[offset:offset+batch_size]) [:,1]
                
                
                
                # Initialise X_train and y_train arrays for this batch
                X_train = []
                y_train = []
                # For each example
                for (sample1,sample2)  in zip(v_batch, lb_batch):
                    
                    
                    jvd = frames_from_video_file(sample1, frame_count )
                    dmj = 32
                    jvd = tf.image.resize(jvd,[dmj,dmj])
  
       
                    X_train.append( jvd  )
                    
                    y_train.append(to_categorical(sample2, n_class)  )
        
                # Make sure they're numpy arrays (as opposed to lists)
                X_train = np.array(X_train)
          
                y_train = np.array(y_train)
      
                yield X_train, y_train          
        
        
        
        

def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    converted_len = int(clip_len * frame_sample_rate)-2
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices      






def B_decoreader_val(data1, data2, batch_size, frame_count,intrvl, n_class, clip_count,  shuffle=True, toption = True ):             

        from decord import VideoReader
        from decord import cpu, gpu
        import decord as de
        
        num_samples = len(data1)
        
        pairs = list(zip(data1, data2))
        
        if shuffle:
            random.shuffle(pairs)
            
  
        for offset in range(0, num_samples, batch_size):
                #print ('startring index: ', offset) 
                # Get the samples you'll use in this batch
                
                
                v_batch = np.array(pairs[offset:offset+batch_size]) [:,0]
                lb_batch = np.array(pairs[offset:offset+batch_size]) [:,1]
                
                
                
                # Initialise X_train and y_train arrays for this batch
                X_train = []
                y_train = []
                # For each example
                for (sample1,sample2)  in zip(v_batch, lb_batch):
                    
                    rdr =  de.VideoReader(sample1, width=224, height=224, ctx=cpu(0))
                    d1 =   rdr.get_batch(range(0, len(rdr) - 1, 3)).asnumpy() 
                    
                    for g99 in range(clip_count):
              
                        video_clip = get_clip(d1, frame_count, intrvl)
                        if toption: 
                            X_train.append(  tf.image.per_image_standardization(video_clip)  )
                        else:
                            X_train.append(  tproc(video_clip)  )
                        y_train.append(to_categorical(sample2, n_class)  )
                      

                # Make sure they're numpy arrays (as opposed to lists)
                X_train = np.array(X_train)
          
                y_train = np.array(y_train)
      
                yield X_train, y_train 
        
        
        
        
def ran_crop(vido):
    n1,n2,n3,n4 =tf.shape(vido).numpy() 
    dcrop =  tf.image.random_crop(vido, size=(n1,n2//2, n2//2 ,3))
    return  np.array (tf.image.resize(dcrop,[n2,n3]))

def vid_auga(vio):
    ff1 = np.random.rand(1) 
    
    
    if 0.0 <= ff1 <= .4:
        vio = tf.image.flip_left_right(vio)
        vio = tf.image.random_brightness(vio, 0.2)
        vio = tf.image.random_saturation(vio, 5, 10)
        
    elif 0.41 <= ff1 <= .8:
        vio = tf.image.flip_up_down(vio)
        vio =  tf.image.random_hue(vio, 0.2)
        vio = tf.image.random_contrast(vio, 0.2, 0.5)
    else:
        vio = ran_crop(vio)
    return K.clip(vio,0,1)


def B_decoreader_aug(data1, data2, batch_size, frame_count,intrvl, n_class, viv=False, shuffle=True, toption = True ):             

        from decord import VideoReader
        from decord import cpu, gpu
        import decord as de
        
        num_samples = len(data1)
        
        pairs = list(zip(data1, data2))
        
        if shuffle:
            random.shuffle(pairs)
            
  
        for offset in range(0, num_samples, batch_size):
                #print ('startring index: ', offset) 
                # Get the samples you'll use in this batch
                
                
                v_batch = np.array(pairs[offset:offset+batch_size]) [:,0]
                lb_batch = np.array(pairs[offset:offset+batch_size]) [:,1]
                
                
                
                # Initialise X_train and y_train arrays for this batch
                X_train = []
                y_train = []
                # For each example
                for (sample1,sample2)  in zip(v_batch, lb_batch):
                    
                    rdr =  de.VideoReader(sample1, width=224, height=224, ctx=cpu(0))
                    d1 =   rdr.get_batch(range(0, len(rdr) - 1, 3)).asnumpy() 
              
                    video_clip = get_clip(d1, frame_count, intrvl)
                    
                    
                    #if viv:  
                    if np.random.choice(10)>6:    
                        h_1 = []
                    
                        for hu in range(frame_count):
                            zlp = get_clip(d1, 6, intrvl)
                            zmg = rearrange(zlp, '(b1 b2)  h w c  -> (b2 h) ( b1 w) c', b1 =2)
                            zmg = np.uint8(tf.image.resize(zmg, [224, 224]))
                            h_1.append(zmg)
               
                        video_clip = np.array(h_1)
                    
                    video_clip = np.uint8((vid_auga(video_clip/255))*255)

                    if toption: 
                         X_train.append(  vmae_norm(video_clip/255)  )  
                         #X_train.append(  tf.image.per_image_standardization(video_clip)  )
                          
                    else:
                         X_train.append(  tproc(video_clip)  )
                    
                    y_train.append(to_categorical(sample2, n_class)  )
        
                # Make sure they're numpy arrays (as opposed to lists)
                X_train = np.array(X_train)
          
                y_train = np.array(y_train)
      
                yield X_train, y_train  
                
            
            
def B_decoreader_augr(data1, data2, batch_size, frame_count,intrvl, n_class,  shuffle=True, toption = True ):             

        from decord import VideoReader
        from decord import cpu, gpu
        import decord as de
        
        num_samples = len(data1)
        
        pairs = list(zip(data1, data2))
        
        if shuffle:
            random.shuffle(pairs)
            
  
        for offset in range(0, num_samples, batch_size):
                #print ('startring index: ', offset) 
                # Get the samples you'll use in this batch
                
                
                v_batch = np.array(pairs[offset:offset+batch_size]) [:,0]
                lb_batch = np.array(pairs[offset:offset+batch_size]) [:,1]
                
                
                
                # Initialise X_train and y_train arrays for this batch
                X_train = []
                y_train = []
                # For each example
                for (sample1,sample2)  in zip(v_batch, lb_batch):
                    
                    rdr =  de.VideoReader(sample1, width=224, height=224, ctx=cpu(0))
                    d1 =   rdr.get_batch(range(0, len(rdr) - 1, 3)).asnumpy() 
              
                    video_clip = get_clip(d1, frame_count, intrvl)
                    
                    
         
                    
                    if np.random.choice(10)>6:
                    
                       kq=[]
                       for g1 in range(6):
                            kq.append(get_clip(d1, frame_count, intrvl)/255)
                            
                       video_clip = np.uint8(np.mean(kq,0)*255)
                    
                    

                    
                     
                    if np.random.choice(10)>7:
                          video_clip = np.uint8((vid_auga(video_clip/255))*255)
                
                 
                      
                    
                    if toption: 
                         X_train.append(  tf.image.per_image_standardization(video_clip)  )
                    else:
                         X_train.append(  tproc(video_clip)  )
                    
                    y_train.append(to_categorical(sample2, n_class)  )
        
                # Make sure they're numpy arrays (as opposed to lists)
                X_train = np.array(X_train)
          
                y_train = np.array(y_train)
      
                yield X_train, y_train              

def B_decoreader(data1, data2, batch_size, frame_count,intrvl, n_class,  shuffle=True, toption = True ):             

        from decord import VideoReader
        from decord import cpu, gpu
        import decord as de
        
        num_samples = len(data1)
        
        pairs = list(zip(data1, data2))
        
        if shuffle:
            random.shuffle(pairs)
            
  
        for offset in range(0, num_samples, batch_size):
                #print ('startring index: ', offset) 
                # Get the samples you'll use in this batch
                
                
                v_batch = np.array(pairs[offset:offset+batch_size]) [:,0]
                lb_batch = np.array(pairs[offset:offset+batch_size]) [:,1]
                
                
                
                # Initialise X_train and y_train arrays for this batch
                X_train = []
                y_train = []
                # For each example
                for (sample1,sample2)  in zip(v_batch, lb_batch):
                    
                    if str(sample2.dtype) =='<U9':
                        sample2 = sample2 
                    else:
                        sample2 = to_categorical(sample2, n_class) 
                    
                    rdr =  de.VideoReader(sample1, width=224, height=224, ctx=cpu(0))
                    d1 =   rdr.get_batch(range(0, len(rdr) - 1, 3)).asnumpy() 
              
                    video_clip = get_clip(d1, frame_count, intrvl)
                      
                    
                    if toption: 
                         X_train.append(  vmae_norm(video_clip/255)  )  
                         #X_train.append(  tf.image.per_image_standardization(video_clip)  )
                    else:
                         X_train.append(  tproc(video_clip)  )
            
                    y_train.append(sample2)
             
 
        
                # Make sure they're numpy arrays (as opposed to lists)
                X_train = np.array(X_train)
          
                y_train = np.array(y_train)
      
                yield X_train, y_train  
        

        
        
        
def B_decoreadery(data1, data2, clps, batch_size, frame_count,intrvl, n_class,  shuffle=True, toption = True ):             

        from decord import VideoReader
        from decord import cpu, gpu
        import decord as de
        
        num_samples = len(data1)
        
        pairs = list(zip(data1, data2))
        
        if shuffle:
            random.shuffle(pairs)
            
  
        for offset in range(0, num_samples, batch_size):
                #print ('startring index: ', offset) 
                # Get the samples you'll use in this batch
                
                
                v_batch = np.array(pairs[offset:offset+batch_size]) [:,0]
                lb_batch = np.array(pairs[offset:offset+batch_size]) [:,1]
                
                
                
                # Initialise X_train and y_train arrays for this batch
                X_train = []
                y_train = []
                # For each example
                for (sample1,sample2)  in zip(v_batch, lb_batch):
                    
                    if str(sample2.dtype) =='<U9':
                        sample2 = sample2 
                    else:
                        sample2 = to_categorical(sample2, n_class) 
                    
                    rdr =  de.VideoReader(sample1, width=224, height=224, ctx=cpu(0))
                    d1 =   rdr.get_batch(range(0, len(rdr) - 1, 1)).asnumpy() 
                    for my in range(clps):
                        intrvl = np.random.uniform(.99,.6,1)
                        video_clip = get_clip(d1, frame_count, intrvl)
                                       
                        if toption: 
                             X_train.append(  vmae_norm(video_clip/255)  )  
                             #X_train.append(  tf.image.per_image_standardization(video_clip)  )
                        else:
                             X_train.append(  tproc(video_clip)  )
                      
     
            
                    y_train.append(sample2)
             
 
        
                # Make sure they're numpy arrays (as opposed to lists)
                X_train = np.array(X_train)
                X_train = tf.reshape(X_train , [batch_size,clps, frame_count, 224, 224, 3])
                y_train = np.array(y_train)
      
                yield X_train, y_train          
        

# def get_clip(in_stream, clip_len, gap_int):
#     vd1 = in_stream
#     clip_d = clip_len
#     dg = len(vd1)
#     gap = round((dg//clip_d)*gap_int)
#     fid = sample_frame_indices(clip_d, gap , dg)
#     v_clip = np.array([vd1[i] for i in fid])
#     return v_clip


def get_clip(in_stream, clip_len, gap_int):
    vd1 = in_stream
    d1,d2,d3,d4 = vd1.shape
    clip_d = clip_len
    dg = len(vd1)
    if dg<100:
        mpt = round(100/dg)
        jq = []
        
        for i in range(mpt):
            jq.append(vd1)
        nvd1 = np.array(jq)
        vd1 = np.reshape(nvd1,[dg*mpt,d2,d3,d4])
        
    dg = len(vd1)
    #print(vd1.shape)
    gap = np.round((dg//clip_d)*gap_int)
    fid = sample_frame_indices(clip_d, gap , dg)
    #print(gap, dg, fid)
    v_clip = np.array([vd1[i] for i in fid])
    return v_clip




 

def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    converted_len = int(clip_len * frame_sample_rate)-2
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices


def shfl_pair2 (test_list1, test_list2):
    
    dsm = len(test_list2)
    
    scl = (list(range(1,  dsm)))
    random.shuffle(scl)
    
    res1 = np.array([test_list1[i] for i in scl])
    res2 = np.array([test_list2[i] for i in scl])
    
    return np.array(res1), np.array(res2)


def video2image(vido, patch_size):
    
    def reconstruct_from_patch( patch, patch_size):
        # This utility function takes patches from a *single* image and
        # reconstructs it back into the image. This is useful for the train
        # monitor callback.
        num_patches = patch.shape[0]
        n = int(np.sqrt(num_patches))
        patch = tf.reshape(patch, (num_patches, patch_size, patch_size, 3))
        rows = tf.split(patch, n, axis=0)
        rows = [tf.concat(tf.unstack(x), axis=1) for x in rows]
        reconstructed = tf.concat(rows, axis=0)
        return reconstructed
    
    
    
    
    
     
    patches = tf.image.extract_patches(vido, sizes=[1, patch_size, patch_size, 1],
                                   strides=[1, patch_size, patch_size, 1],rates=[1, 1, 1, 1], padding="VALID",)
    
    resize = Reshape((-1, patch_size * patch_size * 3))

    pa  = resize(patches)
    
    d1 = []


    for i in range(pa.shape[1]):
        w1 = np.random.choice(len(vido))
        d1.append(pa[w1,i,:])
    
    d1 = np.array(d1)
    return reconstruct_from_patch( d1, patch_size)




def B_decoreader_img(data1, data2,patch_size, batch_size, frame_count,intrvl, n_class,  shuffle=True, toption = True ):             

        from decord import VideoReader
        from decord import cpu, gpu
        import decord as de
        
        num_samples = len(data1)
        
        pairs = list(zip(data1, data2))
        
        if shuffle:
            random.shuffle(pairs)
            
  
        for offset in range(0, num_samples, batch_size):
                #print ('startring index: ', offset) 
                # Get the samples you'll use in this batch
                
                
                v_batch = np.array(pairs[offset:offset+batch_size]) [:,0]
                lb_batch = np.array(pairs[offset:offset+batch_size]) [:,1]
                
                
                
                # Initialise X_train and y_train arrays for this batch
                X_train = []
                y_train = []
                # For each example
                for (sample1,sample2)  in zip(v_batch, lb_batch):
                    
                    rdr =  de.VideoReader(sample1, width=224, height=224, ctx=cpu(0))
                    d1 =   rdr.get_batch(range(0, len(rdr) - 1, 3)).asnumpy() 
              
                    video_clip = get_clip(d1, frame_count, intrvl)
                    if np.random.choice(10)>7:
                        video_clip = np.uint8((vid_auga(video_clip/255))*255)
                      
                    
                    if toption: 
                         X_train.append( video2image  (tf.image.per_image_standardization(video_clip), patch_size)  )
                    else:
                         X_train.append(  video2image  (tproc(video_clip), patch_size)  )
                    
                    y_train.append(to_categorical(sample2, n_class)  )
        
                # Make sure they're numpy arrays (as opposed to lists)
                X_train = np.array(X_train)
          
                y_train = np.array(y_train)
      
                yield X_train, y_train  
        
        
#######################################################

def B_decord_img(data1, data2, batch_size, frame_count,intrvl, n_class, valv =True, shuffle=True, toption = True ):             

        from decord import VideoReader
        from decord import cpu, gpu
        import decord as de
        
        num_samples = len(data1)
        
        pairs = list(zip(data1, data2))
        
        if shuffle:
            random.shuffle(pairs)
            
  
        for offset in range(0, num_samples, batch_size):
                #print ('startring index: ', offset) 
                # Get the samples you'll use in this batch
                
                
                v_batch = np.array(pairs[offset:offset+batch_size]) [:,0]
                lb_batch = np.array(pairs[offset:offset+batch_size]) [:,1]
                
                
                
                # Initialise X_train and y_train arrays for this batch
                X_train = []
                y_train = []
                # For each example
                for (sample1,sample2)  in zip(v_batch, lb_batch):
                    
                    rdr =  de.VideoReader(sample1, width=224, height=224, ctx=cpu(0))
                    d1 =   rdr.get_batch(range(0, len(rdr) - 1, 3)).asnumpy() 
              
                    video_clip = get_clip(d1, frame_count, intrvl)
                    if valv:
                        video_clip = np.uint8((vid_auga(video_clip/255))*255)
                        
                    zmg = rearrange(video_clip, '(b1 b2)  h w c  -> (b2 h) ( b1 w) c', b1 =2)
                    zmg = np.uint8(tf.image.resize(zmg, [224, 224]))
                    
                    video_clip = np.array(zmg)
                    
                    if toption: 
                         X_train.append(  tf.image.per_image_standardization(video_clip)[0]  )
                    else:
                         X_train.append(  tproc(video_clip)[0]  )
                    
                    y_train.append(to_categorical(sample2, n_class)  )
        
                # Make sure they're numpy arrays (as opposed to lists)
                X_train = np.array(X_train)
          
                y_train = np.array(y_train)
      
                yield X_train, y_train  
                



#######################################################

def B_decoreader_clpmg(data1, data2,patch_size, batch_size,frame_count,intrvl, n_class, clip_count,shuffle=True,toption = True ):             

        from decord import VideoReader
        from decord import cpu, gpu
        import decord as de
        
        num_samples = len(data1)
        
        pairs = list(zip(data1, data2))
        
        if shuffle:
            random.shuffle(pairs)
            
  
        for offset in range(0, num_samples, batch_size):
                #print ('startring index: ', offset) 
                # Get the samples you'll use in this batch
                
                
                v_batch = np.array(pairs[offset:offset+batch_size]) [:,0]
                lb_batch = np.array(pairs[offset:offset+batch_size]) [:,1]
                
                
                
                # Initialise X_train and y_train arrays for this batch
                X_train = []
                y_train = []
                # For each example
                for (sample1,sample2)  in zip(v_batch, lb_batch):
                    
                    rdr =  de.VideoReader(sample1, width=224, height=224, ctx=cpu(0))
                    d1 =   rdr.get_batch(range(0, len(rdr) - 1, 3)).asnumpy() 
                    
                    for g99 in range(clip_count):
              
                        video_clip = get_clip(d1, frame_count, intrvl)
                        video_clipa = np.uint8((vid_auga(video_clip/255))*255)
                    
                        video_clip = np.concatenate([ video_clip, video_clipa])
                
                         
                        if toption: 
                             X_train.append( video2image  (tf.image.per_image_standardization(video_clip), patch_size)  )
                        else:
                             X_train.append(  video2image  (tproc(video_clip), patch_size)  )
                            
                            
                        y_train.append(to_categorical(sample2, n_class)  )
                      

                # Make sure they're numpy arrays (as opposed to lists)
                X_train = np.array(X_train)
          
                y_train = np.array(y_train)
      
                yield X_train, y_train 
                      
            
            
            
            
def B_decoreader_mulimg(data1, data2,patch_size, batch_size, frame_count, im_count, intrvl, n_class,  shuffle=True, toption = True ):             

        from decord import VideoReader
        from decord import cpu, gpu
        import decord as de
        
        num_samples = len(data1)
        
        pairs = list(zip(data1, data2))
        
        if shuffle:
            random.shuffle(pairs)
            
  
        for offset in range(0, num_samples, batch_size):
                #print ('startring index: ', offset) 
                # Get the samples you'll use in this batch
                
                
                v_batch = np.array(pairs[offset:offset+batch_size]) [:,0]
                lb_batch = np.array(pairs[offset:offset+batch_size]) [:,1]
                
                
                
                # Initialise X_train and y_train arrays for this batch
                X_train = []
                y_train = []
                # For each example
                for (sample1,sample2)  in zip(v_batch, lb_batch):
                    
                    rdr =  de.VideoReader(sample1, width=224, height=224, ctx=cpu(0))
                    d1 =   rdr.get_batch(range(0, len(rdr) - 1, 3)).asnumpy() 
              
                    video_clip = get_clip(d1, frame_count, intrvl)
                    if np.random.choice(10)>7:
                        video_clip = np.uint8((vid_auga(video_clip/255))*255)
                      
                    for zx in range(im_count):
                        if toption: 
                             X_train.append( video2image  (tf.image.per_image_standardization(video_clip), patch_size)  )
                        else:
                             X_train.append(  video2image  (tproc(video_clip), patch_size)  )
                    
                        y_train.append(to_categorical(sample2, n_class)  )
        
                # Make sure they're numpy arrays (as opposed to lists)
                X_train = np.array(X_train)
#                 a1,a2,a3,a4,a5 = tf.shape(X_train).numpy() 
#                 X_train = tf.reshape(X_train,[a1*a2, a3,a4,a5])
                
                
                y_train = np.array(y_train)
#                 b1,b2,b3  = tf.shape(y_train).numpy() 
#                 y_train = tf.reshape(y_train,[b1*b2, b3 ])
      
                yield X_train, y_train  
                    

            
            
            
################################

def rand_bbox(size, lamb):
    """ Generate random bounding box 
    Args:
        - size: [width, breadth] of the bounding box
        - lamb: (lambda) cut ratio parameter, sampled from Beta distribution
    Returns:
        - Bounding box
    """
    W = size[0]
    H = size[1]
 
    cut_rat = np.sqrt(1. - lamb)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def generate_cutmix_image(image_batch, image_batch_labels, beta):
    """ Generate a CutMix augmented image from a batch 
    Args:
        - image_batch: a batch of input images
        - image_batch_labels: labels corresponding to the image batch
        - beta: a parameter of Beta distribution.
    Returns:
        - CutMix image batch, updated labels
    """
    # generate mixed sample
    lam = np.random.beta(beta, beta)
    rand_index = np.random.permutation(len(image_batch))
    target_a = image_batch_labels
    target_b = image_batch_labels[rand_index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(image_batch[0].shape, lam)
    image_batch_updated = image_batch.copy()
    image_batch_updated[:, bbx1:bbx2, bby1:bby2, :] = image_batch[rand_index, bbx1:bbx2, bby1:bby2, :]
    
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (image_batch.shape[1] * image_batch.shape[2]))
    label = target_a * lam + target_b * (1. - lam)
    
    return image_batch_updated, label



def cutmix_vid(image_batch1,image_batch2, lab1,lab2, beta):
 
    lam = np.random.beta(beta, beta)
    target_a = lab1
    target_b = lab2
    bbx1, bby1, bbx2, bby2 = rand_bbox(image_batch1[0].shape, lam)
    image_batch_updated = image_batch1.copy()
    imv= image_batch2.copy()
    image_batch_updated[:, bbx1:bbx2, bby1:bby2, :] = imv[:, bbx1:bbx2, bby1:bby2, :]
    
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (image_batch1.shape[1] * image_batch1.shape[2]))
    label = target_a * lam + target_b * (1. - lam)
    
    return image_batch_updated, label

def mae_norm(vido):
    vs = tf.ones_like(vido)
    m1 = vs*0.45
    m2 = vs*0.225
    vsf = (vido - m1)/m2
    return vsf

def vmae_norm(vido):
    
    m1 = [0.485, 0.456, 0.406] 
    m2 = [0.229, 0.224, 0.225] 
    vsf = (vido - m1)/m2
    return vsf 
            