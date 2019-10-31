# -*- coding: utf-8 -*-
"""
@author: sio277(shoh4486@naver.com)
"""
import os
import tensorflow as tf

def create_directory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)     
    except OSError:
        print('Error: Creating directory. ' + directory)

def global_variables_list():
    return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

def pixel_checker(image_set):
    """
    image_set: set(image)
    """
    if (0 in image_set) and (1 in image_set) and (len(image_set) == 2):
        pass
    else:
        raise ValueError('Label map should include only 0 and 1.')