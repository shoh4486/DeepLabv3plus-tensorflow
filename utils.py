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