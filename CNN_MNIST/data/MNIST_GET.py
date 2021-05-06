#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 13:31:59 2018
@author: chinwei
"""

import urllib
import pickle
import gzip
import os
import numpy as np


if __name__ == '__main__':

    
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='mnist', 
                        choices=['mnist'],
                        help='dataset name')
    parser.add_argument('--savedir', type=str, default='data', 
                        help='directory to save the dataset')
    
    args = parser.parse_args()
    print(args)
    
    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)
        
    if args.dataset == 'mnist':
        path = 'http://deeplearning.net/data/mnist'
        mnist_filename_all = 'mnist.pkl'
        local_filename = os.path.join(args.savedir, mnist_filename_all)
        urllib.request.urlretrieve("{}/{}.gz".format(path,mnist_filename_all), local_filename+'.gz')
        with gzip.open(local_filename+'.gz', 'rb') as f:
            tr,va,te = pickle.load(f, encoding='latin-1')
        np.save(local_filename, (tr,va,te))
        
        
