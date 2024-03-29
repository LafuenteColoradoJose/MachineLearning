#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 09:55:43 2024

@author: pp
"""

import random

class Perceptron:
    
    def _init_(self, sample, exit, learn_rate=0.01, epoch_number=1000, bias=-1):
        self.sample = sample
        self.exit = exit
        self.learn_rate = learn_rate
        self.epoch_number = epoch_number
        self.bias = bias
        self.number_sample = len(sample)
        self.col_sample = len(sample[0])
        self.weight = []
    
    def trainig(self):
        for sample in self.sample:
            sample.insert(0, self.bias)
            
        for i in range(self.col_sample):
            self.weight.append(random.random())
            self.weight.insert(0, self.bias)
            
        epoch_count = 0
        
        while True:
            erro = False
            