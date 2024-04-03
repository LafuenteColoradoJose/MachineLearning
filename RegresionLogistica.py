#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 09:55:43 2024

@author: pp
"""

import random


class Perceptron:

    def __init__(self, sample, exit, learn_rate=0.01, epoch_number=1000, bias=-1):
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
            u = 0
            j = 0

            for i in range(self.col_sample + 1):
                u = u + self.weight[j] * self.sample[i][j]
                y = self.sing(u)
                if y != self.exit[i]:
                    for j in range(self.col_sample + 1):
                        self.weight[j] = self.weight[j] + self.learn_rate * \
                            (self.exit[i] - y) * self.sample[i][j]
                        erro = True
                        epoch_count = epoch_count + 1
            if erro == False:
                print(('\nEpoch:\n', epoch_count))
                print('-----------------\n')
                break

    def sort(self, sample):
        sample.insert(0, self.bias)
        u = 0

        for i in range(self.col_sample + 1):
            u = u + self.weight[i] * sample[i]
            y = self.sing(u)

            if y == -1:
                print(('Ejemplo', sample))
                print('Clasificacion: P1')

            else:
                print(('Ejemplo:', sample))
                print(('Clasificacion: P2'))

    def sing(self, u):
        return 1 if u >= 0 else -1


samples = [
    [1, 4],
    [5, 7],
    [1, 3],
    [6, 9],
    [1, 2],
    [2, 1],
    [8, 4],
    [9, 6],
    [6, 8],
]

exit = [-1, 1-1, 1-1, -1, 1, 1, 1]

network = Perceptron(samples, exit)

network.trainig()

while True:
    sample = []
    for i in range(2):
        sample.insert(i, float(input('Valor: ')))

network.sort(sample)


