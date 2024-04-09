#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 11:26:56 2024

@author: pp
"""

import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

df = pd.read_excel('./REC020BT3.xlsx')
df.head()

# Imprime las primeras filas del DataFrame
# print(df.head())

# Obtiene información sobre el DataFrame
# df.info()

# Obtiene las dimensiones del DataFrame
# print(df.shape)

# Ordenar datos
df['Description'] = df['Description'].str.strip()
df.dropna(axis=0, subset=['InvoiceNo'], inplace=True)
df['InvoiceNo'] = df['InvoiceNo'].astype('str')
df = df[~df['InvoiceNo'].str.contains('C')]

# Datos de Francia
basket = (df[df['Country'] == 'France']
          .groupby(['InvoiceNo', 'Description'])['Quantity']
          .sum().unstack().reset_index().fillna(0)
          .set_index('InvoiceNo'))

#basquet.head()
def encode_units(x):
    if x <= 0:
        return False
    if x >= 1:
        return True

basket_sets = basket.map(encode_units).astype(bool)
basket_sets.drop('POSTAGE', inplace=True, axis=1)
frequent_itemsets = apriori(basket_sets, min_support=0.07, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

# rules.head()
rules[(rules['lift'] >= 6) & (rules['confidence'] >= 0.8)]

# rules.head(n=22)
basquet2 = (df[df['Country'] == 'Germany']
            .groupby(['InvoiceNo', 'Description'])['Quantity']
            .sum().unstack().reset_index().fillna(0)
            .set_index('InvoiceNo'))
basket_sets2 = basquet2.map(encode_units).astype(bool)
basket_sets2.drop('POSTAGE', inplace=True, axis=1)
frequent_itemsets2 = apriori(basket_sets2, min_support=0.05, use_colnames=True)
rules2 = association_rules(frequent_itemsets2, metric="lift", min_threshold=1)
rules2[(rules2['lift'] >= 4) & (rules2['confidence'] >= 0.5)]
rules2.head(n=22)