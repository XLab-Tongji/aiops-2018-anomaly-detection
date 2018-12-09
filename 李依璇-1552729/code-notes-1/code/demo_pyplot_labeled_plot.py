# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 12:20:27 2018

@author: lenovo
"""

import matplotlib.pyplot as plt

x = range(10)
y1 = [elem*2 for elem in x]
plt.plot(x, y1)

y2 = [elem**2 for elem in x]
plt.plot(x, y2, 'r--')

plt.show()