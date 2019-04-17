# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 23:51:57 2018

@author: Ashutosh
"""
# 4 data out of 1399 were NaN. So analysed Topics of only 1395 news 
"""
 0: 'people, state, infection', 
 1: 'patient, art, treatment', 
 2: 'blood, hospital, doctors', 
 3: 'children, positive, living', 
 4: 'virus, drugs'

0 : 274
1 : 579
2 : 287
3 : 150
4 : 105

"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

x = [0, 1, 2, 3, 4]
y = [274, 579, 287, 150, 105]
y_pos = np.arange(len(x))
fig, ax = plt.subplots(figsize=(16, 8))
ax.bar(x,y,color=['#1b9e77','#d95f02','#7570b3','#e7298a','#66a61e'])
plt.xlabel('Classes', fontsize=12)
plt.ylabel('Number of articles', fontsize=12)
plt.xticks(y_pos, x)
plt.title("Number of articles published by TOI under 5 categories")
C0 = mpatches.Patch(color='#1b9e77',label='people, state, infection')
C1 = mpatches.Patch(color='#d95f02',label='patient, art, treatment')
C2 = mpatches.Patch(color='#7570b3',label='blood, hospital, doctors')
C3 = mpatches.Patch(color='#e7298a',label='children, positive, living')
C4 = mpatches.Patch(color='#66a61e',label='virus, drugs')
plt.legend(handles=[C0,C1,C2,C3,C4], loc=0)
plt.show()
plt.savefig('ClusterFreq.png')