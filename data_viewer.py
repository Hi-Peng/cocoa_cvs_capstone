import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

glcm_df = pd.read_csv("cocoa_features.csv")

print(glcm_df.head())

label_distr = glcm_df['label'].value_counts()

label_name = ['fullyfermented', 'partialfermented', 'underfermented', 'unfermented']

plt.figure(figsize=(20,10))

my_circle = plt.Circle( (0,0), 0.7, color='white')
plt.pie(label_distr, 
        labels=label_name,  
        autopct='%1.1f%%')

p = plt.gcf()
p.gca().add_artist(my_circle)
plt.show()
print(label_distr)