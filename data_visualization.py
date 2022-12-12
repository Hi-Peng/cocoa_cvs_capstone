import matplotlib.pyplot as plt
import pandas as pd

features_data = pd.read_csv('cocoa_features_lab.csv')
colors = {'fullyfermented':'green', 'partialfermented':'orange','underfermented':'yellow','unfermented':'red'}
print(features_data.head(7))

plt.figure(1)
plt.subplot(231)
plt.title("l_kurtosis")  
plt.subplots_adjust(hspace=0.5)
plt.scatter(features_data['index'], features_data['l_kurtosis'], c=features_data['label'].map(colors))

plt.subplot(232)
plt.title("l_skew")  
plt.subplots_adjust(hspace=0.5)
plt.scatter(features_data['index'], features_data['l_skew'], c=features_data['label'].map(colors))

plt.subplot(233)
plt.title("l_tvar")  
plt.subplots_adjust(hspace=0.5)
plt.scatter(features_data['index'], features_data['l_tvar'], c=features_data['label'].map(colors))

plt.subplot(234)
plt.title("l_tmean")  
plt.subplots_adjust(hspace=0.5)
plt.scatter(features_data['index'], features_data['l_tmean'], c=features_data['label'].map(colors))

plt.subplot(235)
plt.title("a_tmean")  
plt.subplots_adjust(hspace=0.5)
plt.scatter(features_data['index'], features_data['a_tmean'], c=features_data['label'].map(colors))

plt.subplot(236)
plt.title("b_tmean")  
plt.subplots_adjust(hspace=0.5)
plt.scatter(features_data['index'], features_data['b_tmean'], c=features_data['label'].map(colors))

plt.show()