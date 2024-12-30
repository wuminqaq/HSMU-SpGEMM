#This script is used to draw Fig .19 and 20;
#The input of this script is 'conversion_time_and_space_conversion.csv', which will be generated after processing 338 matrices using HSMU;

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#Replace with the actual file path
file_path5 = '../../data/conversion_time_and_space_conversion.csv'
df = pd.read_csv(file_path5, header=None)
selected_columns = df.iloc[:, [0, -2, -1]]
selected_columns.to_csv("NHC_space_cost.csv", index=False, header=None)
selected_columns = df.iloc[:, [0, -4, -3]]
selected_columns.to_csv("NHC_time_cost.csv", index=False, header=None)
df = df.replace(0, 0.00001)
x_col_id=3 
total_rows = df.shape[0]
x = np.arange(0,total_rows)
y1 = (df.iloc[:, -4].values)  
y2 = (df.iloc[:, -3].values)  
y3 = (df.iloc[:, -2].values)  
y4 = (df.iloc[:, -1].values)  
y5 = y4/y3 
y6 = y1/y2
Threshold_value = 1000
size_of_point = 1
plt.rcParams['font.size'] = 14
color1 = '#845EC2'
plt.figure(figsize=(8, 4))

plt.scatter(x[y6 <= Threshold_value], y6[y6 <= Threshold_value], label='Format conversion time / time of a single SpGEMM', alpha=0.9, marker='s', s=size_of_point, c=color1)

ax = plt.gca()
ax.set_yticks([0.0, 0.1 ,0.5, 1.0, 1.5]) 
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.11), ncol=1,labelspacing=0.02, fontsize=15, frameon=False, borderaxespad=0)
plt.xticks([])
ax.tick_params(labelsize=11)
plt.xlabel('Matrices', fontsize=18)
plt.tight_layout()
plt.savefig("time_cost_conversion.pdf", format='pdf', bbox_inches="tight", pad_inches=0)

plt.figure(figsize=(8, 4))
plt.scatter(x[y5 <= Threshold_value], y5[y5 <= Threshold_value], label='Compress mask matrix space / CSR space', alpha=0.9, marker='s', s=size_of_point, c=color1)


ax = plt.gca()
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.11), ncol=2,labelspacing=0.02, fontsize=15, frameon=False, borderaxespad=0)
plt.xticks([])
plt.xlabel('Matrices', fontsize=18)

plt.tight_layout()
plt.savefig("space_cost_conversion.pdf", format='pdf', bbox_inches="tight", pad_inches=0)