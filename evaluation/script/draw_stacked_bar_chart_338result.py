#This script is used to draw Fig.17
#The input of this script is 'new_compressed_step_runtime.csv', which will be generated after processing 338 matrices using HSMU;

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
plt.rcParams['font.size'] = 12
fig_size=(8,4)
#Replace with the actual file path
file_path = '../../data/new_compressed_step_runtime.csv'  
df = pd.read_csv(file_path, header=None) 

name1=6
name2=7
name3=8
name4=9
# categories = df["matrix"]
data1 = df[name1]
data2 = df[name2]
data3 = df[name3]
data4 = df[name4]
total_time = data1 + data2 + data3 + data4

data1_percentage = data1 / total_time 
data2_percentage = data2 / total_time 
data3_percentage = data3 / total_time
data4_percentage = data4 / total_time

color1 = '#A9D18E'
color2 = '#AFABAB'
color3 = '#F4B183'
color4 = '#9DC3E6'

labels_array=("Generates auxiliary mask structure", "Malloc", "Extract and sort Ccol", "Form Cval")
index = np.arange(len(df[name1]))
num_data_groups = len(df[name1])
plt.figure(figsize=(8, 3))
bar_width = 8 / num_data_groups*55
alpha_num=1
plt.bar(index, data1_percentage, bar_width, label=labels_array[0], color=color1, alpha = alpha_num)
plt.bar(index, data2_percentage, bar_width, bottom=np.array(data1_percentage), label=labels_array[1], color=color2, alpha = alpha_num)
plt.bar(index, data3_percentage, bar_width, bottom=np.array(data1_percentage) + np.array(data2_percentage), label=labels_array[2], color=color3, alpha = alpha_num)
plt.bar(index, data4_percentage, bar_width, bottom=np.array(data1_percentage) + np.array(data2_percentage) + np.array(data3_percentage), label=labels_array[3], color=color4, alpha = alpha_num)
print(f"mean of {labels_array[0]}:{np.mean(data1_percentage)}")
print(f"mean of {labels_array[1]}:{np.mean(data2_percentage)}")
print(f"mean of {labels_array[2]}:{np.mean(data3_percentage)}")
print(f"mean of {labels_array[3]}:{np.mean(data4_percentage)}")
ax = plt.gca()
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'], fontsize=10)
plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

plt.xlabel('Matrices', fontsize=14)
plt.ylabel('Proportion', fontsize=14)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=4, labelspacing=0, fontsize=9.2, frameon=False)
plt.ylim(0, 1)
plt.xlim(left=-0.41, right=len(df[name1]) )
plt.show()
plt.tight_layout()
plt.savefig("stacked_bar_338result.pdf", format='pdf', bbox_inches="tight", pad_inches=0)