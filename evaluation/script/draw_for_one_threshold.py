import re
import matplotlib.pyplot as plt
import pandas as pd
Critical_bin_id_numbers = []
HSMU_SpGEMM_Geometric_mean_numbers = []
file_path = "test_one_threshold_python_result.txt"
with open(file_path, 'r') as file:
    lines = file.readlines()
    for line in lines:
        if line.startswith('“Critical_bin_id is'):
            match = re.search(r'Critical_bin_id is (\d+)', line)
            if match:
                Critical_bin_id_numbers.append(int(match.group(1)))
        if 'HSMU-SpGEMM Geometric mean:' in line:
            match = re.search(r'HSMU-SpGEMM Geometric mean: (\d+(\.\d+)?)', line)
            if match:
                HSMU_SpGEMM_Geometric_mean_numbers.append(float(match.group(1)))

print("Critical_bin_id array:", Critical_bin_id_numbers)
# print("Cnnz_ctile_rate_Threshold 数组:", Cnnz_ctile_rate_Threshold_numbers)
print("HSMU-SpGEMM Geometric mean array:", HSMU_SpGEMM_Geometric_mean_numbers)

sorted_data = sorted(zip(Critical_bin_id_numbers, HSMU_SpGEMM_Geometric_mean_numbers))
sorted_Critical_bin_id_numbers, sorted_HSMU_SpGEMM_Geometric_mean_numbers = zip(*sorted_data)
print("Sorted Critical_bin_id_numbers:", sorted_Critical_bin_id_numbers)
print("Sorted HSMU_SpGEMM_Geometric_mean_numbers:", sorted_HSMU_SpGEMM_Geometric_mean_numbers)
total_result = pd.DataFrame()
total_result[0] = sorted_Critical_bin_id_numbers 
total_result[1] = sorted_HSMU_SpGEMM_Geometric_mean_numbers  
total_result[2] = (128,256,512,1024,2048,4096,8192,12288,24576) 
final_result_file_path='one_threshold_test_result.csv'
column_names = ['bin id', 'HSMU-SpGEMM Geometric mean', 'nnz number']
total_result.to_csv(final_result_file_path, index=False, header=column_names)

plt.figure(figsize=(8, 3))
color1 = '#A9D18E'
color2 = '#AFABAB'
color3 = '#F4B183'
color4 = '#9DC3E6'
color_array = [color1,color2,color3,color4]
x_values = total_result[0]
y_values = total_result[1]
plt.plot(x_values, y_values,color = color_array[0])
for x, y in zip(x_values, y_values):
    plt.text(x, y, f'{y}', ha='center', va='bottom') 
ax = plt.gca()
# plt.xticks(x_ticks, rotation=45,fontsize=14)
ax.tick_params(axis='y', labelsize=12)
ax.set_yticks([15, 20, 25, 30, 35])#设置刻度
ax.set_xticks([5, 6, 7, 8, 9, 10, 11, 12, 13])#设置刻度
ax.set_xticklabels(['128','256','512','1024','2048','4096','8192','12288','24576'],fontsize=12)#设置刻度上的文本
# 添加图例和标签
plt.xlabel('NNZ threshold',fontsize=14)
plt.ylabel('Geometric mean GFLOPS',fontsize=12)
plt.grid(True)
plt.tight_layout()

plt.savefig("line_chat_of_one_threshold.pdf", format='pdf', bbox_inches="tight", pad_inches=0)
