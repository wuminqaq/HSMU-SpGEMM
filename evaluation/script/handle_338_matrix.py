import pandas as pd
import numpy as np
with open('./matrix338_list.txt', 'r') as file:
    lines = [line.rstrip('\n') for line in file.readlines()]

df_name = pd.DataFrame({'matrix': lines})
df_name.to_csv('output.csv', index=False)

#  Replace with the actual file path
file_path0 = '/home/wm/NHC_SPGEMM/data/NHC_4080S_result.csv'
file_path1 = '/home/wm/open_source_code/spGEMM/nsparse-master/cuda-c/data/Nsparse_4080s_result.csv'
file_path2 = '/home/wm/open_source_code/spGEMM/OpSparse-main/spECK/data/spECK_4080s_result.csv'
file_path3 = '/home/wm/open_source_code/spGEMM/OpSparse-main/OpSparse/data/OpSparse_result4080s.csv'
file_path4 = '/home/wm/open_source_code/spGEMM/cusparse/data/cusparse_4080S_result.csv'

df0 = pd.read_csv(file_path0, header=None)
df1 = pd.read_csv(file_path1, header=None)
df2 = pd.read_csv(file_path2, header=None)
df3 = pd.read_csv(file_path3, header=None)
df4 = pd.read_csv(file_path4, header=None)
name1 = 'HSMU-SpGEMM'
name2 = 'Nsparse'
name3 = 'spECK'
name4 = 'OpSparse'
name5 = 'cuSPARSE'
total_result = pd.DataFrame()
total_result[0] = df_name  
total_result[1] = 0.0  
total_result[2] = 0.0 
total_result[3] = 0.0  
total_result[4] = 0.0  
total_result[5] = 0.0  

need_column_id0 = 7
need_column_id1 = 7
need_column_id2 = 10
need_column_id3 = 4
need_column_id4 = 5

for index, row in total_result.iterrows():
    matrix_name = row[0] 
    col1 = df0.loc[df0[0] == matrix_name, need_column_id0].values
    if len(col1) > 0:
        total_result.at[index, 1] = float(col1[0])
    col2 = df1.loc[df1[0] == matrix_name, need_column_id1].values
    if len(col2) > 0:
        total_result.at[index, 2] = float(col2[0])

    col3 = df2.loc[df2[0] == matrix_name, need_column_id2].values
    if len(col3) > 0:
        total_result.at[index, 3] = float(col3[0])

    col4 = df3.loc[df3[0] == matrix_name, need_column_id3].values
    if len(col4) > 0:
        total_result.at[index, 4] = float(col4[0])

    col5 = df4.loc[df4[0] == matrix_name, need_column_id4].values
    if len(col5) > 0:
        total_result.at[index, 5] = float(col5[0])

final_result_file_path='Total_4080S_result.xlsx'
final_sheet_name='total_338_result'
column_names = ['matrix', 'HSMU-SpGEMM', 'Nsparse', 'spECK', 'OpSparse', 'cuSPARSE']
# total_result.to_excel(final_result_file_path, index=False, header=False)
total_result.to_excel(final_result_file_path, sheet_name=final_sheet_name, index=False, header=column_names)

y1 = total_result[1]
for i in range(2,6):
    y2 = total_result[i]
    y3 = np.divide(y1, y2, out=np.full_like(y1, np.nan), where=(y1 != 0) & (y2 != 0))
    maxspeedup = np.nanmax(y3)
    max_index = np.nanargmax(y3)
    minspeedup = np.nanmin(y3)
    min_index = np.nanargmin(y3)
    print(f"for {column_names[i]},max_index={max_index}, maxspeedup matrix: {total_result.at[max_index,0]},maxspeedup = {maxspeedup:.2f};min_index={min_index}, minspeedup matrix: {total_result.at[min_index,0]},minspeedup = {minspeedup:.2f}")
     
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
best_methods = total_result.iloc[:, 1:].idxmax(axis=1)
method_counts = best_methods.value_counts()
method_percentages = method_counts / len(best_methods) * 100
print("Best performance ratio for each method:")
print(method_percentages)

second_best_methods = total_result.iloc[:, 1:].apply(lambda row: row.nlargest(2).index[-1], axis=1)
second_method_counts = second_best_methods.value_counts()
second_method_percentages = second_method_counts / len(second_best_methods) * 100
print(second_method_percentages)
third_best_methods = total_result.iloc[:, 1:].apply(lambda row: row.nlargest(3).index[-1], axis=1)
third_method_counts = third_best_methods.value_counts()
third_method_percentages = third_method_counts / len(third_best_methods) * 100
print("Third best performance ratio for each method:")
for i in range(1,6):
    print(f"{column_names[i]}: {third_method_percentages[i]:.2f}")
second_worst_methods = total_result.iloc[:, 1:].apply(lambda row: row.nsmallest(2).index[-1], axis=1)
second_worst_method_counts = second_worst_methods.value_counts()
second_worst_method_percentages = second_worst_method_counts / len(second_worst_methods) * 100
print("second worst performance ratio of each method:")
for i in range(1,6):
    print(f"{column_names[i]}: {second_worst_method_percentages[i]:.2f}")

worst_methods = total_result.iloc[:, 1:].idxmin(axis=1)
worst_method_counts = worst_methods.value_counts()
worst_method_percentages = worst_method_counts / len(worst_methods) * 100
print("The worst performance ratio of each method:")
print(worst_method_percentages)

min_value = np.zeros(6)
for i in range(1,6):
    non_zero_values = total_result.iloc[:,i].replace(0, np.nan)
    min_value[i] = non_zero_values.min()
print(min_value)
segment_size = 50
length = len(total_result.iloc[:, 1])
num_segments = length // segment_size + (1 if length % segment_size != 0 else 0)
geometric_mean_array = np.empty(6)
for i in range(1, 6):
    total_result.iloc[:,i] = total_result.iloc[:,i].replace(0.0, min_value[i])
    geometric_means = []
    for j in range(num_segments):
        start_idx = j * segment_size
        end_idx = min((j + 1) * segment_size, length)
        segment_data = total_result.iloc[start_idx:end_idx, i]
        geometric_mean_segment = np.prod(segment_data) ** (1 / length)
        geometric_means.append(geometric_mean_segment)
    geometric_mean_array[i] = np.prod(geometric_means)
    print(f"{column_names[i]} Geometric mean: {geometric_mean_array[i]:.2f}")

for i in range(2, 6):
    print(f"geometric_mean speedup over {column_names[i]}: {geometric_mean_array[1]/geometric_mean_array[i]:.2f}")


                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              

