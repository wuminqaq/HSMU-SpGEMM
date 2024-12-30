import csv

def extract_matrix_categories(matrix_names_file, output_csv):
    # 打开并读取矩阵名称文件
    with open(matrix_names_file, 'r') as file:
        matrix_names = file.read().splitlines()
    
    # 初始化结果列表
    results = []
    
    # 遍历每一个矩阵名称
    for matrix_name in matrix_names:
        mtx_file = f"{matrix_name}.mtx"
        
        try:
            # 打开对应的.mtx文件
            with open(f"/data/total_matrix/{mtx_file}", 'r') as mtx:
                for line in mtx:
                    if line.startswith('% kind:'):
                        # 提取类别信息
                        matrix_kind = line.split(':', 1)[1].strip()
                        results.append((matrix_name, matrix_kind))
                        break
        except FileNotFoundError:
            print(f"File {mtx_file} not found.")
            continue
    
    # 将结果写入CSV文件
    with open(output_csv, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Matrix Name', 'Matrix Kind'])
        csv_writer.writerows(results)

# 使用示例
matrix_names_file = '../script/matrix338_list.txt'  # 替换为你的.txt文件的路径
output_csv = 'extract_categories.csv'   # 输出的CSV文件路径
extract_matrix_categories(matrix_names_file, output_csv)
