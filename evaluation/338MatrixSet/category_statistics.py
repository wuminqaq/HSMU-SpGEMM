import csv

def calculate_category_statistics(input_csv, output_csv):
    # 初始化字典来存储每种类别对应的矩阵数量
    category_count = {}

    # 读取extract_categories.csv文件并统计每种类别对应的矩阵数量
    with open(input_csv, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)  # 跳过标题行
        for row in csv_reader:
            matrix_category = row[1]  # 第二列是矩阵类别
            if matrix_category not in category_count:
                category_count[matrix_category] = 1
            else:
                category_count[matrix_category] += 1

    # 将统计结果写入category_statistics_results.csv文件
    with open(output_csv, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Matrix Category', 'Matrix Count'])
        for category, count in category_count.items():
            csv_writer.writerow([category, count])

# 使用示例
input_csv = 'extract_categories.csv'  # 输入的CSV文件路径
output_csv = 'category_statistics_results.csv'  # 输出的CSV文件路径
calculate_category_statistics(input_csv, output_csv)
