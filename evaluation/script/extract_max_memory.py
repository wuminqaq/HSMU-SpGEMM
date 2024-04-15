with open('gpu_memory_usage.txt', 'r') as file:
    lines = file.readlines()

memory_usage = {}
current_matrix = None

for line in lines:
    line = line.strip()
    if not line:
        continue
    if line.endswith('result is below:'):
        current_matrix = line.split()[0]
        memory_usage[current_matrix] = []
    else:
        memory_usage[current_matrix].append(int(line))

max_memory_usage = {}
for matrix, usages in memory_usage.items():
    max_memory_usage[matrix] = max(usages)
output_filename = 'max_memory_usage.txt'
with open(output_filename, 'w') as output_file:
    for matrix, max_usage in max_memory_usage.items():
        output_file.write(f"{matrix}: {max_usage}\n")

for matrix, max_usage in max_memory_usage.items():
    print(f"{matrix}: {max_usage}")
