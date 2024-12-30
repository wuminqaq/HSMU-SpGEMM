import os
import sys
import csv


filename = "18matrix_list.csv"

total = sum(1 for line in open(filename))
print(total)

with open(filename) as csvfile:
    csv_reader = csv.reader(csvfile)
    header = next(csv_reader)
    for i in range(1, total):
        cur_row = next(csv_reader)
        matrix_name = cur_row[2]
        matrix_url = "https://suitesparse-collection-website.herokuapp.com/MM/" + cur_row[1] + "/" + cur_row[2] + ".tar.gz"
        os.system("wget " + matrix_url)
        os.system("tar --strip-components=1 -zxvf " + matrix_name + ".tar.gz ")
        os.system("rm -rf " + matrix_name + ".tar.gz")
