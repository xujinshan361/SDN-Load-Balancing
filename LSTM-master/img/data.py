import csv
import os

file = open('data.csv')

reader = csv.reader(file)
a = list(reader)
#for i in range(len(list(reader))):
#for i in range(6000):
    # print(a[i][1],end=",")
for i in range(6000):
    print(i,end=",")