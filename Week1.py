import io
import csv
import pandas as pd
#import pandas as pd data = pd.read_csv("bwq.csv")
num_attributes=6
a=[]
print("The given Dataset is")
with open("C:\MLDS CSV FILES\Climate1.csv",'r') as csvfile:
 reader=csv.reader(csvfile)
 for row in reader:
  a.append(row)
  print(row)
print("The initial hypothesis is")
hypothesis=['0']*num_attributes
print(hypothesis)
for j in range(0,num_attributes):
 hypothesis[j]=a[1][j]

print(hypothesis)
#Find the Maximum specific Hypothesis
print("Find S: Finding maximal Specific Hypothesis")
for i in range(1,len(a)):
  if a[i][num_attributes]=='Yes' or a[i][num_attributes]== 'yes':
   for j in range(0,num_attributes):
     if a[i][j]!=hypothesis[j]:
       hypothesis[j]='?'
     else :
       hypothesis[j]=a[i][j]
  print("For Training instance No: {0} the hypothesis is ". format(i), hypothesis)
