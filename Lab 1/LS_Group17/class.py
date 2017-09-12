import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def Prior(i):
    sum=0
    for i in files:
        sum+=files[i]
    return files[i]/sum

def mu1(i):
	filename="Train"+str(i)+".txt"
	sum=0
	count=0
	for line in open(filename):
		sum+=float(line.split(' ')[0])
		count+=1
	return sum/count

def mu2(i):
	filename="Train"+str(i)+".txt"
	sum=0
	count=0
	for line in open(filename):
		sum+=float(line.split(' ')[1])
		count+=1
	return sum/count

def sig1(i):
	filename="Train"+str(i)+".txt"
	mean=mu1(i)
	sum=0
	count=0
	for line in open(filename):
		sum+=(float(line.split(' ')[0])-mean)**2
		count+=1
	return sum/count

def sig2(i):
	filename="Train"+str(i)+".txt"
	mean=mu2(i)
	sum=0
	count=0
	for line in open(filename):
		sum+=(float(line.split(' ')[1])-mean)**2
		count+=1
	return sum/count

def sig(i):
	return (sig1(i)+sig2(i))/2

def g_value(i, x1, x2):
    return -((x1-mu1(i))*(x1-mu1(i))+(x2-mu2(i))*(x2-mu2(i)))/(2*sig)+Prior(i)

files={}
ClsNo=int(input("Number of classes: "))
colors=['y', 'r', 'b', 'g']
print(sig(ClsNo))
'''
while(ClsNo):
    filenumber=input("Please Enter filenumber: ")
    filename="Class"+filenumber+".txt"
    with open(filename) as f:
        files[int(filenumber)]=sum(1 for _ in f)
    df=pd.read_csv(filename, delim_whitespace=True)
    plt.scatter(df[[0]], df[[1]], s=2, c=colors[ClsNo])
    ClsNo-=1

print(Prior(1))
plt.show()
plt.savefig('Train.png')
'''
