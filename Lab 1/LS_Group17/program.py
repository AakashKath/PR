#to compile run $python3 program.py < input
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

def mu(i, *x):#will return mean of the required class
	print("NO!")

def sig():#will return variance of the required class
	print("NO!")

def covmat(i, j):#creates covariance matrix
	E=[[]]
	E[0][0]=sig(i, i)
	E[0][1]=sig(i, j)
	E[1][0]=E[0][1]
	E[1][1]=sig(j, j)

def matmul(*x):#matrix multiplication
	print("NO!")

def g(i, *x):
	res=-0.5*(trans(x-mui)*inv(covmat(i))*(x-mu(i))+math.log(det(covmat(i)))#Still incomplete(will return value of g)
	return res

def gp(*x):#compare g for all classes
	if g(1, *x)>g(2, *x):
		if g(1, *x)>g(3, *x):
			return 1
		elif g(1, *x)<g(3, *x):
			return 3
	elif g(1, *x)<g(2, *x):
		if g(2, *x)>g(3, *x):
			return 2
		elif g(2, *x)<g(3, *x):
			return 3

allClasses=[]#list to store name of all classes
Classes=int(input("Number of classes? "))
print("All required classes? ")
for i in range(Classes):
	cn=input()
	allClasses.append(cn)

CaseNo=int(input("Case number? \n"))

if CaseNo==1:#All the cases as asked in the question
	maxx=float('-inf')#find the range of graph to be plotted
	minx=float('inf')
	maxy=float('-inf')
	miny=float('inf')
	for i in range(len(allClasses)):
		for j in open(allClasses[i]):
			if maxx<float(j.split()[0]):
				maxx=float(j.split()[0])
			if minx>float(j.split()[0]):
				minx=float(j.split()[0])
			if maxy<float(j.split()[1]):
				maxy=float(j.split()[1])
			if miny>float(j.split()[1]):
				miny=float(j.split()[1])
	minx=int(minx-3)#increased the limit for better presentation
	maxx=int(maxx+3)
	miny=int(miny-3)
	maxy=int(maxy+3)
	for i in range(minx, maxx):#decision region plot
		for j in range(miny, maxy):
			x=list(range(2))
			x[0]=i
			x[1]=j
			print(gp(*x), end=' ')#gi(x) will be called over here and passed to graph for plotting
		print('\n')
elif CaseNo==2:#rest of the cases
	print("Second Case is on!")
elif CaseNo==3:
	print("Third Case is on!")
elif CaseNo==4:
	print("Fourth Case is on!")