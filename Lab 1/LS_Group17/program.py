import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

def mu(i, *x):
	print("NO!")

def sig():
	print("NO!")

def covmat(i, j):
	E=[[]]
	E[0][0]=sig(i, i)
	E[0][1]=sig(i, j)
	E[1][0]=E[0][1]
	E[1][1]=sig(j, j)

def matmul(*x):
	print("NO!")

def g(i, *x):
	res=-0.5*(trans(x-mui)*inv(covmat(i))*(x-mu(i))+math.log(det(covmat(i)))	#Still incomplete
	return res

def gp(*x):
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

allClasses=[]
Classes=int(input("Number of classes? "))
print("All required classes? ")
for i in range(Classes):
	cn=input()
	allClasses.append(cn)

CaseNo=int(input("Case number? \n"))

if CaseNo==1:
	maxx=float('-inf')
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
	minx=int(minx-3)
	maxx=int(maxx+3)
	miny=int(miny-3)
	maxy=int(maxy+3)
	for i in range(minx, maxx):
		for j in range(miny, maxy):
			x=list(range(2))
			x[0]=i
			x[1]=j
			print(gp(*x), end=' ')
		print('\n')
elif CaseNo==2:
	print("Second Case is on!")
elif CaseNo==3:
	print("Third Case is on!")
elif CaseNo==4:
	print("Fourth Case is on!")