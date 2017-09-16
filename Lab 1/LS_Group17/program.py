#to compile run $python3 program.py < input
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math


#calculating g of x and returning value
def g(i, *x) :
	
	res=-0.5*( ( np.dot( np.dot( np.transpose( x-meanvec[i-1] ),( cov[i-1] )),(x-meanvec[i-1]))) + math.log(np.linalg.det(cov[i-1]))+math.log(len(xi[i-1])/sumoflen) )
	#res=-0.5*(trans(x-mui)*inv(covmat(i))*(x-mu(i))+math.log(det(covmat(i)))) #Still incomplete(will return value of g)
	return res
	
#comparing values of each class and returning corespondingly
def gp(x, classname):#compare g for all classes
	if (classname)==3 :
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
	elif (classname)==2 :
		if g(1, *x)>g(2, *x):
			return 1
		else :
			return 2
	else :
		return 1


#calculating mean vector
def singlematrix(x,y):
	arr=[]
	meanx=(np.sum(x))/len(x)
	meany=(np.sum(y))/len(y)
	arr.append(meanx)
	arr.append(meany)
	return np.array(arr) 


#calculating covariance matrix
def covmatrix(x,y):
	arr=[]
	sigmaxx=0
	sigmaxy=0
	sigmayx=0
	sigmayy=0
	meanx=(np.sum(x))/len(x)
	meany=(np.sum(y))/len(y)
	for i in xrange(0,len(x)):
		sigmaxx+=(x[i]-meanx)**2
		sigmaxy+=(x[i]-meanx)*(y[i]-meany)
		sigmayx+=(y[i]-meany)*(x[i]-meanx)
		sigmayy+=(y[i]-meany)**2

	arr.append([sigmaxx/len(x),sigmaxy/len(x)])
	arr.append([sigmayx/len(x),sigmayy/len(x)])
	return np.array(arr)


alltrain=[]													#list to store name of all classes
Classes=int(raw_input("Number of classes to see : "))		#no. of classes

print("Name of All required classes  : ")
for i in range(Classes):
	cn=raw_input()
	cn="LS_Group17/"+cn
	alltrain.append(cn)

xi=[]														#xi vector storing x coordinate of train data
yi=[]														#yi vector storing y coordinate of train data
sumoflen=0

for i in xrange(0,Classes):
	x, y=np.loadtxt(alltrain[i] , delimiter=' ', unpack=True)
	xi.append(x)
	yi.append(y)
	sumoflen+=len(xi)


xites=[]													#xites vector storing x coordinate of test data
yites=[]													#yites vector storing y coordinate of test data

for i in xrange(0,Classes):
	var=alltrain[i].find("n")
	testfile="NLS_Group17/"+"Test"+str(alltrain[i][var+1])+".txt"
	x, y=np.loadtxt(testfile , delimiter=' ', unpack=True)
	xites.append(x)
	yites.append(y)

output='Class'
for i in xrange(0,Classes):
	var=alltrain[i].find("n")
	output+=str(alltrain[i][var+1])
filen=output


meanvec=[]													#mean vector

for i in xrange(0,Classes):
	meanvec.append(singlematrix(xi[i] ,yi[i] ))

#print (xvec1)

cov=[]														#covariance matrix

for i in xrange(0,Classes):
	cov.append(covmatrix(xi[i] ,yi[i] ))

confu=[]
for i in xrange(0,Classes):											#confusion matrix
	confu.append([])
	for j in xrange(0,Classes):
		confu[i].append(0)

confu=np.array(confu)


# print (cov)

#covavg=(cov+cov2+cov1)/3
#print (covavg)

# print(xvec1[0])


# gi1=giofx(meanvec1,cov1)
# gi2=giofx(meanvec2,cov2)
# gi3=giofx(meanvec3,cov3)




CaseNo=int(raw_input("Case number : \n"))

#find the range of graph to be plotted
maxx=float('-inf')
minx=float('inf')
maxy=float('-inf')
miny=float('inf')
for i in range(len(alltrain)):
	for j in open(alltrain[i]):
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




title=''

xaxis=[]							#x value for contour
yaxis=[]							#y value for contour
gaxis=[]							#z value for contour

plt.figure(0)

#All the cases as asked in the question

print output

if CaseNo==1:
	

	for i in xrange(0,len(cov)):
		cov[i][0][0]=(cov[i][0][0]+cov[i][1][1])/2
		cov[i][1][1]=(cov[i][0][0]+cov[i][1][1])/2
		cov[i][1][0]=cov[i][0][1]=0
	cova=[[0.0,0.0],[0.0,0.0]]
	cova=np.array(cova)
	for i in xrange(0,len(cov)):
		cova+=cov[i]
	cova/=len(cov)

	for i in xrange(0,len(cov)):
		cov[i] = cova
	

	for i in range(3*minx, 3*maxx):#decision region plot
		gaxis.append([])
		xaxis.append(i)
		for j in range(3*miny, 3*maxy):
			x=list(range(2))
			x[0]=(1.0*i)/3
			x[1]=(1.0*j)/3
			if i==minx:
				yaxis.append(j)

			reftoclass=(gp(x,Classes))#gi(x) will be called over here and passed to graph for plotting
			
			gaxis[i-3*minx].append(max(g(1,*x),g(2,*x)))

			color=''
			
			if reftoclass==1 :
				color="red"
			elif reftoclass==2 :
				color="blue"
			elif reftoclass==3 :
				color="green"
			
			plt.scatter(x[0] , x[1] , c=color,alpha=0.8)

			

	output= output + "_" +"Case1.png"
	title='Case1'
	
	for i in xrange(0,Classes):
		for j in range(0,len(xites[i])):
			x[0]=xites[i][j]
			x[1]=yites[i][j]
			val=gp(x,Classes)
				
			confu[i][val-1]+=1
				# else :
				# 	if i==0:
				# 		confu[0][1]+=1
				# 	else :
				# 		confu[1][0]+=1
	print confu
		# print('\n')

elif CaseNo==2:#rest of the cases
	print("Second Case is on!")

	cova=[[0.0,0.0],[0.0,0.0]]
	
	cova=np.array(cova)

	for i in xrange(0,len(cov)):
		cova+=cov[i]
	cova/=len(cov)

	for i in xrange(0,len(cov)):
		cov[i] = cova

	for i in range(3*minx, 3*maxx):#decision region plot
		gaxis.append([])
		xaxis.append(i)
		for j in range(3*miny, 3*maxy):
			x=list(range(2))
			x[0]=(1.0*i)/3
			x[1]=(1.0*j)/3
			if i==minx:
				yaxis.append(j)

			reftoclass=(gp(x,Classes))#gi(x) will be called over here and passed to graph for plotting
			
			gaxis[i-3*minx].append(max(g(1,*x),g(2,*x)))
			

			color=''
			
			if reftoclass==1 :
				color="red"
			elif reftoclass==2 :
				color="blue"
			elif reftoclass==3 :
				color="green"
			
			plt.scatter(x[0] , x[1] , c=color,alpha=0.8)
			
	output=output + "_" +"Case2.png"
	title='Case2'
	for i in xrange(0,Classes):
		for j in range(0,len(xites[i])):
			x[0]=xites[i][j]
			x[1]=yites[i][j]
			val=gp(x,Classes)
				
			confu[i][val-1]+=1

	print confu


elif CaseNo==3:
	print("Third Case is on!")

	for i in xrange(0,len(cov)):
		cov[i][0][0]=(cov[i][0][0]+cov[i][1][1])/2
		cov[i][1][1]=(cov[i][0][0]+cov[i][1][1])/2
		cov[i][1][0]=cov[i][0][1]=0

	for i in range(3*minx, 3*maxx):#decision region plot
		gaxis.append([])
		xaxis.append(i)
		for j in range(3*miny, 3*maxy):
			x=list(range(2))
			x[0]=(1.0*i)/3
			x[1]=(1.0*j)/3
			if i==minx:
				yaxis.append(j)

			reftoclass=(gp(x,Classes))#gi(x) will be called over here and passed to graph for plotting
			
			gaxis[i-3*minx].append(max(g(1,*x),g(2,*x)))
			

			color=''
			
			if reftoclass==1 :
				color="red"
			elif reftoclass==2 :
				color="blue"
			elif reftoclass==3 :
				color="green"
			
			plt.scatter(x[0] , x[1] , c=color,alpha=0.8)
			
	output=output + "_" +"Case3.png"
	title='Case3'

	for i in xrange(0,Classes):
		for j in range(0,len(xites[i])):
			x[0]=xites[i][j]
			x[1]=yites[i][j]
			val=gp(x,Classes)
				
			confu[i][val-1]+=1

	print confu


elif CaseNo==4:
	print("Fourth Case is on!")
	
	for i in range(3*minx, 3*maxx):#decision region plot
		gaxis.append([])
		xaxis.append(i)
		for j in range(3*miny, 3*maxy):
			x=list(range(2))
			x[0]=(1.0*i)/3
			x[1]=(1.0*j)/3
			if i==minx:
				yaxis.append(j)

			reftoclass=(gp(x,Classes))#gi(x) will be called over here and passed to graph for plotting
			
			gaxis[i-3*minx].append(max(g(1,*x),g(2,*x)))
			
			color=''
			
			if reftoclass==1 :
				color="red"
			elif reftoclass==2 :
				color="blue"
			elif reftoclass==3 :
				color="green"
			
			plt.scatter(x[0] , x[1] , c=color,alpha=0.8)
			
	output=output + "_" +"Case4.png"
	title='Case4'

	for i in xrange(0,Classes):
		for j in range(0,len(xites[i])):
			x[0]=xites[i][j]
			x[1]=yites[i][j]
			val=gp(x,Classes)
				
			confu[i][val-1]+=1

	print confu


# plt.scatter(x1, y1, 'co', label='Class1!')

print output

dignal=[]

confu=np.array(confu)

accuracy 	=	(1.0*(sum(confu.diagonal())))/( np.sum(confu) )
percision=[]
recall=[]
fmeasure=[]
rowq=np.sum(confu,axis=1)
colq=np.sum(confu,axis=0)
for i in xrange(0,Classes):
	
	percision.append((1.0*confu[i][i])/(rowq[i]))
	recall.append((1.0*confu[i][i])/(colq[i]))

	fmeasure.append(float((percision[i]*recall[i])/(2.0*(percision[i]+recall[i]))))

filenl=filen + "_" + "_Case" + str(CaseNo) +".txt"
fo=open(filenl,"w")
for i in xrange(Classes):
	for j in range(0,Classes):
		fo.write(str(confu[i][j]))
		fo.write("\t")
	fo.write("\n")
stri="accuracy :"+str(accuracy)+"\n"
fo.write(stri)

for i in xrange(0,Classes):
	stri="Class"+str(i+1)+"\n"
	fo.write(stri)
	stri="percision : " + str(percision[i])+"\n"
	fo.write(stri)
	stri="recall : " + str(recall[i])+"\n"
	fo.write(stri)
	stri="f-score : "+str(fmeasure[i])+"\n"
	fo.write(stri)

mpercision=(1.0*sum(percision)/Classes)
stri="mean percision :"+str(mpercision)+"\n"

fo.write(stri)

mrecall=(1.0*sum(recall)/Classes)
stri="mean recall :"+str(mrecall)+"\n"
fo.write(stri)
	
fo.close()

color=['firebrick','midnightblue','lawngreen']
for i in xrange(0,Classes):
	var	=	alltrain[i].find("n")
	clas="Class"+str(alltrain[i][var+1])+"!"
	plt.scatter(xites[i], yites[i], c=color[i], label=clas,alpha=0.8)
	

plt.xlabel('x')
plt.ylabel('y')
plt.title(title)
plt.legend()
plt.savefig(filename=output)

gaxis=np.array(gaxis)
gaxis=np.transpose(gaxis)

plt.figure(1)
for i in xrange(0,Classes):
	var	=	alltrain[i].find("n")
	clas="Class"+str(alltrain[i][var+1])+"!"
	plt.scatter(xites[i], yites[i], c=color[i], label="cddcsx",alpha=0.7)

CS =plt.contour(xaxis,yaxis,gaxis,c="black",label="cddcsx")

plt.clabel(CS, fontsize=5, inline=True)
plt.title('Contour')

plt.savefig(filename="cont_"+output)
