#!coding:utf8
from PIL import ImageFilter, Image
import numpy as np 
import time
def paraseUNIPen(rawfilename):
	"""
	input rawfilename: unipen format file, http://archive.ics.uci.edu/ml/machine-learning-databases/pendigits/
	return : allimages =[(label,[]),(label,[]),,,,], label means 0 ... 9 , [] stands for the the images coordinate x,y,x,y arrays
	"""
	f=open(rawfilename)
	start=False
	oldlabel="a"
	allimages=[]
	oneimage=[]
	for line in f:
		if "?" in line:  #get lable non pen down and up times
			line=line[line.rindex("T")+1:]
			line.strip()
			penupdown=line[:line.rindex("?")].strip()
			label=line[line.rindex("?")+1:].strip().replace('"',"")
			# print penupdown, label
			if "-" in  penupdown:
				penstart=int(penupdown.split("-")[0])
				penend=int(penupdown.split("-")[1])
				pendowntime=penend-penstart+1
			else:
				pendowntime=1
			continue
		if "PEN_DOWN" in line:
			start=True
			continue
		if "PEN_UP" in line:
			pendowntime-=1
			if pendowntime==0:
				start=False
				allimages.append((label,oneimage))
				oneimage=[]
			continue
		if ".DT" in line:
			continue

		if start==True:
			data=line.strip().split("  ")
			# print data
			xcord=int(data[0])
			ycord=int(data[1])
			# print xcord,ycord
			oneimage.append(xcord)
			oneimage.append(ycord)
	return allimages,len(allimages)



def digit_enhancement(image,flip_top_bottom=True):
	if flip_top_bottom:
		pic_new=image.transpose(Image.FLIP_TOP_BOTTOM)
	else:
		pic_new=image
	for i in range(1000):
		pic_new=pic_new.filter(ImageFilter.SMOOTH)
	for i in range(500):
	    for j in range(500):
	    	if pic_new.getpixel((i,j))!=255:
	        	pic_new.putpixel((i,j), 0) #  label[i][j]为 0 1 2 3 ，每个像素被分成了这4类
	return pic_new	

def savedata2file(allimages,number_of_images,image_enchancement=True):
	for i in range(number_of_images):
		label= allimages[i][0]
		oneimage=allimages[i][1]
		oneimage=np.array(oneimage).reshape([-1,2])
		pic_new = Image.new("L", (500, 500),255) 
		for pix in oneimage:
			pic_new.putpixel((pix[0],pix[1]),0)
		if image_enchancement:
			pic_new=digit_enhancement(pic_new)
		# pic_new.show()      	
		pic_new.save(str(i)+"_"+label+".png", "PNG")
def poit2vector(image_pointList,image_enchancement=False):
	"""
	image_pointList: stands for all points of an image
	return : an image vector ,format to a list, len=500x500
	"""
	# init_value=[255 for i in range(500*500)]
	# retMat = np.array(init_value)
	im = Image.new("L", (500, 500),255) 	
	oneimage=np.array(image_pointList).reshape([-1,2])
	for pix in oneimage:
		# retMat[pix[0]*500+pix[1]]=0
		im.putpixel((pix[0],pix[1]),0)
	if image_enchancement:
		im=digit_enhancement(im)
	image_vector= list(im.getdata())

	# print len(image_vector)
	return image_vector

def loadDataset(allimages):
	numFiles=len(allimages)
	dataSet = np.zeros([numFiles,500*500],int)
	hwLabels = np.zeros([numFiles,10]) 
	for i in range(numFiles):
		# print i
		label=int(allimages[i][0])
		hwLabels[i][label] = 1.0
		dataSet[i]=poit2vector(allimages[i][1])
	return dataSet, hwLabels


if __name__=="__main__":
	rawfilename="pendigits-orig.tra"
	print ("===parasing image from UNIPEN format")
	allimages,number_of_images=paraseUNIPen(rawfilename)
	print ("number of images:", number_of_images)
	print ("===get training dataset and labels")
	# savedata2file(allimages,number_of_images)
	train_dataSet,train_labels=loadDataset(allimages[:700])
	
	from sklearn.neural_network import MLPClassifier 
	clf = MLPClassifier(hidden_layer_sizes=(100,),
                    activation='logistic', solver='adam',
                    learning_rate_init = 0.0001, max_iter=2000)
	# print(clf)
	print ("===start training the MLP model")
	start=time.clock()
	clf.fit(train_dataSet,train_labels)
	end=time.clock()
	print "time consumer:", end-start

	test_dataSet,test_labels=loadDataset(allimages[700:800])
	print ("===use the trained model to predict the result")
	res = clf.predict(test_dataSet)
	print ("===check out the accuracy")
	error_num = 0
	num = len(test_dataSet)
	for i in range(num):
		if np.sum(res[i] == test_labels[i]) < 10: 
			error_num += 1                     
	print("Total num:",num," Wrong num:", error_num,"  WrongRate:",error_num / float(num))