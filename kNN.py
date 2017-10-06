from numpy import *
import operator

def createDataSet():
	group=array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
	labels=['A','A','B','B']
	return group,labels

#2-1 k-邻近算法
def classify0(inX,dataSet,labels,k):
	#构造与dataSet格式相同的矩阵并计算距离
	dataSetSize=dataSet.shape[0]
	diffMat=tile(inX,(dataSetSize,1))-dataSet
	sqDiffMat=diffMat**2
	sqDistances=sqDiffMat.sum(axis=1)
	distances=sqDistances**0.5
	#根据距离排序，argsort（）返回排序后的下标 默认升序
	sortedDistIndicies=distances.argsort()
	classCount={}
	#统计前k个距离小的点的标签并计数
	for i in range(k):
		voteIlabel=labels[sortedDistIndicies[i]]
		classCount[voteIlabel]=classCount.get(voteIlabel,0)+1
	sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
	#返回数量最多的标签
	return sortedClassCount[0][0]



#2-2 将文本记录转换为Numpy的解析程序	
def file2matrix(filename):
	fr=open(filename)
	arraylines=fr.readlines()
	numberOflines=len(arraylines)
	returnMat=zeros((numberOflines,3))
	classLabelVector=[]
	index=0
	#分别将每行的数据读到returnMat中，标签读到classLableVector中
	for line in arraylines:
		line=line.strip()
		listFormLine=line.split('\t')#以空格为标志分割数据
		returnMat[index ,:]=listFormLine[0:3]
		classLabelVector.append(int(listFormLine[-1]))
		index+=1
	return returnMat,classLabelVector

#2-3 归一化特征值
def autoNorm(dataSet):
	minVals=dataSet.min(0)
	maxVals=dataSet.max(0)
	ranges=maxVals-minVals
	normDataSet=zeros(shape(dataSet))
	m=dataSet.shape[0]
	normDataSet=dataSet-tile(minVals,(m,1))
	normDataSet=normDataSet/tile(ranges,(m,1))
	return normDataSet,ranges,minVals
	
#2-4 分类器针对约会网站的测试代码
def datingClassTest():
	hoRatio=0.50
	datingDataMat,datingLabels=file2matrix('datingTestSet2.txt')
	normMat,ranges,minVals=autoNorm(datingDataMat)
	m=normMat.shape[0]
	numTestVecs=int(m*hoRatio)
	errorCount=0.0
	for i in range(numTestVecs):
		classifierResult=classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
		print("the classifier came back with:%d,the real answer is:%d"%(classifierResult,datingLabels[i]))
		if(classifierResult!=datingLabels[i]):errorCount+=1.0
	print("the total error rate is :%f" %(errorCount/float(numTestVecs)))
	print(errorCount)
	
#2-5 约会网站预测函数
def classifyPerson():
	resultList=['不喜欢','有点喜欢','很喜欢']
	percentTats=float(input("每天玩游戏所占百分比？"))
	ffMiles=float(input("每年飞行的里程？"))
	iceCream=float(input("一年吃的冰淇淋的数量（升）？"))
	datingDataMat,datingLabels=file2matrix('datingTestSet2.txt')
	normMat,ranges,minVals=autoNorm(datingDataMat)
	inArr=array([ffMiles,percentTats,iceCream])
	inArr=array((inArr - minVals) / ranges)
	classifierResult=classify0(inArr,normMat,datingLabels,3)
	print("you will probably like this person:",resultList[classifierResult-1])

#使用sklearn实现KNN的例子
def sklearnKNN():
	resultList=['不喜欢','有点喜欢','很喜欢']
	percentTats=float(input("每天玩游戏所占百分比？"))
	ffMiles=float(input("每年飞行的里程？"))
	iceCream=float(input("一年吃的冰淇淋的数量（升）？"))
	datingDataMat,datingLabels=file2matrix('datingTestSet2.txt')
	normMat,ranges,minVals=autoNorm(datingDataMat)
	inArr=[ffMiles,percentTats,iceCream]
	inArr = array([(inArr - minVals) / ranges])
	X_test=[[37777,5.99911,1.58877]]
	from sklearn.neighbors import KNeighborsClassifier
	neigh=KNeighborsClassifier(n_neighbors=3)
	neigh.fit(normMat,datingLabels)
	Y_pred=neigh.predict(inArr)
	print(Y_pred)
	print("you will probably like this person:", resultList[Y_pred[0] - 1])


if __name__=='__main__':

	#group,labels=createDataSet()
	#print(classify0([0,0],group,labels,3))
	
	# db,dl=file2matrix('datingTestSet2.txt')
	# print(db)
	# normMat,ranges,minVals=autoNorm(db)
	# print(normMat)
	# datingClassTest()
	
	# classifyPerson()

	sklearnKNN()



	
