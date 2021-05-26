print("-----------------= Students' Info =--------------------")
print("Names:")
print("- Omar Alqurashi    , ID: 1742589")
print("- Mohammed Alzahrani, ID: 1740166")
print("- Mohammed Alharbi  , ID: 1740373")
print(Sys.time())
print(Sys.Date())
print("========================Start==========================")
print("---------------------Preparation-----------------------")
#install.packages("xgboost") #eXtreme Gradient Boosting package
library(xgboost)
#install.packages("e1071") # SVM
library(e1071)
#install.packages("JOUSBoost") # AdaBoost
library(JOUSBoost)
#install.packages("randomForest") # Random Forests
library(randomForest)
#install.packages("DMwR") # "Data Mining with R" for Oversampling
library(DMwR)
#install.packages("caret") # for Cross-Validation purpose
library(caret)
#install scmamp library:
#if (!require("devtools")) {
#  install.packages("devtools")
#}
#devtools::install_github("b0rxa/scmamp")
library(scmamp)

rm(list=ls())

g_dataset<-read.table("glass0.txt",sep=",")  #Imbalanced Dataset: Glass
s_dataset<-read.table("spambase.txt",sep=",")#Standard Dataset: Spambase
t_dataset<-read.table("twonorm.txt",sep=",") #Standard Dataset: Twonorm
v_dataset<-read.table("vehicle0.txt",sep=",")#Imbalanced Dataset: Vehicle

g_dataset$V10<-as.factor(g_dataset$V10) #categorical labels
v_dataset$V19<-as.factor(v_dataset$V19) #categorical labels

print("Glass Dimensions:")
print(dim(g_dataset))

print("Spambase Dimensions:")
print(dim(s_dataset))

print("Twonorm Dimensions:")
print(dim(t_dataset))

print("Vehicle Dimensions:")
print(dim(v_dataset))

#Perform XGBoost for Imbalanced Datasets (Glass,Vehicle)
performXGBoost_G_V <- function(train){
	yy<-as.character(train[,ncol(train)])
	yy[yy=="positive"]<-+1
	yy[yy=="negative"]<- 0
	y<-as.numeric(yy)
	x<-as.matrix(train[,-ncol(train)])
	class(x)<-"numeric"
	return(xgboost(data = x, label = y,nrounds = 5,objective = "binary:hinge"))
}

#Perform XGBoost for Standard Datasets (Spambase,Twonorm)
performXGBoost_S_T <- function(train){
	yy<-as.character(train[,ncol(train)])
	yy[yy=="1"]<-+1 # I mean, no need for that :)
	yy[yy=="0"]<- 0 # Neither this one
	y<-as.numeric(yy)
	x<-as.matrix(train[,-ncol(train)])
	class(x)<-"numeric"
	return(xgboost(data = x, label = y,nrounds = 5,objective = "binary:hinge"))
}

#Perform SVM for Imbalanced Datasets (Glass,Vehicle)
performSVM_G_V <- function(train){
	yy<-as.character(train[,ncol(train)])
	yy[yy=="positive"]<-+1
	yy[yy=="negative"]<--1
	y<-as.factor(yy)
	x<-train[,-ncol(train)]
	return(svm(x,y,scale=FALSE)) # Default Kernel: Radial basis
}

#Perform SVM for Standard Datasets (Spambase,Twonorm)
performSVM_S_T <- function(train){
	yy<-as.character(train[,ncol(train)])
	yy[yy=="1"]<-+1
	yy[yy=="0"]<--1
	y<-as.factor(yy)
	x<-train[,-ncol(train)]
	return(svm(x,y,scale=FALSE)) # Default Kernel: Radial basis
}

#Perform Random Forests for all Datasets
performRandomForests <- function(train){
	x<-train[,-ncol(train)]
	y<-as.factor(train[,ncol(train)])
	return(randomForest(x,y))
}

#Perform AdaBoost for Imbalanced Datasets (Glass,Vehicle)
performAdaBoost_G_V <- function(train){
	yy<-as.character(train[,ncol(train)])
	yy[yy=="positive"]<-+1
	yy[yy=="negative"]<--1
	y<-as.numeric(yy)
	x<-as.matrix(train[,-ncol(train)])
	class(x)<-"numeric"
	return(adaboost(x,y))
}

#Perform AdaBoost for Standard Datasets (Spambase,Twonorm)
performAdaBoost_S_T <- function(train){
	yy<-as.character(train[,ncol(train)])
	yy[yy=="1"]<-+1
	yy[yy=="0"]<--1
	y<-as.numeric(yy)
	x<-as.matrix(train[,-ncol(train)])
	class(x)<-"numeric"
	return(adaboost(x,y))
}

#Performance Measure Calculation Function:
calPerfMeasures <- function(fit,x_test,y_test){
	p<-predict(fit,x_test)
	perf<-table(y_test,p)
	tn<-perf[1,1]
	tp<-perf[2,2]
	fp<-perf[1,2]
	fn<-perf[2,1]
	acc<-(tp+tn)/(tp+tn+fp+fn)
	bac<- 0.5*((tp/(tp+fn))+(tn/(tn+fp)))
	f1<-(2*tp)/(2*tp+fp+fn)
	tpr = tp/(tp + fn)
	fpr = fp/(fp + tn)
	return(c(acc,bac,f1,tpr,fpr))
}

#Plot performance of several classifiers using ROC curves:
plotROC <- function(c1_tpr, c1_fpr, c2_tpr, c2_fpr, c3_tpr, c3_fpr, c4_tpr, c4_fpr,data_name){
	plot(c(0,c1_fpr,1),c(0,c1_tpr,1) , col="green" ,type="l",xlab="MFPR",ylab="MTPR",main=data_name)
	lines(c(0,c2_fpr,1),c(0,c2_tpr,1), col="blue")
	lines(c(0,c3_fpr,1),c(0,c3_tpr,1), col="purple")
	lines(c(0,c4_fpr,1),c(0,c4_tpr,1), col="black")
	abline(a=0, b=1,col="red")
	legend(0.5,0.4,c("XGBoost","SVM","Random Forests","AdaBoost"),col=c("green","blue","purple","black"),lwd=3,bg="transparent")
}

#Performance Measure print Function:
printPerfMeasures <- function(g_acc,s_acc,t_acc,v_acc,g_bac,s_bac,t_bac,v_bac,g_f1,s_f1,t_f1,v_f1){
	print("glass0:")
	print(sprintf("MACC = %f, SD = %f", mean(g_acc),sd(g_acc)))
	print(sprintf("MBAC = %f, SD = %f", mean(g_bac),sd(g_bac)))
	print(sprintf("MF1  = %f, SD = %f", mean(g_f1),sd(g_f1)))
	print("spambase:")
	print(sprintf("MACC = %f, SD = %f", mean(s_acc),sd(s_acc)))
	print(sprintf("MBAC = %f, SD = %f", mean(s_bac),sd(s_bac)))
	print(sprintf("MF1  = %f, SD = %f", mean(s_f1),sd(s_f1)))
	print("twonorm:")
	print(sprintf("MACC = %f, SD = %f", mean(t_acc),sd(t_acc)))
	print(sprintf("MBAC = %f, SD = %f", mean(t_bac),sd(t_bac)))
	print(sprintf("MF1  = %f, SD = %f", mean(t_f1),sd(t_f1)))
	print("vehicle0:")
	print(sprintf("MACC = %f, SD = %f", mean(v_acc),sd(v_acc)))
	print(sprintf("MBAC = %f, SD = %f", mean(v_bac),sd(v_bac)))
	print(sprintf("MF1  = %f, SD = %f", mean(v_f1),sd(v_f1)))
}

print("================Oversampling===============")
print("-------------------before------------------")
print("Num. Records in Glass Dataset Labeled as 'positive':")
print(sum(g_dataset=="positive"))
print("Num. Records in Glass Dataset Labeled as 'negative':")
print(sum(g_dataset=="negative"))
print("Num. Records in Vehicle Dataset Labeled as 'positive':")
print(sum(v_dataset=="positive"))
print("Num. Records in Vehicle Dataset Labeled as 'negative':")
print(sum(v_dataset=="negative"))

newData <- SMOTE(V10 ~ ., g_dataset, perc.over = 100,perc.under=0)
g_dataset<-rbind(g_dataset,newData)
g_dataset<-na.omit(g_dataset)

newData <- SMOTE(V19 ~ ., v_dataset, perc.over = 100,perc.under=0)
v_dataset<-rbind(v_dataset,newData)
v_dataset<-na.omit(v_dataset)

print("-------------------after-------------------")
print("Num. Records in Glass Dataset Labeled as 'positive':")
print(sum(g_dataset=="positive"))
print("Num. Records in Glass Dataset Labeled as 'negative':")
print(sum(g_dataset=="negative"))
print("Num. Records in Vehicle Dataset Labeled as 'positive':")
print(sum(v_dataset=="positive"))
print("Num. Records in Vehicle Dataset Labeled as 'negative':")
print(sum(v_dataset=="negative"))
print("=============Oversampling End============")
print("") #space

k<-5
g_folds<-createFolds(1:nrow(g_dataset),k,list=F)
s_folds<-createFolds(1:nrow(s_dataset),k,list=F)
t_folds<-createFolds(1:nrow(t_dataset),k,list=F)
v_folds<-createFolds(1:nrow(v_dataset),k,list=F)

# Performance Measures for Glass Dataset
xg_g_acc_test<-rep(NA,k)
xg_g_bac_test<-rep(NA,k)
xg_g_f1_test<-rep(NA,k)
xg_g_tpr_test<-rep(NA,k)
xg_g_fpr_test<-rep(NA,k)

svm_g_acc_test<-rep(NA,k)
svm_g_bac_test<-rep(NA,k)
svm_g_f1_test<-rep(NA,k)
svm_g_tpr_test<-rep(NA,k)
svm_g_fpr_test<-rep(NA,k)

#(rfs -> Random Forests)
rfs_g_acc_test<-rep(NA,k)
rfs_g_bac_test<-rep(NA,k)
rfs_g_f1_test<-rep(NA,k)
rfs_g_tpr_test<-rep(NA,k)
rfs_g_fpr_test<-rep(NA,k)

ada_g_acc_test<-rep(NA,k)
ada_g_bac_test<-rep(NA,k)
ada_g_f1_test<-rep(NA,k)
ada_g_tpr_test<-rep(NA,k)
ada_g_fpr_test<-rep(NA,k)

# Performance Measures for Spambase Dataset
xg_s_acc_test<-rep(NA,k)
xg_s_bac_test<-rep(NA,k)
xg_s_f1_test<-rep(NA,k)
xg_s_tpr_test<-rep(NA,k)
xg_s_fpr_test<-rep(NA,k)

svm_s_acc_test<-rep(NA,k)
svm_s_bac_test<-rep(NA,k)
svm_s_f1_test<-rep(NA,k)
svm_s_tpr_test<-rep(NA,k)
svm_s_fpr_test<-rep(NA,k)

#(rfs -> Random Forests)
rfs_s_acc_test<-rep(NA,k)
rfs_s_bac_test<-rep(NA,k)
rfs_s_f1_test<-rep(NA,k)
rfs_s_tpr_test<-rep(NA,k)
rfs_s_fpr_test<-rep(NA,k)

ada_s_acc_test<-rep(NA,k)
ada_s_bac_test<-rep(NA,k)
ada_s_f1_test<-rep(NA,k)
ada_s_tpr_test<-rep(NA,k)
ada_s_fpr_test<-rep(NA,k)

# Performance Measures for Twonorm Dataset
xg_t_acc_test<-rep(NA,k)
xg_t_bac_test<-rep(NA,k)
xg_t_f1_test<-rep(NA,k)
xg_t_tpr_test<-rep(NA,k)
xg_t_fpr_test<-rep(NA,k)

svm_t_acc_test<-rep(NA,k)
svm_t_bac_test<-rep(NA,k)
svm_t_f1_test<-rep(NA,k)
svm_t_tpr_test<-rep(NA,k)
svm_t_fpr_test<-rep(NA,k)

#(rfs -> Random Forests)
rfs_t_acc_test<-rep(NA,k)
rfs_t_bac_test<-rep(NA,k)
rfs_t_f1_test<-rep(NA,k)
rfs_t_tpr_test<-rep(NA,k)
rfs_t_fpr_test<-rep(NA,k)

ada_t_acc_test<-rep(NA,k)
ada_t_bac_test<-rep(NA,k)
ada_t_f1_test<-rep(NA,k)
ada_t_tpr_test<-rep(NA,k)
ada_t_fpr_test<-rep(NA,k)

# Performance Measures for Vehicle Dataset
xg_v_acc_test<-rep(NA,k)
xg_v_bac_test<-rep(NA,k)
xg_v_f1_test<-rep(NA,k)
xg_v_tpr_test<-rep(NA,k)
xg_v_fpr_test<-rep(NA,k)

svm_v_acc_test<-rep(NA,k)
svm_v_bac_test<-rep(NA,k)
svm_v_f1_test<-rep(NA,k)
svm_v_tpr_test<-rep(NA,k)
svm_v_fpr_test<-rep(NA,k)

#(rfs -> Random Forests)
rfs_v_acc_test<-rep(NA,k)
rfs_v_bac_test<-rep(NA,k)
rfs_v_f1_test<-rep(NA,k)
rfs_v_tpr_test<-rep(NA,k)
rfs_v_fpr_test<-rep(NA,k)

ada_v_acc_test<-rep(NA,k)
ada_v_bac_test<-rep(NA,k)
ada_v_f1_test<-rep(NA,k)
ada_v_tpr_test<-rep(NA,k)
ada_v_fpr_test<-rep(NA,k)

print("==== Start 5-Folds Cross-Validation ====")
for(i in 1:k){
	print("") #space
	print(sprintf("_______________ i = %.0f ________________",i))
	g_ind<-which(g_folds==i)
	s_ind<-which(s_folds==i)
	t_ind<-which(t_folds==i)
	v_ind<-which(v_folds==i)
	
	g_train<-g_dataset[-g_ind,]
	s_train<-s_dataset[-s_ind,]
	t_train<-t_dataset[-t_ind,]
	v_train<-v_dataset[-v_ind,]
	
	g_test<-g_dataset[g_ind,]
	s_test<-s_dataset[s_ind,]
	t_test<-t_dataset[t_ind,]
	v_test<-v_dataset[v_ind,]

	print("============= Training... ============")
	g_xg_fit <- performXGBoost_G_V(g_train)
	s_xg_fit <- performXGBoost_S_T(s_train)
	t_xg_fit <- performXGBoost_S_T(t_train)
	v_xg_fit <- performXGBoost_G_V(v_train)
	print("---------- XGBoost Trained -----------")
	
	g_svm_fit <- performSVM_G_V(g_train)
	s_svm_fit <- performSVM_S_T(s_train)
	t_svm_fit <- performSVM_S_T(t_train)
	v_svm_fit <- performSVM_G_V(v_train)
	print("------------ SVM Trained -------------")
	
	g_rfs_fit <- performRandomForests(g_train)
	s_rfs_fit <- performRandomForests(s_train)
	t_rfs_fit <- performRandomForests(t_train)
	v_rfs_fit <- performRandomForests(v_train)
	print("------ Random Forests Trained --------")
	
	g_ada_fit <- performAdaBoost_G_V(g_train)
	s_ada_fit <- performAdaBoost_S_T(s_train)
	t_ada_fit <- performAdaBoost_S_T(t_train)
	v_ada_fit <- performAdaBoost_G_V(v_train)
	print("--------- AdaBoost Trained -----------")

	print("============= Testing... =============")
	#--------------- XGBoost testing -------------
	#g_test with XGBoost:
	yy_test<-as.character(g_test[,ncol(g_test)])
	yy_test[yy_test=="positive"]<- +1
	yy_test[yy_test=="negative"]<- 0
	y<-as.numeric(yy_test)
	x<-as.matrix(g_test[,-ncol(g_test)])
	class(x)<-"numeric"
	# calculate performance measures
	perf_measures<-calPerfMeasures(g_xg_fit,x,y)
	xg_g_acc_test[i]<-perf_measures[1] #acc
	xg_g_bac_test[i]<-perf_measures[2] #bac
	xg_g_f1_test[i]<-perf_measures[3] #f1
	xg_g_tpr_test[i]<-perf_measures[4] #tpr
	xg_g_fpr_test[i]<-perf_measures[5] #fpr

	#s_test with XGBoost:
	yy_test<-as.character(s_test[,ncol(s_test)])
	yy_test[yy_test=="1"]<- +1
	yy_test[yy_test=="0"]<- 0
	y<-as.numeric(yy_test)
	x<-as.matrix(s_test[,-ncol(s_test)])
	class(x)<-"numeric"
	# calculate performance measures
	perf_measures<-calPerfMeasures(s_xg_fit,x,y)
	xg_s_acc_test[i]<-perf_measures[1] #acc
	xg_s_bac_test[i]<-perf_measures[2] #bac
	xg_s_f1_test[i]<-perf_measures[3] #f1
	xg_s_tpr_test[i]<-perf_measures[4] #tpr
	xg_s_fpr_test[i]<-perf_measures[5] #fpr

	#t_test with XGBoost:
	yy_test<-as.character(t_test[,ncol(t_test)])
	yy_test[yy_test=="1"]<- +1
	yy_test[yy_test=="0"]<- 0
	y<-as.numeric(yy_test)
	x<-as.matrix(t_test[,-ncol(t_test)])
	class(x)<-"numeric"
	# calculate performance measures
	perf_measures<-calPerfMeasures(t_xg_fit,x,y)
	xg_t_acc_test[i]<-perf_measures[1] #acc
	xg_t_bac_test[i]<-perf_measures[2] #bac
	xg_t_f1_test[i]<-perf_measures[3] #f1
	xg_t_tpr_test[i]<-perf_measures[4] #tpr
	xg_t_fpr_test[i]<-perf_measures[5] #fpr

	#v_test with XGBoost:
	yy_test<-as.character(v_test[,ncol(v_test)])
	yy_test[yy_test=="positive"]<- +1
	yy_test[yy_test=="negative"]<- 0
	y<-as.numeric(yy_test)
	x<-as.matrix(v_test[,-ncol(v_test)])
	class(x)<-"numeric"
	# calculate performance measures
	perf_measures<-calPerfMeasures(v_xg_fit,x,y)
	xg_v_acc_test[i]<-perf_measures[1] #acc
	xg_v_bac_test[i]<-perf_measures[2] #bac
	xg_v_f1_test[i]<-perf_measures[3] #f1
	xg_v_tpr_test[i]<-perf_measures[4] #tpr
	xg_v_fpr_test[i]<-perf_measures[5] #fpr
	print("---------- XGBoost Tested ------------")

	#---------------- SVM testing -------------")
	#g_test with SVM:
	yy_test<-as.character(g_test[,ncol(g_test)])
	yy_test[yy_test=="positive"]<- +1
	yy_test[yy_test=="negative"]<- -1
	y<-as.factor(yy_test)
	x<-g_test[,-ncol(g_test)]
	# calculate performance measures
	perf_measures<-calPerfMeasures(g_svm_fit,x,y)
	svm_g_acc_test[i]<-perf_measures[1] #acc
	svm_g_bac_test[i]<-perf_measures[2] #bac
	svm_g_f1_test[i]<-perf_measures[3] #f1
	svm_g_tpr_test[i]<-perf_measures[4] #tpr
	svm_g_fpr_test[i]<-perf_measures[5] #fpr

	#s_test with SVM:
	yy_test<-as.character(s_test[,ncol(s_test)])
	yy_test[yy_test=="1"]<- +1
	yy_test[yy_test=="0"]<- -1
	y<-as.factor(yy_test)
	x<-s_test[,-ncol(s_test)]
	# calculate performance measures
	perf_measures<-calPerfMeasures(s_svm_fit,x,y)
	svm_s_acc_test[i]<-perf_measures[1] #acc
	svm_s_bac_test[i]<-perf_measures[2] #bac
	svm_s_f1_test[i]<-perf_measures[3] #f1
	svm_s_tpr_test[i]<-perf_measures[4] #tpr
	svm_s_fpr_test[i]<-perf_measures[5] #fpr

	#t_test with SVM:
	yy_test<-as.character(t_test[,ncol(t_test)])
	yy_test[yy_test=="1"]<- +1
	yy_test[yy_test=="0"]<- -1
	y<-as.factor(yy_test)
	x<-t_test[,-ncol(t_test)]
	# calculate performance measures
	perf_measures<-calPerfMeasures(t_svm_fit,x,y)
	svm_t_acc_test[i]<-perf_measures[1] #acc
	svm_t_bac_test[i]<-perf_measures[2] #bac
	svm_t_f1_test[i]<-perf_measures[3] #f1
	svm_t_tpr_test[i]<-perf_measures[4] #tpr
	svm_t_fpr_test[i]<-perf_measures[5] #fpr

	#v_test with SVM:
	yy_test<-as.character(v_test[,ncol(v_test)])
	yy_test[yy_test=="positive"]<- +1
	yy_test[yy_test=="negative"]<- -1
	y<-as.factor(yy_test)
	x<-v_test[,-ncol(v_test)]
	# calculate performance measures
	perf_measures<-calPerfMeasures(v_svm_fit,x,y)
	svm_v_acc_test[i]<-perf_measures[1] #acc
	svm_v_bac_test[i]<-perf_measures[2] #bac
	svm_v_f1_test[i]<-perf_measures[3] #f1
	svm_v_tpr_test[i]<-perf_measures[4] #tpr
	svm_v_fpr_test[i]<-perf_measures[5] #fpr
	print("------------- SVM Tested -------------")

	#------------ Random Forests testing --------
	#g_test with RandomForests:
	x<-g_test[,-ncol(g_test)]
	y<-as.factor(g_test[,ncol(g_test)])
	# calculate performance measures
	perf_measures<-calPerfMeasures(g_rfs_fit,x,y)
	rfs_g_acc_test[i]<-perf_measures[1] #acc
	rfs_g_bac_test[i]<-perf_measures[2] #bac
	rfs_g_f1_test[i]<-perf_measures[3] #f1
	rfs_g_tpr_test[i]<-perf_measures[4] #tpr
	rfs_g_fpr_test[i]<-perf_measures[5] #fpr

	#s_test with RandomForests:
	x<-s_test[,-ncol(s_test)]
	y<-as.factor(s_test[,ncol(s_test)])
	# calculate performance measures
	perf_measures<-calPerfMeasures(s_rfs_fit,x,y)
	rfs_s_acc_test[i]<-perf_measures[1] #acc
	rfs_s_bac_test[i]<-perf_measures[2] #bac
	rfs_s_f1_test[i]<-perf_measures[3] #f1
	rfs_s_tpr_test[i]<-perf_measures[4] #tpr
	rfs_s_fpr_test[i]<-perf_measures[5] #fpr

	#t_test with RandomForests:
	x<-t_test[,-ncol(t_test)]
	y<-as.factor(t_test[,ncol(t_test)])
	# calculate performance measures
	perf_measures<-calPerfMeasures(t_rfs_fit,x,y)
	rfs_t_acc_test[i]<-perf_measures[1] #acc
	rfs_t_bac_test[i]<-perf_measures[2] #bac
	rfs_t_f1_test[i]<-perf_measures[3] #f1
	rfs_t_tpr_test[i]<-perf_measures[4] #tpr
	rfs_t_fpr_test[i]<-perf_measures[5] #fpr

	#v_test with RandomForests:
	x<-v_test[,-ncol(v_test)]
	y<-as.factor(v_test[,ncol(v_test)])
	# calculate performance measures
	perf_measures<-calPerfMeasures(v_rfs_fit,x,y)
	rfs_v_acc_test[i]<-perf_measures[1] #acc
	rfs_v_bac_test[i]<-perf_measures[2] #bac
	rfs_v_f1_test[i]<-perf_measures[3] #f1
	rfs_v_tpr_test[i]<-perf_measures[4] #tpr
	rfs_v_fpr_test[i]<-perf_measures[5] #fpr
	print("----- Random Forests Tested ----------")

	#-------------- AdaBoost testing ------------
	#g_test with AdaBoost:
	yy_test<-as.character(g_test[,ncol(g_test)])
	yy_test[yy_test=="positive"]<- +1
	yy_test[yy_test=="negative"]<- -1
	y<-as.numeric(yy_test)
	x<-as.matrix(g_test[,-ncol(g_test)])
	class(x)<-"numeric"
	# calculate performance measures
	perf_measures<-calPerfMeasures(g_ada_fit,x,y)
	ada_g_acc_test[i]<-perf_measures[1] #acc
	ada_g_bac_test[i]<-perf_measures[2] #bac
	ada_g_f1_test[i]<-perf_measures[3] #f1
	ada_g_tpr_test[i]<-perf_measures[4] #tpr
	ada_g_fpr_test[i]<-perf_measures[5] #fpr

	#s_test with AdaBoost:
	yy_test<-as.character(s_test[,ncol(s_test)])
	yy_test[yy_test=="1"]<- +1
	yy_test[yy_test=="0"]<- -1
	y<-as.numeric(yy_test)
	x<-as.matrix(s_test[,-ncol(s_test)])
	class(x)<-"numeric"
	# calculate performance measures
	perf_measures<-calPerfMeasures(s_ada_fit,x,y)
	ada_s_acc_test[i]<-perf_measures[1] #acc
	ada_s_bac_test[i]<-perf_measures[2] #bac
	ada_s_f1_test[i]<-perf_measures[3] #f1
	ada_s_tpr_test[i]<-perf_measures[4] #tpr
	ada_s_fpr_test[i]<-perf_measures[5] #fpr

	#t_test with AdaBoost:
	yy_test<-as.character(t_test[,ncol(t_test)])
	yy_test[yy_test=="1"]<- +1
	yy_test[yy_test=="0"]<- -1
	y<-as.numeric(yy_test)
	x<-as.matrix(t_test[,-ncol(t_test)])
	class(x)<-"numeric"
	# calculate performance measures
	perf_measures<-calPerfMeasures(t_ada_fit,x,y)
	ada_t_acc_test[i]<-perf_measures[1] #acc
	ada_t_bac_test[i]<-perf_measures[2] #bac
	ada_t_f1_test[i]<-perf_measures[3] #f1
	ada_t_tpr_test[i]<-perf_measures[4] #tpr
	ada_t_fpr_test[i]<-perf_measures[5] #fpr

	#v_test with AdaBoost:
	yy_test<-as.character(v_test[,ncol(v_test)])
	yy_test[yy_test=="positive"]<- +1
	yy_test[yy_test=="negative"]<- -1
	y<-as.numeric(yy_test)
	x<-as.matrix(v_test[,-ncol(v_test)])
	class(x)<-"numeric"
	# calculate performance measures
	perf_measures<-calPerfMeasures(v_ada_fit,x,y)
	ada_v_acc_test[i]<-perf_measures[1] #acc
	ada_v_bac_test[i]<-perf_measures[2] #bac
	ada_v_f1_test[i]<-perf_measures[3] #f1
	ada_v_tpr_test[i]<-perf_measures[4] #tpr
	ada_v_fpr_test[i]<-perf_measures[5] #fpr
	print("--------- AdaBoost Tested ------------")
}
print("") #space
print("---------Calculations of Performance Measures----------")
print("------------------------XGBoost------------------------")
printPerfMeasures(xg_g_acc_test,xg_s_acc_test,xg_t_acc_test,xg_v_acc_test,
	xg_g_bac_test,xg_s_bac_test,xg_t_bac_test,xg_v_bac_test,
	xg_g_f1_test,xg_s_f1_test,xg_t_f1_test,xg_v_f1_test)
print("--------------------------SVM--------------------------")
printPerfMeasures(svm_g_acc_test,svm_s_acc_test,svm_t_acc_test,svm_v_acc_test,
	svm_g_bac_test,svm_s_bac_test,svm_t_bac_test,svm_v_bac_test,
	svm_g_f1_test,svm_s_f1_test,svm_t_f1_test,svm_v_f1_test)
print("-------------------Random Forests----------------------")
printPerfMeasures(rfs_g_acc_test,rfs_s_acc_test,rfs_t_acc_test,rfs_v_acc_test,
	rfs_g_bac_test,rfs_s_bac_test,rfs_t_bac_test,rfs_v_bac_test,
	rfs_g_f1_test,rfs_s_f1_test,rfs_t_f1_test,rfs_v_f1_test)
print("-----------------------AdaBoost------------------------")
printPerfMeasures(ada_g_acc_test,ada_s_acc_test,ada_t_acc_test,ada_v_acc_test,
	ada_g_bac_test,ada_s_bac_test,ada_t_bac_test,ada_v_bac_test,
	ada_g_f1_test,ada_s_f1_test,ada_t_f1_test,ada_v_f1_test)
print("============= Cross-Validation Finished ===============")
print("") #space

#======================Plotting...========================
par(mfrow =c(2,2))
plotROC(mean(xg_g_tpr_test), mean(xg_g_fpr_test), mean(svm_g_tpr_test), mean(svm_g_fpr_test), mean(rfs_g_tpr_test), mean(rfs_g_fpr_test), mean(ada_g_tpr_test), mean(ada_g_fpr_test),"Glass Identification")
plotROC(mean(xg_s_tpr_test), mean(xg_s_fpr_test), mean(svm_s_tpr_test), mean(svm_s_fpr_test), mean(rfs_s_tpr_test), mean(rfs_s_fpr_test), mean(ada_s_tpr_test), mean(ada_s_fpr_test),"Spambase Data Set")
plotROC(mean(xg_t_tpr_test), mean(xg_t_fpr_test), mean(svm_t_tpr_test), mean(svm_t_fpr_test), mean(rfs_t_tpr_test), mean(rfs_t_fpr_test), mean(ada_t_tpr_test), mean(ada_t_fpr_test),"Twonorm Data Set")
plotROC(mean(xg_v_tpr_test), mean(xg_v_fpr_test), mean(svm_v_tpr_test), mean(svm_v_fpr_test), mean(rfs_v_tpr_test), mean(rfs_v_fpr_test), mean(ada_v_tpr_test), mean(ada_v_fpr_test),"Vehicle Silhouttes")

print("================= Statistical Tests ===================")
print("------------------------MACC---------------------------")
perf_macc<-data.frame(
	XGBoost = c(mean(xg_g_acc_test),mean(xg_s_acc_test),mean(xg_t_acc_test),mean(xg_v_acc_test)),
	SVM = c(mean(svm_g_acc_test),mean(svm_s_acc_test),mean(svm_t_acc_test),mean(svm_v_acc_test)),
	RandomForests = c(mean(rfs_g_acc_test),mean(rfs_s_acc_test),mean(rfs_t_acc_test),mean(rfs_v_acc_test)),
	AdaBoost = c(mean(ada_g_acc_test),mean(ada_s_acc_test),mean(ada_t_acc_test),mean(ada_v_acc_test)),
	row.names=c("Glass","Spambase","Twonorm","Vehicle"))
print(postHocTest(data = perf_macc, test ="friedman", correct = "bergmann"))

print("------------------------MBAC---------------------------")
perf_mbac<-data.frame(
	XGBoost = c(mean(xg_g_bac_test),mean(xg_s_bac_test),mean(xg_t_bac_test),mean(xg_v_bac_test)),
	SVM = c(mean(svm_g_bac_test),mean(svm_s_bac_test),mean(svm_t_bac_test),mean(svm_v_bac_test)),
	RandomForests = c(mean(rfs_g_bac_test),mean(rfs_s_bac_test),mean(rfs_t_bac_test),mean(rfs_v_bac_test)),
	AdaBoost = c(mean(ada_g_bac_test),mean(ada_s_bac_test),mean(ada_t_bac_test),mean(ada_v_bac_test)),
	row.names=c("Glass","Spambase","Twonorm","Vehicle"))
print(postHocTest(data = perf_mbac, test ="friedman", correct = "bergmann"))

print("-------------------------MF1---------------------------")
perf_mf1<-data.frame(
	XGBoost = c(mean(xg_g_f1_test),mean(xg_s_f1_test),mean(xg_t_f1_test),mean(xg_v_f1_test)),
	SVM = c(mean(svm_g_f1_test),mean(svm_s_f1_test),mean(svm_t_f1_test),mean(svm_v_f1_test)),
	RandomForests = c(mean(rfs_g_f1_test),mean(rfs_s_f1_test),mean(rfs_t_f1_test),mean(rfs_v_f1_test)),
	AdaBoost = c(mean(ada_g_f1_test),mean(ada_s_f1_test),mean(ada_t_f1_test),mean(ada_v_f1_test)),
	row.names=c("Glass","Spambase","Twonorm","Vehicle"))
print(postHocTest(data = perf_mf1, test ="friedman", correct = "bergmann"))

print("=========================END===========================")
print("-----------------= Students' Info =--------------------")
print("Names:")
print("- Omar Alqurashi    , ID: 1742589")
print("- Mohammed Alzahrani, ID: 1740166")
print("- Mohammed Alharbi  , ID: 1740373")
print(Sys.time())
print(Sys.Date())