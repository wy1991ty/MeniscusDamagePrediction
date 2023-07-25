# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 14:25:27 2022

@author: wy
"""
from sklearn.model_selection import train_test_split,cross_val_score
#from sklearn import cross_validation
from sklearn.metrics import recall_score, accuracy_score,precision_score,f1_score
import xlrd
import numpy as np
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest,VarianceThreshold
from sklearn.feature_selection import chi2,f_classif,mutual_info_classif
#from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
#from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt
from sklearn import preprocessing
#import numpy as np
from metric_learn import LMNN
# sklearn.datasets import load_iris
#######数据读取#######
def read_data(tables,label): 
    x_train, x_test, y_train, y_test =train_test_split(tables, label, test_size=0.3, random_state=0)
    return x_train,y_train,x_test,y_test
#######Excel转数组###########
def import_excel(excel):
    for rown in range(excel.nrows):
        for col in range(excel.ncols):
            if col==0:
                label[rown,col]=excel.cell_value(rown,col)
            else:
                tables[rown,(col-1)]=excel.cell_value(rown,col)
    #print(tables)
    return tables,label
###数据预处理#######
def data_preprocessing(values):
    #数据标准化
    data = preprocessing.scale(values,axis=1) 
    return data
 #缺失处理
 #特征优化
def Select(tables,label):
      #移除低方差
      #tables =VarianceThreshold(threshold=(.75 * (1 - .75))).fit_transform(tables)
      #卡方检测,ANOVA F值，信息熵
     selector=SelectKBest(chi2,k=6)
      #selector=SelectKBest(f_classif,k=20)#ANOVA F值
      #selector=SelectKBest(mutual_info_classif,k=10)#信息熵
     selector.fit(tables,label)
     print("selected index:",selector.get_support(True))
      #score=selector.scores_
      #P=selector.pvalues_
      #tables=selector.transform(tables)
      #PCAn_components=10,
      #pca = PCA(svd_solver='full',whiten=True,copy=True)
     # tables=pca.fit_transform(tables)
     return tables,selector.get_support(True)
  #度量学习
def metric_learn(tables,label):
   # iris_data = load_iris()
    lmnn = LMNN(k=2, learn_rate=1e-4)
    lmnn.fit(tables,label)
    X_1 = lmnn.transform(tables)
    return  X_1
   
######逻辑回归#######
def log_pre(x_train,y_train,x_test,y_test):
    clf=LogisticRegression(solver='liblinear',max_iter=40)
    clf.fit(x_train,y_train)
    accuracy_train,accuracy_test,recall_train,recall_test,precision_train,precision_test,f1_train,f1_test,predict_test=classfision(clf,x_train,y_train,x_test,y_test)
    predict_proba=clf.predict_proba(x_test)
    print(predict_proba)
    #predict_proba = clf.decision_function(y_test)
    fpr, tpr, thresholds = roc_curve(y_test, predict_proba[:, 1])
    roc_auc=auc(fpr,tpr)
    print("LOG AUC:",roc_auc)
    plt.figure()
    plt.title('ROC of LR in test data_feature select')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.0])
    plt.ylabel('tpr')
    plt.xlabel('fpr')
    plt.show()
    return accuracy_train,accuracy_test,recall_train,recall_test,precision_train,precision_test,f1_train,f1_test,predict_test,predict_proba
#######svm#########
def svm_pre(x_train,y_train,x_test,y_test):
    clf=svm.SVC(kernel='linear',probability=True)
    clf.fit(x_train,y_train)
    accuracy_train,accuracy_test,recall_train,recall_test,precision_train,precision_test,f1_train,f1_test,predict_test=classfision(clf,x_train,y_train,x_test,y_test)
    predict_proba=clf.predict_proba(x_test)
    print(predict_proba)
    #predict_proba = clf.decision_function(y_test)
    fpr, tpr, thresholds = roc_curve(y_test, predict_proba[:, 1])
    roc_auc=auc(fpr,tpr)
    print("SVM AUC:",roc_auc)
    plt.figure()
    plt.title('')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.0])
    plt.ylabel('tpr')
    plt.xlabel('fpr')
    plt.show()
    return accuracy_train,accuracy_test,recall_train,recall_test,precision_train,precision_test,f1_train,f1_test

####预测结果
def classfision(clf,x_train,y_train,x_test,y_test):
    predict_train=clf.predict(x_train)
    predict_test=clf.predict(x_test)
    print(predict_test)
    accuracy_train=accuracy_score(predict_train,y_train)
    accuracy_test=accuracy_score(predict_test,y_test)
    recall_train=recall_score(predict_train,y_train)
    recall_test=recall_score(predict_test,y_test)
    precision_train=precision_score(predict_train,y_train)
    precision_test=precision_score(predict_test,y_test)
    f1_train=f1_score(predict_train,y_train)
    f1_test=f1_score(predict_test,y_test)
    return accuracy_train,accuracy_test,recall_train,recall_test,precision_train,precision_test,f1_train,f1_test,predict_test

if __name__ == '__main__':
  #将excel表格的内容导入到列表中
  ExcelFile=xlrd.open_workbook(r'E:\博士资料\2023.1-3杂物\度量学习论文\banyueban.xls')
  table = ExcelFile.sheets()[0]
  tables =np.zeros(shape=(157,30))
  label=np.zeros(shape=(157,1))
  tables,label=import_excel(table)
  tables=data_preprocessing(tables)
  #tables,score,P=Select(tables,label)
  #tables,feature=Select(tables,label)
  tables=metric_learn(tables,label)
  x_train,y_train,x_test,y_test=read_data(tables,label)
  svm_accuracy_train,svm_accuracy_test,svm_recall_train,svm_recall_test,svm_precision_train,svm_precision_test,svm_f1_train,svm_f1_test=svm_pre(x_train,y_train,x_test,y_test)
  log_accuracy_train,log_accuracy_test,log_recall_train,log_recall_test,log_precision_train,log_precision_test,log_f1_train,log_f1_test,predict_test, predict_proba=log_pre(x_train,y_train,x_test,y_test)
  print("svm训练集准确度,召回率,精确度,F1:",svm_accuracy_train,svm_recall_train,svm_precision_train,svm_f1_train)
  print("svm测试集准确度和召回率,精确度,F1:",svm_accuracy_test,svm_recall_test,svm_precision_test,svm_f1_test)
  print("log训练集准确度和召回率,精确度,F1:",log_accuracy_train,log_recall_train,log_precision_train,log_f1_train)
  print("log测试集准确度和召回率,精确度,F1:",log_accuracy_test,log_recall_test,log_precision_test,log_f1_test)

  