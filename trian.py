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
#######Read data#######
def read_data(tables,label): 
    x_train, x_test, y_train, y_test =train_test_split(tables, label, test_size=0.3, random_state=0)
    return x_train,y_train,x_test,y_test
#########Excel to array conversion#########
def import_excel(excel):
    for rown in range(excel.nrows):
        for col in range(excel.ncols):
            if col==0:
                label[rown,col]=excel.cell_value(rown,col)
            else:
                tables[rown,(col-1)]=excel.cell_value(rown,col)
    #print(tables)
    return tables,label
###Data preprocessing#######
def data_preprocessing(values):
    #data standardization
    data = preprocessing.scale(values,axis=1) 
    return data
def Select(tables,label):
    
     selector=SelectKBest(chi2,k=6)

     selector.fit(tables,label)
     print("selected index:",selector.get_support(True))

     return tables,selector.get_support(True)
 #LNMM
def metric_learn(tables,label):

    lmnn = LMNN(k=2, learn_rate=1e-4)
    lmnn.fit(tables,label)
    X_1 = lmnn.transform(tables)
    return  X_1
   
########MLP#######
def MLP_pre(x_train,y_train,x_test,y_test):
    clf=MLPClassifier(hidden_layer_sizes=(50,50),activation='identity', solver='sgd',
                      learning_rate='adaptive',learning_rate_init=0.001,alpha=5,
                      max_iter=5000)
    clf.fit(x_train,y_train)
    accuracy_train,accuracy_test,recall_train,recall_test,precision_train,precision_test,f1_train,f1_test,predict_test=classfision(clf,x_train,y_train,x_test,y_test)
    predict_proba=clf.predict_proba(x_test)

    fpr, tpr, thresholds = roc_curve(y_test, predict_proba[:, 1])
    roc_auc=auc(fpr,tpr)
    print("MLP AUC:",roc_auc)
    plt.figure()
    plt.title('ROC of MLP in test data_feature select')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.0])
    plt.ylabel('tpr')
    plt.xlabel('fpr')
    plt.show()
    print(predict_proba)

    return accuracy_train,accuracy_test,recall_train,recall_test,precision_train,precision_test,f1_train,f1_test,predict_test,predict_proba
######RRF######
def RFF_pre(x_train,y_train,x_test,y_test):
    clf=RandomForestClassifier(n_estimators=3,max_features=0.3)
    clf.fit(x_train,y_train)
    accuracy_train,accuracy_test,recall_train,recall_test,precision_train,precision_test,f1_train,f1_test,predict_test=classfision(clf,x_train,y_train,x_test,y_test)
    predict_proba=clf.predict_proba(x_test)
    print(predict_proba)
    #predict_proba = clf.decision_function(y_test)
    fpr, tpr, thresholds = roc_curve(y_test, predict_proba[:, 1])
    roc_auc=auc(fpr,tpr)
    print("RFF AUC:",roc_auc)
    plt.figure()
    plt.title('ROC of RF in test data_feature select')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.0])
    plt.ylabel('tpr')
    plt.xlabel('fpr')
    plt.show()
    return accuracy_train,accuracy_test,recall_train,recall_test,precision_train,precision_test,f1_train,f1_test,predict_test, predict_proba
#######Adaboost######
def boost_pre(x_train,y_train,x_test,y_test):
    clf=AdaBoostClassifier(learning_rate=1.11)
    clf.fit(x_train,y_train)
    accuracy_train,accuracy_test,recall_train,recall_test,precision_train,precision_test,f1_train,f1_test,predict_test=classfision(clf,x_train,y_train,x_test,y_test)
    predict_proba=clf.predict_proba(x_test)
    print(predict_proba)

    fpr, tpr, thresholds = roc_curve(y_test, predict_proba[:, 1])
    roc_auc=auc(fpr,tpr)
    print("BOOST AUC:",roc_auc)
    plt.figure()
    plt.title('ROC of BOOST in test data_feature select')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.0])
    plt.ylabel('tpr')
    plt.xlabel('fpr')
    plt.show()
    return accuracy_train,accuracy_test,recall_train,recall_test,precision_train,precision_test,f1_train,f1_test,predict_test,predict_proba
######logistic#######
def log_pre(x_train,y_train,x_test,y_test):
    clf=LogisticRegression(solver='liblinear',max_iter=40)
    clf.fit(x_train,y_train)
    accuracy_train,accuracy_test,recall_train,recall_test,precision_train,precision_test,f1_train,f1_test,predict_test=classfision(clf,x_train,y_train,x_test,y_test)
    predict_proba=clf.predict_proba(x_test)
    print(predict_proba)
 
    fpr, tpr, thresholds = roc_curve(y_test, predict_proba[:, 1])
    roc_auc=auc(fpr,tpr)
    print("LOG AUC:",roc_auc)
    plt.figure()
    plt.title('ROC of LR-LMNN in test')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.0])
    plt.ylabel('tpr')
    plt.xlabel('fpr')
    plt.show()
    plt.savefig("output_log.png", dpi=300)
    return accuracy_train,accuracy_test,recall_train,recall_test,precision_train,precision_test,f1_train,f1_test,predict_test,predict_proba
#######svm#########
def svm_pre(x_train,y_train,x_test,y_test):
    clf=svm.SVC(kernel='linear',probability=True)
    clf.fit(x_train,y_train)
    accuracy_train,accuracy_test,recall_train,recall_test,precision_train,precision_test,f1_train,f1_test,predict_test=classfision(clf,x_train,y_train,x_test,y_test)
    predict_proba=clf.predict_proba(x_test)
    print(predict_proba)
    fpr, tpr, thresholds = roc_curve(y_test, predict_proba[:, 1])
    roc_auc=auc(fpr,tpr)
    print("SVM AUC:",roc_auc)
    plt.figure()
    plt.title('ROC of SVM-LMNN in test')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.0])
    plt.ylabel('tpr')
    plt.xlabel('fpr')
    plt.show()
    plt.savefig("output_svm.png", dpi=500)
    return accuracy_train,accuracy_test,recall_train,recall_test,precision_train,precision_test,f1_train,f1_test

###result
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
  #export excel
  ExcelFile=xlrd.open_workbook(r'E:\banyueban.xls')
  table = ExcelFile.sheets()[0]
  tables =np.zeros(shape=(157,30))
  label=np.zeros(shape=(157,1))
  tables,label=import_excel(table)
  tables=data_preprocessing(tables)
  x_train,y_train,x_test,y_test=read_data(tables,label)
  MLP_accuracy_train,MLP_accuracy_test,MLP_recall_train,MLP_recall_test,MLP_precision_train,MLP_precision_test,MLP_f1_train,MLP_f1_test,predict_test, predict_proba=MLP_pre(x_train,y_train,x_test,y_test)
  boost_accuracy_train,boost_accuracy_test,boost_recall_train,boost_recall_test,boost_precision_train,boost_precision_test,boost_f1_train,boost_f1_test,predict_test, predict_proba=boost_pre(x_train,y_train,x_test,y_test)
  RFF_accuracy_train,RFF_accuracy_test,RFF_recall_train,RFF_recall_test,RFF_precision_train,RFF_precision_test,RFF_f1_train,RFF_f1_test,predict_test, predict_proba=RFF_pre(x_train,y_train,x_test,y_test)
  svm_accuracy_train,svm_accuracy_test,svm_recall_train,svm_recall_test,svm_precision_train,svm_precision_test,svm_f1_train,svm_f1_test=svm_pre(x_train,y_train,x_test,y_test)
  log_accuracy_train,log_accuracy_test,log_recall_train,log_recall_test,log_precision_train,log_precision_test,log_f1_train,log_f1_test,predict_test, predict_proba=log_pre(x_train,y_train,x_test,y_test)
  print("svm Accuracy,Recall,Precision,F1:",svm_accuracy_test,svm_recall_test,svm_precision_test,svm_f1_test)
  print("log Accuracy,Recall,Precision,F1:",log_accuracy_test,log_recall_test,log_precision_test,log_f1_test)
  print("boost Accuracy,Recall,Precision,F1:",boost_accuracy_test,boost_recall_test,boost_precision_test,boost_f1_test)
  print("RFF Accuracy,Recall,Precision,F1:",RFF_accuracy_test,RFF_recall_test,RFF_precision_test,RFF_f1_test)
  print("MLP Accuracy,Recall,Precision,F1:",MLP_accuracy_test,MLP_recall_test,MLP_precision_test,MLP_f1_test)

  