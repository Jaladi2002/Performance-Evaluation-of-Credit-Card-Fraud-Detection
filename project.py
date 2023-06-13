from email.mime import base
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import confusion_matrix, matthews_corrcoef
import itertools
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import ExtraTreesClassifier
import streamlit as st
#import main

st.title("Performance Evaluation of Credit Card Fraud Detection")
st.write("Evaluating the performance of different classifiers to predict whether the credit card transaction is fraudulent or not")

def metrics(cm):
  TP  = cm[1][1]
  TN = cm[0][0]
  FP = cm[0][1]
  FN = cm[1][0]
  accuracy = (TP+TN)/(TP+TN+FP+FN)
  precision = TP/(TP+FP) 
  recall = TP/(TP+FN)
  return accuracy,precision,recall

import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):


    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    return plt


data = pd.read_csv("creditcard.csv")
data = data.drop(['Time'],axis=1)
sc = StandardScaler()
amount = data['Amount'].values
data['Amount'] = sc.fit_transform(amount.reshape(-1,1))

X = data.iloc[:,:-1]
y = data.iloc[:,-1]

smote=SMOTE(random_state=0)
X,y = smote.fit_resample(X,y)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.3)

n = st.selectbox("Select Action", ("Select","Dataset","Run a model without Adaboost","Run a model with Adaboost"))
if n == "Run a model without Adaboost":
    name = st.selectbox("Select model",("Select","Random Forest","Decision Tree","Logistic Regression","Extra Tree","XGBoost"))
    def model_clf(name):
        st.header(name)
        if name=="Random Forest":
            random_forest = RandomForestClassifier()
            random_forest.fit(X_train,y_train)
            yrf_pred = random_forest.predict(X_test)
            cnf_rf = confusion_matrix(y_test,yrf_pred)
            accuracy,precision,recall = metrics(cnf_rf)
            mcc = matthews_corrcoef(y_test,yrf_pred)
            return (cnf_rf,accuracy,precision,recall,mcc)
        elif name=="Decision Tree":
            decision_tree = DecisionTreeClassifier(criterion='entropy',random_state=0)
            decision_tree.fit(X_train,y_train)
            ydt_pred = decision_tree.predict(X_test)
            cnf_dt = confusion_matrix(y_test,ydt_pred)
            accuracy,precision,recall = metrics(cnf_dt)
            mcc = matthews_corrcoef(y_test,ydt_pred)
            return (cnf_dt,accuracy,precision,recall,mcc)
        elif name=="Logistic Regression":
            logistic_reg = LogisticRegression(random_state = 42)
            logistic_reg.fit(X_train,y_train)
            lr_pred = logistic_reg.predict(X_test)
            cnf_lr = confusion_matrix(y_test,lr_pred)
            accuracy,precision,recall = metrics(cnf_lr)
            mcc = matthews_corrcoef(y_test,lr_pred)
            return (cnf_lr,accuracy,precision,recall,mcc)
        elif name=="Extra Tree":
            extra_tree = ExtraTreesClassifier()
            extra_tree.fit(X_train,y_train)
            extra_tree_pred = extra_tree.predict(X_test)
            cnf_et = confusion_matrix(y_test,extra_tree_pred)
            accuracy,precision,recall = metrics(cnf_et)
            mcc = matthews_corrcoef(y_test,extra_tree_pred)
            return (cnf_et,accuracy,precision,recall,mcc)
        elif name=="XGBoost":
            xg_boost = XGBClassifier()
            xg_boost.fit(X_train,y_train)
            xgb_pred = xg_boost.predict(X_test)
            cnf_xgb = confusion_matrix(y_test,xgb_pred)
            accuracy,precision,recall = metrics(cnf_xgb)
            mcc = matthews_corrcoef(y_test,xgb_pred)
            return (cnf_xgb,accuracy,precision,recall,mcc)
    
    if name!="Select":
        conf,acc,pre,rec,mcc = model_clf(name)
        st.write("Confusion matrix")
        st.write(conf)
        st.write("Accuracy(%): ",acc*100)
        st.write("Precision(%): ",pre*100)
        st.write("Recall(%): ",rec*100)
        st.write("MCC: ",mcc)
if n=="Dataset":
    st.header("Dataset")
    st.write(data)
    st.write("Instances: ",data.shape[0])
    st.write("Attributes: ",data.shape[1])
if n=="Run a model with Adaboost":
    model = st.selectbox("Select model",("Select","Random Forest","Decision Tree","Logistic Regression","Extra Tree"))
    st.header(model+" with Adaboost")
    def model_ada(model):
        if model=="Random Forest":
            rf_ada = AdaBoostClassifier(base_estimator=RandomForestClassifier(), random_state=1)
            rf_ada.fit(X_train, y_train)
            yrf_pred_ada = rf_ada.predict(X_test)
            cnf_rf_ada = confusion_matrix(y_test,yrf_pred_ada)
            accuracy,precision,recall = metrics(cnf_rf_ada)
            mcc = matthews_corrcoef(y_test,yrf_pred_ada)
            return (cnf_rf_ada,accuracy,precision,recall,mcc)
        elif model=="Decision Tree":
            dt_ada = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(),random_state=1)
            dt_ada.fit(X_train,y_train)
            ydt_pred_ada = dt_ada.predict(X_test)
            cnf_dt_ada = confusion_matrix(y_test,ydt_pred_ada)
            accuracy,precision,recall = metrics(cnf_dt_ada)
            mcc = matthews_corrcoef(y_test,ydt_pred_ada)
            return (cnf_dt_ada,accuracy,precision,recall,mcc)
        elif model=="Logistic Regression":
            lr_ada = AdaBoostClassifier(base_estimator=LogisticRegression(),random_state=1)
            lr_ada.fit(X_train,y_train)
            lr_pred_ada = lr_ada.predict(X_test)
            cnf_lr_ada = confusion_matrix(y_test,lr_pred_ada)
            accuracy,precision,recall = metrics(cnf_lr_ada)
            mcc = matthews_corrcoef(y_test,lr_pred_ada)
            return (cnf_lr_ada,accuracy,precision,recall,mcc)
        elif model=="Extra Tree":
            extra_tree_ada = AdaBoostClassifier(base_estimator=ExtraTreesClassifier(),random_state=1)
            extra_tree_ada.fit(X_train,y_train)
            extra_tree_pred_ada = extra_tree_ada.predict(X_test)
            cnf_et_ada = confusion_matrix(y_test,extra_tree_pred_ada)
            accuracy,precision,recall = metrics(cnf_et_ada)
            mcc = matthews_corrcoef(y_test,extra_tree_pred_ada)
            return (cnf_et_ada,accuracy,precision,recall,mcc)
        elif model=="XGBoost":
            xg_boost_ada = AdaBoostClassifier(base_estimator=XGBClassifier(),random_state=1)
            xg_boost_ada.fit(X_train,y_train)
            xgb_pred_ada = xg_boost_ada.predict(X_test)
            cnf_xgb_ada = confusion_matrix(y_test,xgb_pred_ada)
            accuracy,precision,recall = metrics(cnf_xgb_ada)
            mcc = matthews_corrcoef(y_test,xgb_pred_ada)
            return (cnf_xgb_ada,accuracy,precision,recall,mcc)
    
    if model!="Select":
        conf,acc,pre,rec,mcc = model_ada(model)
        st.write("Confusion matrix")
        st.write(conf)
        st.write("Accuracy(%): ",acc*100)
        st.write("Precision(%): ",pre*100)
        st.write("Recall(%): ",rec*100)
        st.write("MCC: ",mcc)