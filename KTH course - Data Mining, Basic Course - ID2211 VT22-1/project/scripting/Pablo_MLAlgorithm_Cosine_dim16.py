from multiprocessing.sharedctypes import Value
import networkx as nx
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from karateclub import DeepWalk
from sklearn.svm import SVC


from numpy import dot
from numpy.linalg import norm

import networkx as nx
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from karateclub import DeepWalk
import openpyxl 

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import brier_score_loss
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

import numpy as np

class train_dfs:
    def __init__(self, d, df_n, model_number, simfunc):
        self.dimension = d
        self.df_name=df_n
        self.model_param = model_number
        self.sim_func = simfunc

            
    def tuning_model(self, xt, yt, xtestdf, y_true):
            RANDOM_STATE=0
            #----------------------------SET THE MODEL
            #model_param =6
            if self.model_param == 1:
                            print("RandomForest...")
                            n_estimators=[100,300,500,1000]
                            max_depth=[5,8,15,25,30]
                            min_samples_split=[5,10,15,100]
                            min_samples_leaf=[1, 2,5,10]


                            parameter_candidates=dict(n_estimators=n_estimators,max_depth=max_depth
                            ,min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
   
                            model = GridSearchCV(estimator=RandomForestClassifier(random_state=RANDOM_STATE), 
                            param_grid=parameter_candidates, scoring='roc_auc', cv=5,
                            refit=True, error_score=0, n_jobs=-1)
                            modelName="RandomForest"
           
            elif self.model_param == 2:
                            print("DecisionTree...")
                            parameter_candidates = {'criterion': ['entropy','gini'], 'max_depth': [6,8,10,12]}
                            model = GridSearchCV(DecisionTreeClassifier(random_state=RANDOM_STATE), parameter_candidates, cv=5,  scoring='roc_auc')
                            modelName="DecisionTree"
           
            elif self.model_param == 3:
                            print("GradientBoostingClassifier...")
                            parameter_candidates = {'learning_rate': [0.01, 0.02, 0.03,0.05, 0.06,0.07, 0.08,0.09,0.1]
                            ,'subsample': [0.9, 0.5, 0.2], 'n_estimators' : [100,500, 1000]
                             , 'max_depth': [4,5,6,8]
                             }
    
                            model = GridSearchCV(estimator=GradientBoostingClassifier(random_state=RANDOM_STATE), 
                                param_grid = parameter_candidates, scoring = 'roc_auc', cv = 5, n_jobs=1)
    
                            modelName="GradientBoostingClassifier"
            elif self.model_param == 4:
                            print("AdaBoostClassifier...")
                            abc = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(random_state=RANDOM_STATE))

                            parameter_candidates = {"base_estimator__criterion" : ["gini", "entropy"] ,
                                          "base_estimator__splitter" : ["best", "random"], 
                                          "n_estimators": [10,50, 250,500,1000]
                                          ,'learning_rate':[0.01,0.02, 0.03,0.04,0.05,0.06,0.07,0.08,0.09, 0.1]
                                        }
                            model = GridSearchCV(abc, param_grid=parameter_candidates,verbose=3,scoring='roc_auc',n_jobs=-1)
                            modelName="AdaBoostClassifier"


            elif self.model_param == 5:
                            print("XGBClassifier...")
                            parameter_candidates = {#------------XGBOOST 
                                            'min_child_weight': [1, 5, 10],
                                            #'gamma': [0.5, 1,1.5,  2],
                                            'subsample': [0.6, 0.8, 1.0],
                                            'max_depth': [3, 4, 5,10],
                                            "learning_rate":[0.01,0.03, 0.3, 0.1],
                                            "n_estimators":[100, 500, 1000]
                                            }
            
                            model = GridSearchCV(estimator=XGBClassifier(random_state=RANDOM_STATE), 
                                    param_grid=parameter_candidates, scoring='roc_auc', n_jobs= -1, cv=5 )
                            modelName = "XGBClassifier"


            elif self.model_param == 6:
                            print("LogisticRegression...")
                            #-----------------------------
                            parameter_candidates = {'C':[0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]}#{'penalty': ['l1', 'l2'], 'C':np.logspace(-3,3,7)# [0.01, 0.1, 1, 10, 100]
                           
                            logreg = LogisticRegression()
                            model = GridSearchCV(logreg, param_grid = parameter_candidates,scoring='roc_auc',cv=5)
                            modelName="LogisticRegression"
           ###############
            elif self.model_param == 7:
                            print("SVM...")
                            parameter_candidates = {'C': [0.1, 1, 10, 100, 1000]
                            , 'gamma': [ 0.1, 0.01, 0.001, 0.0001]
                            ,'kernel': ['rbf','linear']
                                                 }
                            model = GridSearchCV(SVC(probability=True,random_state=RANDOM_STATE),
                                parameter_candidates, scoring='roc_auc')
                            modelName="SVM"
            #----------------------------END SET THE MODEL
            import time

            start_time = time.time()
            model.fit(xt, yt)
            model_time=time.time() - start_time 
            
            print("End of training: after fit function-----------")

            #-----------------
            #predict for i,j not in G.edges   compare with primary
            y_pred_balanced=model.predict_proba(xtestdf)
            y_pred_balanced_class=model.predict(xtestdf)
            #END OF THE TRAIN AND TEST PHASE


            ytestdf_balan1=pd.DataFrame(y_true, columns=['hasEdge'])
            y_true_balanc=ytestdf_balan1['hasEdge']
            #[[1]]---AUC_SCORE  
            auc_score = roc_auc_score(y_true_balanc, y_pred_balanced[:,1])#y_true, y_pred from balanced test (x_test_balanced)
            print("auc:",auc_score)
            #[[2]]---MEAN_SQUARED_ERROR
            mserror=mean_squared_error(y_true_balanc, y_pred_balanced[:,1])#regression metric
            print("MSError",mserror)

            #Classification metrics:

            #[[3]]---ACCURACY_SCORE
            acc= accuracy_score(y_true_balanc, y_pred_balanced_class)
            #[[4]]---RECALL
            recall = recall_score(y_true_balanc, y_pred_balanced_class)
            #[[5]]---PRECISION
            pre = precision_score(y_true_balanc, y_pred_balanced_class)
            #[[6]]---BRIER_SCORE
            bri = brier_score_loss(y_true_balanc, y_pred_balanced_class,pos_label=1)

            #[[7]]---F1_SCORE
            F1 = f1_score(y_true_balanc, y_pred_balanced_class)


            #confusion_matrix=pd.crosstab(y_test,y_pred, rownames=['Actual'], colnames=['Predicted'])
            #sn.heatmap(confusion_matrix, annot=True)
            #plt.show()
            #---------------------------------WRITING TO EXCEL FILE
                    # to create a new blank Workbook object
            wb = openpyxl.Workbook()
                    
                    # Get workbook active sheet  
                    # from the active attribute
            sheet = wb.active
                    #--------------
            c1 = sheet['A1']
            c1.value = "AUC"
                    
                    # B2 means column = 2 & row = 2.
            c2 = sheet['B1']
            c2.value = "MSerror"
                    
            c3 = sheet['C1']
            c3.value = "accuracy"
                    
            c4 = sheet['D1']
            c4.value = "recall"
                    
            c5 = sheet['E1']
            c5.value = "precision"

            c6 = sheet['F1']
            c6.value = "brier_score"
                    
            c7 = sheet['G1']
            c7.value = "F1_score"

            c8 = sheet['H1']
            c8.value = "Time"
            #--------------
            v1 = sheet['A2']
            v1.value = auc_score
                    
                    # B2 means column = 2 & row = 2.
            v2 = sheet['B2']
            v2.value = mserror
                    
            v3 = sheet['C2']
            v3.value = acc
                    
            v4 = sheet['D2']
            v4.value = recall
                    
            v5 = sheet['E2']
            v5.value = pre

            v6 = sheet['F2']
            v6.value = bri
                    
            v7 = sheet['G2']
            v7.value = F1

            v8 = sheet['H2']
            v8.value = model_time

            v9 = sheet['I2']
            v9.value= str(model.best_params_)

            name="similarity_"+str(self.sim_func)+"_"+modelName+"_dimension_"+str(self.dimension)+"_dfName_"+self.df_name
            wb.save(name+".xlsx")
            #-----------------------------------------CALLING ML FUNCTIONS
    