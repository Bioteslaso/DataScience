from Pablo_MLAlgorithm_Cosine_dim16 import train_dfs
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
from random import sample
import numpy as np

def  similarity(e,type):
                    if type==1:#Cosine similarity
                            return dot(X[e[0]], X[e[1]])/(norm(X[e[0]])*norm(X[e[1]]))
                    if type==2:#Euclidean similarity
                            return np.linalg.norm(X[e[0]]-X[e[1]])
                    return 0

dname=['government_edges']#'politician_edges'Done  Lucas:'company_edges',
dim_list=[16]#,64,128]-----Only dimension = 16
similarity_functions = [1]#, 2]----Only Cosine similarity
iter = 0

for dataSetIndex in range(2):#data set
 for d in dim_list:

   #-------------------reading data set + node Embedding 
   df_path="Dataset/facebook/"+dname[dataSetIndex]+".txt"#politician_edges,company_edges, government_edges
   df = pd.read_csv(df_path)#---test-1  ----Is a connected directed graph
            
   G = nx.from_pandas_edgelist(df, "node_1","node_2", create_using=nx.DiGraph())
            
   print("DeepWalk starts...")

   # train model and generate embedding
   model = DeepWalk(walk_length=100, dimensions=d, window_size=1)
   model.fit(G)
   embedding = model.get_embedding()
   print("End of DeepWalk")

   nodes =list(range(len(G)))
   X = embedding[nodes]
   for sim_func in similarity_functions:
        
            #----------------------
            #parameter2: datasetfile:
            k=int(G.number_of_edges()*0.1)
            edges_as_x_test=sample(G.edges(), k) 

            num_of_test_edges = len(edges_as_x_test)
            num_of_train_edges = G.number_of_edges()-num_of_test_edges# len(G) is equal to G.number of nodes without isolated nodes(after removing10%edges)

            print("number of train edges:", num_of_train_edges )

            num_of_train_rows=num_of_train_edges*2
            print("number of train edges*2:", num_of_train_rows )

            x_train=np.zeros( num_of_train_rows )
            y_train=np.zeros( num_of_train_rows )

            

            row=-1# for tracing rows in x_train 
            #for i,j in G.edges():
            for e in G.edges():
                    if e not in edges_as_x_test:

                            row=row+1
                            x_train[row] = similarity(e,sim_func)
                            #x_train[row][1] = similarity(e,2)

                            y_train[row]=1 #means src---->dest  edge exists         
            print("last row for xtrain with y=1",row)
            #-----------------x train for non edges


            nedges = list(nx.non_edges(G))
            #for i,j in nedges:
            row=row+1
            start_none_ind_in_x_train=row

            nedgesStart_ind=row

            k=num_of_train_rows-nedgesStart_ind#we need this amount of nonedges for training
            non_edge_random=sample(nedges, k) 

            for none in non_edge_random:#nedges:
                if row<num_of_train_rows:
                        x_train[row] =similarity(none, sim_func) #dot(X[none[0]], X[none[1]])/(norm(X[none[0]])*norm(X[none[1]]))
                       # x_train[row][1] =similarity(none,2) #dot(X[none[0]], X[none[1]])/(norm(X[none[0]])*norm(X[none[1]]))
                        row=row+1
                        #y_train[row]=0 #means src ---->dest does not exist no need . y_train is zero array

            print("end train nonedge")
            #-----------------
            #DATAFRAM
            xt1 = pd.DataFrame(x_train,columns=['Similarity'])
            yt1 = pd.DataFrame(y_train,columns=['hasEdge'])

            xt=xt1[['Similarity']]
            yt=yt1['hasEdge']
            #-----------------COPY
            num_of_Test_samples=len(edges_as_x_test)

            x_test=np.zeros(num_of_Test_samples)#np.zeros(( num_of_test_samples , dimension*2), dtype=float)#dimension of embedding:64
            #y_test=np.ones((num_of_Test_samples), dtype=float)
            row=0

            for (i,j) in edges_as_x_test:
                            x_test[row] = similarity((i,j),sim_func)#dot(X[i], X[j])/(norm(X[i])*norm(X[j]))
                            row=row+1

            n_balanced_test_sample=num_of_Test_samples*2
            x_test_balanced = np.zeros(n_balanced_test_sample)
            y_true = np.zeros(n_balanced_test_sample)

            print("num num_of_Test_samples,",num_of_Test_samples)
            print("n_balanced_test_sample",n_balanced_test_sample)
            #-------------------------------------Start random none for test
            x = 0#non_edge_random
            ne_ind=0
            for i in range( num_of_Test_samples):#should be predict y=0 from train(from rows related to nedges)
                    
                    ne = nedges[ne_ind]
                    while(ne in non_edge_random):
                            ne_ind = ne_ind + 1
                            ne=nedges[ne_ind]
                    x_test_balanced[i]=similarity(ne,sim_func)
                    x = x + 1
            print("************ after balance testingt:y=0")  

            #-------------------------------------End random nedge for test
            t = 0
            while(x<n_balanced_test_sample):# Should be predicted y=1
                    x_test_balanced[x] = x_test[t]

                    y_true[x] = 1
                    x=x+1
                    t=t+1

            print("************ after balance testingt:y=1")  
            xtestdf1=pd.DataFrame(x_test_balanced, columns=['Similarity'])
            xtestdf=xtestdf1[['Similarity']]
            #--------------------END Copy
            #For each model
            for M in range(1,5):#7):#8):  without 8(SVM model) 

                #----------------------
                t = train_dfs(d,dname[dataSetIndex], M, sim_func)#politician_edges,company_edges, government_edges
                t.tuning_model(xt, yt, xtestdf, y_true)
                iter= iter+1
                print("Similarity_"+str(sim_func)+"_dname:"+dname[dataSetIndex]+" dim_list:"+str(d)+" ModelNumber:"+str(M)+"iteration:",str(iter))
       
