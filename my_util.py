import datetime
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd
import pickle

from models import *

def check_model(comment,train_x,train_y,write_importance=False):
    now = datetime.datetime.now()
    name = str(now.month)+"_"+str(now.day)+"_"+str(now.hour)+"_"+str(now.minute)
    f = open("logs.txt","a")
    f.write("\n")
    f.write(name + "\n")
    scores = cross_val_score(my_model(),train_x,train_y,scoring="neg_mean_squared_error",cv=4)
    scores = np.sqrt(-np.array(scores))
    print(scores)
    f.write("---cross val scoers---\n")
    for i in range(4):
        f.write(str(scores[i])+" ")
    f.write("\n")
    f.write(str(np.mean(scores)))
    f.write("\n")
    f.write("---comment---\n")
    f.write(comment)
    f.write("\n")

    if write_importance:
        easy = my_model()
        easy.fit(train_x,train_y)
        f = open("./feature_importances/"+name+"_ftxt","w")
        rm = easy.model
        d1 = rm[-1].get_booster().get_score(importance_type='gain')
        for key in d1:
            f.write(key+" "+str(d1[key])+"\n")
        f.write("\n")
        f.close()

def check_linear(comment,train_x,train_y,write_importance=False):
    now = datetime.datetime.now()
    name = str(now.month)+"_"+str(now.day)+"_"+str(now.hour)+"_"+str(now.minute)
    f = open("logs_linear.txt","a")
    f.write("\n")
    f.write(name + "\n")
    scores = cross_val_score(linear_model(),train_x,train_y,scoring="neg_mean_squared_error",cv=4)
    scores = np.sqrt(-np.array(scores))
    print(scores)
    f.write("---cross val scoers---\n")
    for i in range(4):
        f.write(str(scores[i])+" ")
    f.write("\n")
    f.write(str(np.mean(scores)))
    f.write("\n")
    f.write("---comment---\n")
    f.write(comment)
    f.write("\n")

    # if write_importance:
    #     easy = my_model()
    #     easy.fit(train_x,train_y)
    #     f = open("./feature_importances/"+name+"_ftxt","w")
    #     rm = easy.model
    #     d1 = rm[-1].get_booster().get_score(importance_type='gain')
    #     for key in d1:
    #         f.write(key+" "+str(d1[key])+"\n")
    #     f.write("\n")
    #     f.close()


    
def commit(train_x,train_y,test,name,seeds):
    pred_all = []
    for i in range(len(seeds)):
        print(i)
        model = my_model(seeds[i])
        model.fit(train_x,train_y)
        pred = model.predict(test)
        pred_all.append(pred)
        pickle.dump(model, open(name+"seed_"+str(seeds[i])+".pkl", "wb"))
    pred_all = np.array(pred_all)
    pred = np.mean(pred_all,axis=0)
    pred = pd.DataFrame(pred)
    pred.columns=["pred"]
    pred.index = test.index
    pred = pd.concat([test["id"],pred],axis=1)
    pred.to_csv(name+".csv",header=False,index=False)