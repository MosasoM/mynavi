import numpy as np
import xgboost as xgb
from basic_feature import *
from sklearn.pipeline import Pipeline
import lightgbm as lgbm
from sklearn.metrics import mean_squared_error
import datetime
from single import *
import pickle
from sklearn.metrics import accuracy_score

class classfy_pre:
    def __init__(self):
        self.steps = [
            ("parse_area",parse_area_size()),
            ("parse_room",parse_rooms()),
            ("parse_old",parse_how_old()),
            ("height_enc",height_encoder()),
            ("ex_dist",extract_district()),
            ("label_dist",district_encoder()),
            ("acc_ext",access_extractor()),
            ("tr_enc",train_encoder()),
            ("parking_encoder",parking_encoder()),
            ("dir_enc",direction_encoder()),
            ("info_enc",info_encoder()),
            ("m_d_p",add_mean_dist_price()),
            ("p_con_time",parse_contract_time()),
            ("fac",fac_encoder()),
            ("bath",bath_encoder()),
            ("kit",kitchin_encoder()),
            ("env",env_encoder()),
            ("angle_stat",add_mean_angle_price()),
            ("structure_enc",structure_label_encoder()),
            ("mean_struct",add_mean_structure_price()),
            # ("walk_label",walk_cat()),
            # ("mean_walk",add_mean_walk_price()),
            # ("mean_moyori",add_moyori_walk_price()),
            # ("mean_p0",add_p1_walk_price()),
            # ("mean_rldk",add_rldk_price()),
#             ("m_c_p",add_mean_city_price()), 精度低下
            ("cross",cross_features()),
            # ("kmeans_label",k_means_label()),
            # ("clust_label",add_clust_price()),
            # ("area_walk",area_and_walk_label()),
            # ("area_walk_mean",aw_label_mean()),
            ("drop_unnecessary",drop_unnecessary())
        ]

class poor_pre:
    def __init__(self):
        self.steps = [
            ("parse_area",parse_area_size()),
            ("parse_room",parse_rooms()),
            ("parse_old",parse_how_old()),
            ("height_enc",height_encoder()),
            ("ex_dist",extract_district()),
            ("label_dist",district_encoder()),
            ("acc_ext",access_extractor()),
            ("tr_enc",train_encoder()),
            ("parking_encoder",parking_encoder()),
            ("dir_enc",direction_encoder()),
            ("info_enc",info_encoder()),
            ("m_d_p",add_mean_dist_price()),
            ("p_con_time",parse_contract_time()),
            ("fac",fac_encoder()),
            ("bath",bath_encoder()),
            ("kit",kitchin_encoder()),
            ("env",env_encoder()),
            ("angle_stat",add_mean_angle_price()),
            ("structure_enc",structure_label_encoder()),
            ("mean_struct",add_mean_structure_price()),
            # ("walk_label",walk_cat()),
            ("mean_walk",add_mean_walk_price()),
            ("mean_moyori",add_moyori_walk_price()),
            # ("mean_p0",add_p1_walk_price()),
            # ("mean_rldk",add_rldk_price()),
#             ("m_c_p",add_mean_city_price()), 精度低下
            ("cross",cross_features()),
            ("kmeans_label",k_means_label()),
            ("clust_label",add_clust_price()),
            # ("area_walk",area_and_walk_label()),
            # ("area_walk_mean",aw_label_mean()),
            ("drop_unnecessary",drop_unnecessary())
        ]



class rich_pre:
    def __init__(self):
        self.steps = [
            ("parse_area",parse_area_size()),
            ("parse_room",parse_rooms()),
            ("parse_old",parse_how_old()),
            ("height_enc",height_encoder()),
            ("ex_dist",extract_district()),
            ("label_dist",district_encoder()),
            ("acc_ext",access_extractor()),
            ("tr_enc",train_encoder()),
            ("parking_encoder",parking_encoder()),
            ("dir_enc",direction_encoder()),
            ("info_enc",info_encoder()),
            ("m_d_p",add_mean_dist_price()),
            ("p_con_time",parse_contract_time()),
            ("fac",fac_encoder()),
            ("bath",bath_encoder()),
            ("kit",kitchin_encoder()),
            ("env",env_encoder()),
            ("angle_stat",add_mean_angle_price()),
            ("structure_enc",structure_label_encoder()),
            ("mean_struct",add_mean_structure_price()),
            # ("walk_label",walk_cat()),
            ("mean_walk",add_mean_walk_price()),
            ("mean_moyori",add_moyori_walk_price()),
            # ("mean_p0",add_p1_walk_price()),
            # ("mean_rldk",add_rldk_price()),
#             ("m_c_p",add_mean_city_price()), 精度低下
            ("cross",cross_features()),
            ("kmeans_label",k_means_label()),
            ("clust_label",add_clust_price()),
            # ("area_walk",area_and_walk_label()),
            # ("area_walk_mean",aw_label_mean()),
            ("drop_unnecessary",drop_unnecessary())
        ]


def pre_checker(x,y):
    # hoge = classfy_pre()
    # pre_step = hoge.steps
    # pp = poor_pre()
    # ppstep =pp.steps
    rp = rich_pre()
    rpstep = rp.steps
    # m1 = [
    #         ("pre", Pipeline(steps = pre_step)),
    #         ("summy",dummy())
    #     ]
    # m2 = [
    #         ("pre", Pipeline(steps = ppstep)),
    #         ("summy",dummy())
    #     ]
    m3 = [
            ("pre", Pipeline(steps = rpstep)),
            ("summy",dummy())
        ]
    # m1 = Pipeline(steps=m1)
    # m2 = Pipeline(steps=m2)
    m3 = Pipeline(steps=m3)
    # m1.fit(x,y)
    # m2.fit(x,y)
    m3.fit(x,y)
    return m3


        
        
class easy_model_poor:
    def __init__(self,is_log=False):
        rp = poor_pre()
        rpstep = rp.steps
        rich_step_xgb = [
            ("pre",Pipeline(steps=rpstep)),
            ("xgb",xgb.XGBRegressor(random_state=8888,max_depth=8))
        ]
        rich_step_rfr = [
            ("pre",Pipeline(steps=rpstep)),
            ("rfr",RandomForestRegressor(random_state=8888,max_depth=8))
        ]
        rich_step_lgbm = [
            ("pre",Pipeline(steps=rpstep)),
            ("lgbm",lgbm.LGBMRegressor(random_state=8888,max_depth=8))
        ]
        self.models = [
            Pipeline(steps=rich_step_xgb),
            # Pipeline(steps=rich_step_rfr),
            # Pipeline(steps=rich_step_lgbm),
        ]
        self.is_log=is_log
    def fit(self,x,y):
        tar = y
        if self.is_log:
            tar = np.log(y)
        for model in self.models:
            model.fit(x,tar)
        return self
    def predict(self,x):
        temp = np.zeros(len(x.values))
        for model in self.models:
            pred = model.predict(x)
            temp += np.array(pred)
        temp = temp/len(self.models)
        predict = temp
        if self.is_log:
            predict = np.exp(predict)
        return predict
    
    def get_params(self,deep=True):
        return {}

class easy_model_rich:
    def __init__(self,is_log=False):
        rp = rich_pre()
        rpstep = rp.steps
        rich_step_xgb = [
            ("pre",Pipeline(steps=rpstep)),
            ("xgb",xgb.XGBRegressor(random_state=8888,max_depth=8))
        ]
        rich_step_rfr = [
            ("pre",Pipeline(steps=rpstep)),
            ("rfr",RandomForestRegressor(random_state=8888,max_depth=8))
        ]
        rich_step_lgbm = [
            ("pre",Pipeline(steps=rpstep)),
            ("lgbm",lgbm.LGBMRegressor(random_state=8888,max_depth=8))
        ]
        self.models = [
            Pipeline(steps=rich_step_xgb),
            # Pipeline(steps=rich_step_rfr),
            # Pipeline(steps=rich_step_lgbm),
        ]
        self.is_log=is_log
    def fit(self,x,y):
        tar = y
        if self.is_log:
            tar = np.log(y)
        for model in self.models:
            model.fit(x,tar)
        return self
    def predict(self,x):
        temp = np.zeros(len(x.values))
        for model in self.models:
            pred = model.predict(x)
            temp += np.array(pred)
        temp = temp/len(self.models)
        predict = temp
        if self.is_log:
            predict = np.exp(predict)
        return predict
    
    def get_params(self,deep=True):
        return {}

def check_poor(comment,train_x,train_y):
    now = datetime.datetime.now()
    name = str(now.month)+"_"+str(now.day)+"_"+str(now.hour)+"_"+str(now.minute)
    f = open("logs_poor.txt","a")
    f.write("\n")
    f.write(name + "\n")
    scores = cross_val_score(easy_model_poor(),train_x,train_y,scoring="neg_mean_squared_error",cv=4)
    scores = np.sqrt(-np.array(scores))
    print(scores)
    f.write("---cross val scoers---\n")
    for i in range(4):
        f.write(str(scores[i])+" ")
    f.write("\n")
    f.write("---comment---\n")
    f.write(comment)
    f.write("\n")
    
    easy = easy_model_poor()
    easy.fit(train_x,train_y)
    f = open("./feature_importances/"+name+"_f_poor.txt","w")
    rm = easy.models[0]
    d1 = rm[-1].get_booster().get_score(importance_type='gain')
    for key in d1:
        f.write(key+" "+str(d1[key])+"\n")
    f.write("\n")
    f.close()

def check_rich(comment,train_x,train_y):
    now = datetime.datetime.now()
    name = str(now.month)+"_"+str(now.day)+"_"+str(now.hour)+"_"+str(now.minute)
    f = open("logs_rich.txt","a")
    f.write("\n")
    f.write(name + "\n")
    scores = cross_val_score(easy_model_rich(),train_x,train_y,scoring="neg_mean_squared_error",cv=4)
    scores = np.sqrt(-np.array(scores))
    print(scores)
    f.write("---cross val scoers---\n")
    for i in range(4):
        f.write(str(scores[i])+" ")
    f.write("\n")
    f.write("---comment---\n")
    f.write(comment)
    f.write("\n")
    
    easy = easy_model_rich()
    easy.fit(train_x,train_y)
    f = open("./feature_importances/"+name+"_f_rich.txt","w")
    rm = easy.models[0]
    d1 = rm[-1].get_booster().get_score(importance_type='gain')
    for key in d1:
        f.write(key+" "+str(d1[key])+"\n")
    f.write("\n")
    f.close()
    
def each_models(train_x,train_y):
    easy = easy_model()
    easy.fit(train_x,train_y)
    p1 = easy.models[0].predict(train_x)
    p2 = easy.models[1].predict(train_x)
    p3 = easy.models[2].predict(train_x)
    print(np.sqrt(mean_squared_error(p1,train_y)))
    print(np.sqrt(mean_squared_error(p2,train_y)))
    print(np.sqrt(mean_squared_error(p3,train_y)))
    
def commit(model,train_x,train_y,test,name):
    model.fit(train_x,train_y)
    pred = model.predict(test)
    pred = pd.DataFrame(pred)
    pred.columns=["pred"]
    pred.index = test.index
    pred = pd.concat([test["id"],pred],axis=1)
    pred.to_csv(name+".csv",header=False,index=False)
    pickle.dump(model, open(name+".pkl", "wb"))





def classify_ensemble(models,x_valid):
    p = []
    predict = [0 for i in range(len(x_valid))]
    for m in models:
        p.append(m.predict(x_valid))
    for i in range(len(x_valid)):
        temp = 0
        for j in range(len(models)):
            temp += p[j][i]/3
        predict[i] = temp
        if temp > 0.9:
            predict[i] = 1
        else:
            predict[i] = 0
    predict = np.array(predict,dtype=np.int64)
    return predict

class bin_model_score:
    def __init__(self,is_log=True,threshold=200000):
        hoge = classfy_pre()
        pre_step = hoge.steps
        m1 = [
            ("pre", Pipeline(steps = pre_step)),
            ("xgb",xgb.XGBRegressor(max_depth=8,min_child_weight=0,random_state=7777,objective="reg:logistic"))
        ]
        m2 = [
            ("pre", Pipeline(steps = pre_step)),
            ("rfr",RandomForestRegressor(random_state=7777))
        ]
        m3 = [
            ("pre", Pipeline(steps = pre_step)),
            ("lgi",LogisticRegression(random_state=7777))
        ]
        m4 = [
            ("pre", Pipeline(steps = pre_step)),
            ("lgbm",lgbm.LGBMRegressor(random_state=7777))
        ]
        
        pp = poor_pre()
        ppstep =pp.steps
        poor_step_xgb = [
            ("pre",Pipeline(steps=ppstep)),
            ("xgb",xgb.XGBRegressor(random_state=7777))
#           ("xgb",xgb.XGBRegressor(n_estimators=140,min_child_weight=5,max_depth=9,learning_rate=0.2,random_state=7777))
        ]
        poor_step_rfr = [
            ("pre",Pipeline(steps=ppstep)),
            ("rfr",RandomForestRegressor(random_state=7777))
        ]
        poor_step_lgbm = [
            ("pre",Pipeline(steps=ppstep)),
            ("lgbm",lgbm.LGBMRegressor(random_state=7777))
        ]
        
        
        rp = rich_pre()
        rpstep = rp.steps
        rich_step_xgb = [
            ("pre",Pipeline(steps=rpstep)),
            ("xgb",xgb.XGBRegressor(random_state=7777,max_depth=8))
#            ("xgb",xgb.XGBRegressor(n_estimators=140,min_child_weight=5,max_depth=7,learning_rate=0.1,random_state=7777)) 
        ]
        rich_step_rfr = [
            ("pre",Pipeline(steps=rpstep)),
            ("rfr",RandomForestRegressor(random_state=7777,max_depth=8))
        ]
        rich_step_lgbm = [
            ("pre",Pipeline(steps=rpstep)),
            ("lgbm",lgbm.LGBMRegressor(random_state=7777,max_depth=8))
        ]
        
        self.p_models = [
            Pipeline(steps=poor_step_xgb),
            # Pipeline(steps=poor_step_rfr),
            # Pipeline(steps=poor_step_lgbm),
        ]
        self.r_models = [
            Pipeline(steps=rich_step_xgb),
            # Pipeline(steps=rich_step_rfr),
            # Pipeline(steps=rich_step_lgbm),
        ]    
        self.c_models = [
            Pipeline(steps=m1),
            # Pipeline(steps=m2),
            Pipeline(steps=m3),
            Pipeline(steps=m4),
        ]
        self.threshold = threshold
        self.is_log = is_log
    def fit(self,x,y):
        temp = y.values
        is_rich_label = [0 for i in range(len(temp))]
        for i in range(len(temp)):
            if temp[i] > self.threshold:
                is_rich_label[i] = 1
        for i in range(len(self.c_models)):
            self.c_models[i].fit(x,is_rich_label)
        temp = pd.concat([x,y],axis=1)
        th = self.threshold
        train_rich = temp.query("賃料 > @th")
        train_poor = temp.query("賃料 <= @th")
        temp_y_rich = train_rich["賃料"]
        temp_x_rich = train_rich.drop(["賃料"],axis = 1)
        temp_y_poor = train_poor["賃料"]
        temp_x_poor = train_poor.drop(["賃料"],axis = 1)
        if self.is_log:
            temp_y_rich = np.log(temp_y_rich.values)
        for model in self.p_models:
            model.fit(temp_x_poor,temp_y_poor)
        for model in self.r_models:
            model.fit(temp_x_rich,temp_y_rich)
        return self
    def predict(self,x):
        sep = classify_ensemble(self.c_models,x)
        
        tar = x.assign(isrich=sep)
        rich_group = tar.query("isrich == 1")
        poor_group = tar.query("isrich == 0")
        rich_group = rich_group.drop("isrich",axis=1)
        poor_group = poor_group.drop("isrich",axis=1)
        
        temp = np.zeros(len(rich_group.values))
        for model in self.r_models:
            pred = model.predict(rich_group)
            temp += np.array(pred)
        temp = temp/len(self.r_models)
        rich_predict = temp
        
        temp = np.zeros(len(poor_group.values))
        for model in self.p_models:
            pred = model.predict(poor_group)
            temp += np.array(pred)
        temp = temp/len(self.p_models)
        poor_predict = temp
        
        if self.is_log:
            rich_predict = np.exp(rich_predict)
        
        rich_predict = pd.DataFrame(rich_predict)
        poor_predict = pd.DataFrame(poor_predict)
        rid = rich_group["id"].reset_index(drop=True)
        pid = poor_group["id"].reset_index(drop=True)
        rich_predict = rich_predict.reset_index(drop=True)
        poor_predict = poor_predict.reset_index(drop=True)
        rich_predict = pd.concat([rid,rich_predict],axis = 1)
        poor_predict = pd.concat([pid,poor_predict],axis = 1)
        rich_predict.head()
        ans = pd.concat([rich_predict,poor_predict],ignore_index=True)
        ans.columns = ["id","Price"]
        buf = []
        for num in x["id"]:
            buf.append(ans.query("id==@num")["Price"].values)
        return buf
    def each_predict_score(self,train_x,train_y):
#         temp_x = train_x.drop("賃料",axis = 1)
#         temp_y = train_x["賃料"]
        x_train,x_valid,y_train,y_valid = train_test_split(train_x,train_y,random_state=7777)
        self.fit(x_train,y_train)
        th = self.threshold
        hoge = pd.concat([x_valid,y_valid],axis = 1)
        rich_group = hoge.query("賃料 > @th")
        poor_group = hoge.query("賃料 <= @th")
        rich_y = rich_group["賃料"].values
        poor_y = poor_group["賃料"].values
        rich_group = rich_group.drop("賃料",axis=1)
        poor_group = poor_group.drop("賃料",axis=1)
        
        temp = np.zeros(len(rich_group.values))
        for model in self.r_models:
            pred = model.predict(rich_group)
            temp += np.array(pred)
        temp = temp/len(self.r_models)
        rich_predict = temp
        rich_predict = np.exp(rich_predict)
        
        temp = np.zeros(len(poor_group.values))
        for model in self.p_models:
            pred = model.predict(poor_group)
            temp += np.array(pred)
        temp = temp/len(self.p_models)
        poor_predict = temp
        
        score1 = mean_squared_error(rich_predict,rich_y)
        score2 = mean_squared_error(poor_predict,poor_y)
        score1 = np.sqrt(score1)
        score2 = np.sqrt(score2)
        
        return score1,score2
        
        
        
    def get_params(self,deep=True):
        return {
        "threshold":self.threshold,
        "is_log":self.is_log
        }

def check_specs(comment,train_x,train_y,border=200000):
    now = datetime.datetime.now()
    name = str(now.month)+"_"+str(now.day)+"_"+str(now.hour)+"_"+str(now.minute)
    f = open("logs.txt","a")
    f.write("\n")
    f.write(name + "\n")
    scores = cross_val_score(bin_model_score(threshold=border),train_x,train_y,scoring="neg_mean_squared_error")
    scores = np.sqrt(-np.array(scores))
    print(scores)
    f.write("---cross val scoers---\n")
    for i in range(3):
        f.write(str(scores[i])+" ")
    f.write("\n")
    bms = bin_model_score(threshold=border)
    rich_score,poor_score = bms.each_predict_score(train_x,train_y)
    print(rich_score)
    print(poor_score)
    f.write("----rich socre,poor score---\n")
    f.write(str(rich_score)+" "+str(poor_score)+"\n")
    f.write("---comment---\n")
    f.write(comment)
    f.write("\n")
    f.close()
    f = open("./feature_importances/"+name+"_f.txt","w")
    rm = bms.r_models[0]
    pm = bms.p_models[0]
    d1 = rm[-1].get_booster().get_score(importance_type='gain')
    d2 = pm[-1].get_booster().get_score(importance_type='gain')
    for key in d1:
        f.write(key+" "+str(d1[key])+"\n")
    f.write("\n")
    f.write("*************************\n")
    for key in d2:
        f.write(key+" "+str(d2[key])+"\n")

def check_splitter(x_train,x_valid,y_train,y_valid,threshold=150000):
    temp = y_train.values
    is_rich_label = [0 for i in range(len(temp))]
    hoge = classfy_pre()
    pre_step = hoge.steps
    m1 = [
        ("pre", Pipeline(steps = pre_step)),
        ("xgb",xgb.XGBRegressor(random_state=7777,objective="reg:logistic"))
    ]
    m2 = [
        ("pre", Pipeline(steps = pre_step)),
        ("rfr",RandomForestRegressor(random_state=7777))
    ]
    m3 = [
        ("pre", Pipeline(steps = pre_step)),
        ("lgi",LogisticRegression(random_state=7777))
    ]
    m4 = [
        ("pre", Pipeline(steps = pre_step)),
        ("lgbm",lgbm.LGBMRegressor(random_state=7777))
    ]
    models = [
        Pipeline(steps=m1),
        Pipeline(steps=m2),
        Pipeline(steps=m3),
        Pipeline(steps=m4)
    ]
    for i in range(len(temp)):
        if temp[i] > threshold:
            is_rich_label[i] = 1
    for i in range(len(models)):
        models[i].fit(x_train,is_rich_label)

    pred = classify_ensemble(models,x_valid)
    temp = y_valid.values
    is_rich_label = [0 for i in range(len(temp))]
    for i in range(len(temp)):
        if temp[i] > threshold:
            is_rich_label[i] = 1
    print(accuracy_score(pred,is_rich_label))

    