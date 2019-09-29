import numpy as np
import xgboost as xgb
from basic_feature import *
from sklearn.pipeline import Pipeline

class classfy_pre:
    def __init__(self):
        self.steps = [
            ("drop_id",drop_id()),
            ("parse_area",parse_area_size()),
            ("parse_room",parse_rooms()),
            ("parse_old",parse_how_old()),
            ("height_enc",height_encoder()),
            ("ex_dist",extract_district()),
            ("label_dist",district_encoder()),
            ("acc_ext",access_extractor()),
            ("parking_encoder",parking_encoder()),
            ("drop_unnecessary",drop_unnecessary())
        ]

class poor_pre:
    def __init__(self):
        self.steps = [
            ("drop_id",drop_id()),
            ("parse_area",parse_area_size()),
            ("parse_room",parse_rooms()),
            ("parse_old",parse_how_old()),
            ("height_enc",height_encoder()),
            ("ex_dist",extract_district()),
            ("label_dist",district_encoder()),
            ("acc_ext",access_extractor()),
            ("parking_encoder",parking_encoder()),
            ("drop_unnecessary",drop_unnecessary())
        ]

class rich_pre:
    def __init__(self):
        self.steps = [
            ("drop_id",drop_id()),
            ("parse_area",parse_area_size()),
            ("parse_room",parse_rooms()),
            ("parse_old",parse_how_old()),
            ("height_enc",height_encoder()),
            ("ex_dist",extract_district()),
            ("label_dist",district_encoder()),
            ("acc_ext",access_extractor()),
            ("parking_encoder",parking_encoder()),
            ("drop_unnecessary",drop_unnecessary())
        ]

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
    def __init__(self,is_log=False,threshold=200000):
        hoge = classfy_pre()
        pre_step = hoge.steps
        m1 = [
            ("pre", Pipeline(steps = pre_step)),
            ("xgb",xgb.XGBRegressor(max_depth=8,min_child_weight=0,random_state=7777))
        ]
        m2 = [
            ("pre", Pipeline(steps = pre_step)),
            ("rfr",RandomForestRegressor(random_state=7777))
        ]
        m3 = [
            ("pre", Pipeline(steps = pre_step)),
            ("lgi",LogisticRegression())
        ]
        pp = poor_pre()
        ppstep =pp.steps
        poor_step_xgb = [
            ("pre",Pipeline(steps=ppstep)),
            ("xgb",xgb.XGBRegressor())
#           ("xgb",xgb.XGBRegressor(n_estimators=140,min_child_weight=5,max_depth=9,learning_rate=0.2,random_state=7777))
        ]
        poor_step_rfr = [
            ("pre",Pipeline(steps=ppstep)),
            ("rfr",RandomForestRegressor())
#           ("xgb",xgb.XGBRegressor(n_estimators=140,min_child_weight=5,max_depth=9,learning_rate=0.2,random_state=7777))
        ]
        rp = rich_pre()
        rpstep = rp.steps
        rich_step_xgb = [
            ("pre",Pipeline(steps=rpstep)),
            ("xgb",xgb.XGBRegressor())
#            ("xgb",xgb.XGBRegressor(n_estimators=140,min_child_weight=5,max_depth=7,learning_rate=0.1,random_state=7777)) 
        ]
        rich_step_rfr = [
            ("pre",Pipeline(steps=rpstep)),
            ("rfr",RandomForestRegressor())
#           ("xgb",xgb.XGBRegressor(n_estimators=140,min_child_weight=5,max_depth=9,learning_rate=0.2,random_state=7777))
        ]
        
        self.p_models = [
            Pipeline(steps=poor_step_xgb),
            Pipeline(steps=poor_step_rfr)
        ]
        self.r_models = [
            Pipeline(steps=rich_step_xgb),
            Pipeline(steps=rich_step_rfr)
        ]    
        self.c_models = [
            Pipeline(steps=m1),
            Pipeline(steps=m2),
            Pipeline(steps=m3),
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
            temp_y_poor = np.log(temp_y_poor.values)
            temp_y_rich = np.log(temp_y_rich.values)
        for model in self.p_models:
            model.fit(temp_x_poor,temp_y_poor)
        for model in self.r_models:
            model.fit(temp_x_rich,temp_y_rich)
        return self
    def predict(self,x):
        sep = classify_ensemble(self.c_models,x)
#         rich_predict = self.rich_model.predict(x)
#         poor_predict = self.poor_model.predict(x)
#         tot_predict = [0 for i in range(len(x.values))]
#         for i in range(len(x.values)):
#             tot_predict[i] = sep[i]*rich_predict[i]+(1-sep[i])*poor_predict[i]
#         return tot_predict
        
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
            poor_predict = np.exp(poor_predict)
        
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
    def get_params(self,deep=True):
        return {
        "threshold":self.threshold,
        "is_log":self.is_log
        }

def pre_checker(x,y):
    hoge = classfy_pre()
    pre_step = hoge.steps
    pp = poor_pre()
    ppstep =pp.steps
    rp = rich_pre()
    rpstep = rp.steps
    m1 = [
            ("pre", Pipeline(steps = pre_step)),
            ("summy",dummy())
        ]
    m2 = [
            ("pre", Pipeline(steps = ppstep)),
            ("summy",dummy())
        ]
    m3 = [
            ("pre", Pipeline(steps = rpstep)),
            ("summy",dummy())
        ]
    m1 = Pipeline(steps=m1)
    m2 = Pipeline(steps=m2)
    m3 = Pipeline(steps=m3)
    m1.fit(x,y)
    m2.fit(x,y)
    m3.fit(x,y)
    return m1,m2,m3