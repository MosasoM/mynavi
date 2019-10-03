from rich_model import *
from poor_model import *

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
            # ("m_d_p",add_mean_dist_price()),
            ("p_con_time",parse_contract_time()),
            ("fac",fac_encoder()),
            ("bath",bath_encoder()),
            ("kit",kitchin_encoder()),
            ("env",env_encoder()),
            # ("angle_stat",add_mean_angle_price()),
            ("structure_enc",structure_label_encoder()),
            # ("mean_struct",add_mean_structure_price()),
            ("cross",cross_features()),
            ("drop_unnecessary",drop_unnecessary())
        ]

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
    def __init__(self,threshold=150000):
        hoge = classfy_pre()
        pre_step = hoge.steps
        m1 = [
            ("pre", Pipeline(steps = pre_step)),
            ("xgb",xgb.XGBRegressor(max_depth=8,min_child_weight=0,random_state=7777,objective="reg:logistic"))
        ]
        m2 = [
            ("pre", Pipeline(steps = pre_step)),
            ("lgi",LogisticRegression(random_state=7777))
        ]
        m3 = [
            ("pre", Pipeline(steps = pre_step)),
            ("lgbm",lgbm.LGBMRegressor(random_state=7777))
        ]
        self.p_models = [
            easy_model_poor()
        ]
        self.r_models = [
            easy_model_rich()
        ]    
        self.c_models = [
            Pipeline(steps=m1),
            Pipeline(steps=m2),
            Pipeline(steps=m3),
        ]
        self.threshold = threshold
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
        }

def check_specs(comment,train_x,train_y,border=150000):
    now = datetime.datetime.now()
    name = str(now.month)+"_"+str(now.day)+"_"+str(now.hour)+"_"+str(now.minute)
    f = open("logs.txt","a")
    f.write("\n")
    f.write(name + "\n")
    scores = cross_val_score(bin_model_score(threshold=border),train_x,train_y,scoring="neg_mean_squared_error",cv=4)
    scores = np.sqrt(-np.array(scores))
    print(scores)
    print(np.mean(scores))
    f.write("---cross val scoers---\n")
    for i in range(4):
        f.write(str(scores[i])+" ")
    f.write("\n")
    f.write(str(np.mean(scores)))
    f.write("\n")
    f.write("---comment---\n")
    f.write(comment)
    f.write("\n")

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