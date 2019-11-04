from sklearn.pipeline import Pipeline
from preprocess_block import *
import xgboost as xgb
from catboost import CatBoostRegressor,CatBoostClassifier
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
import lightgbm as lgbm
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Lasso,Ridge,ElasticNet
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import scale
from sklearn.svm import SVR

import numpy as np
import random
import keras.backend as K
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, Activation,Dropout

import datetime

np.random.seed(seed=0)
session_conf = tf.ConfigProto(
    intra_op_parallelism_threads=1,
    inter_op_parallelism_threads=1
)

tf.set_random_seed(0)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

class my_preprocess:
    def __init__(self,seed):
        self.steps = [
            ("parse_area",parse_area_size()),
            ("parse_room",parse_rooms()),
            ("parse_old",parse_how_old()),
            ("height_enc",height_encoder()),
            ("ex_dist",extract_district()),
            ("label_dist",district_encoder()),
            ("acc_ext",access_extractor()),
            ("tr_enc",train_encoder()),
            ("dir_enc",direction_encoder()),
            ("info_enc",info_encoder()),
            ("p_con_time",parse_contract_time()),
            ("fac",fac_encoder()),
            ("bath",bath_encoder()),
            ("kit",kitchin_encoder()),
            ("env",env_encoder()),
            ("structure_enc",structure_label_encoder()),
            ("dist2main_st",dist_to_main_station()),
            ("short_main_st",shortest2main_st()),
            ("knn_tika1",knn_tika1()),
            ("actual_height",actual_height()),
            ("middle_class",middle_class_centers()),
            ("high_class",heigh_class_center()),
            ("middle_high",middle_high_centers()),

            ("m_d_p",add_mean_dist_price()),
            ("mean_struct",add_mean_structure_price()),
            ("mean_walk",add_mean_walk_price()),
            ("mean_moyori",add_moyori_walk_price()),
            ("angle_stat",add_mean_angle_price()),

            ("lda",lda()),

            ("drop_unnecessary",drop_unnecessary()),
            ("cross",cross_features()),

            ("NMF_train_walk",NMF_train_walk(seed)),
            ("NMF_fac",NMF_fac(seed)),
            
]

class my_linear:
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
            ("dir_enc",direction_encoder()),
            ("info_enc",info_encoder()),
            ("p_con_time",parse_contract_time()),
            ("fac",fac_encoder()),
            ("bath",bath_encoder()),
            ("kit",kitchin_encoder()),
            ("env",env_encoder()),
            ("structure_enc",structure_label_encoder()),
            ("dist2main_st",dist_to_main_station()),
            ("short_main_st",shortest2main_st()),
            ("knn_tika1",knn_tika1()),
            ("actual_height",actual_height()),
            ("middle_class",middle_class_centers()),
            ("high_class",heigh_class_center()),
            ("middle_high",middle_high_centers()),

            ("m_d_p",add_mean_dist_price()),
            ("mean_walk",add_mean_walk_price()),
            ("mean_moyori",add_moyori_walk_price()),
            ("angle_stat",add_mean_angle_price()),
            ("mean_struct",add_mean_structure_price()),

            ("drop_for_lin",drop_for_linear()),
            ("drop_unnecessary",drop_unnecessary()),
            ("cross",cross_features()),


            ("dist_oh",district_onehot()),
            ("st_onehot",structure_onehot()),
            ("dir_oh",direction_onehot()),
        ]


class my_model:
    def __init__(self,seed=7777):
        rp = my_preprocess(seed)
        rpstep = rp.steps
        rich_step_xgb = [
            ("pre",Pipeline(steps=rpstep)),
            ("xgb",xgb.XGBRegressor(random_state=seed,eta = 0.2,max_depth=9,min_child_weight=1,subsample=1.0))
        ]
        self.model = Pipeline(steps=rich_step_xgb)
    def fit(self,x,y):
        self.model.fit(x,y)
    def predict(self,x):
        pred = self.model.predict(x)
        return pred
    def get_params(self,deep=True):
        return {}

class linear_model:
    def __init__(self,seed=7777):
        rp = my_linear()
        # rp = my_preprocess(seed)
        rpstep = rp.steps
        rich_step_xgb = [
            ("pre",Pipeline(steps=rpstep)),
            # ("xgb",Ridge(max_iter=3000,alpha=0.005))
            ("dummy",dummy_scale()),
            ("svr",SVR(C=3.0,epsilon=0.1))
        ]
        self.model = Pipeline(steps=rich_step_xgb)
    def fit(self,x,y):
        self.model.fit(x,y)
    def predict(self,x):
        pred = self.model.predict(x)
        return pred
    def get_params(self,deep=True):
        return {}

class nn_model:
    def __init__(self):
        rp = my_linear()
        rpstep = rp.steps
        pp_steps = [
            ("pre",Pipeline(steps=rpstep)),
            ("dummy",dummy())
        ]
        self.preprocess = Pipeline(steps=pp_steps)
        ########
        self.model = Sequential()
        self.model.add(Dense(100))
        self.model.add(Activation("relu"))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(100))
        self.model.add(Activation("relu"))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(100))
        self.model.add(Activation("relu"))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(1))
        self.model.add(Activation("relu"))
        self.model.compile(optimizer="Adam",loss="mean_squared_error")
        #########
    def fit(self,x,y):
        self.preprocess.fit(x,y)
        data = self.preprocess.predict(x)
        data = scale(data)
        ############
        self.model.fit(data,y.values,epochs=100, batch_size=128,verbose=1)
        ###########
        return self
    def predict(self,x):
        data = self.preprocess.predict(x)
        data = scale(data)
        pred = self.model.predict(data)
        return pred

    def transform(self,x):
        data = self.preprocess.predict(x)
        data = scale(data)
        pred = self.model.predict(data)
        return x.assign(nn_pred=pred)
    
    def get_params(self,deep=True):
        return {}

def make_model(unit,layer):
    model = Sequential()
    for i in range(layer):
        model.add(Dense(unit))
        model.add(Activation("relu"))
        model.add(Dropout(0.2))
    model.add(Dense(1))
    model.add(Activation("relu"))
    model.compile(optimizer="Adam",loss="mean_squared_error")
    return model

class classify_model:
    def __init__(self,seed=7777):
        tp = tree_preprocess(seed)
        tp = tp.steps
        self.seed = seed
        self.tree_preprocess=Pipeline(steps=[
            ("pre",Pipeline(steps=tp)),
            ("dummy",dummy())
        ])
        self.rich_submodel = [
            xgb.XGBClassifier(random_state=seed,eta = 0.2,max_depth=9,min_child_weight=1,subsample=1.0),
            CatBoostClassifier(random_state=seed,logging_level="Silent"),
            RandomForestClassifier(random_state=seed),
            lgbm.LGBMClassifier(random_state=seed,learning_rate=0.2,min_child_weight=1,max_depth=9,n_estimators=500)
        ]
        self.poor_submodel = [
            xgb.XGBClassifier(random_state=seed,eta = 0.2,max_depth=9,min_child_weight=1,subsample=1.0),
            CatBoostClassifier(random_state=seed,logging_level="Silent"),
            RandomForestClassifier(random_state=seed),
            lgbm.LGBMClassifier(random_state=seed,learning_rate=0.2,min_child_weight=1,max_depth=9,n_estimators=500)
        ]
        self.very_rich_submodel = [
            xgb.XGBClassifier(random_state=seed,eta = 0.2,max_depth=9,min_child_weight=1,subsample=1.0),
            CatBoostClassifier(random_state=seed,logging_level="Silent"),
            RandomForestClassifier(random_state=seed),
            lgbm.LGBMClassifier(random_state=seed,learning_rate=0.2,min_child_weight=1,max_depth=9,n_estimators=500)
        ]
    def reset(self):
        self.rich_submodel = [
            xgb.XGBClassifier(random_state=self.seed,eta = 0.2,max_depth=9,min_child_weight=1,subsample=1.0),
            CatBoostClassifier(random_state=self.seed,logging_level="Silent"),
            RandomForestClassifier(random_state=self.seed),
            lgbm.LGBMClassifier(random_state=self.seed,learning_rate=0.2,min_child_weight=1,max_depth=9,n_estimators=500)
        ]
        self.poor_submodel = [
            xgb.XGBClassifier(random_state=self.seed,eta = 0.2,max_depth=9,min_child_weight=1,subsample=1.0),
            CatBoostClassifier(random_state=self.seed,logging_level="Silent"),
            RandomForestClassifier(random_state=self.seed),
            lgbm.LGBMClassifier(random_state=self.seed,learning_rate=0.2,min_child_weight=1,max_depth=9,n_estimators=500)
        ]
        self.very_rich_submodel = [
            xgb.XGBClassifier(random_state=self.seed,eta = 0.2,max_depth=9,min_child_weight=1,subsample=1.0),
            CatBoostClassifier(random_state=self.seed,logging_level="Silent"),
            RandomForestClassifier(random_state=self.seed),
            lgbm.LGBMClassifier(random_state=self.seed,learning_rate=0.2,min_child_weight=1,max_depth=9,n_estimators=500)
        ]

    def fit(self,x,y):
        kf = KFold(n_splits=10,shuffle=False)
        self.tree_preprocess.fit(x,y)
        buf = []
        for train_index, test_index in kf.split(x):
            x_train, x_test = x.iloc[train_index], x.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            tx = self.tree_preprocess.predict(x_train)
            temp = y_train.values
            rich_y = np.where(temp > 200000,1,0)
            vr_y = np.where(temp > 400000,1,0)
            poor_y = np.where(temp < 300000,1,0)



            for model in self.rich_submodel:
                model.fit(tx,rich_y)
            for model in self.poor_submodel:
                model.fit(tx,poor_y)
            for model in self.very_rich_submodel:
                model.fit(tx,vr_y)


            tx = self.tree_preprocess.predict(x_test)

            temp = []
            for model in self.rich_submodel:
                pred = model.predict(tx).tolist()
                temp.append(pred)
            for model in self.poor_submodel:
                pred = model.predict(tx).tolist()
                temp.append(pred)
            for model in self.poor_submodel:
                pred = model.predict(tx).tolist()
                temp.append(pred)
            temp = np.array(temp)
            buf.append(temp)
            self.reset()

        first_stage = np.concatenate(buf,axis=1)
        first_stage = first_stage.T
        np.savetxt(str(self.seed)+"_class_train.csv",first_stage)

        self.reset()
        tx = self.tree_preprocess.predict(x)
        temp = y.values
        rich_y = np.where(temp > 200000,1,0)
        vr_y = np.where(temp > 400000,1,0)
        poor_y = np.where(temp < 300000,1,0)
        
        for model in self.rich_submodel:
            model.fit(tx,rich_y)
        for model in self.poor_submodel:
            model.fit(tx,poor_y)
        for model in self.very_rich_submodel:
            model.fit(tx,vr_y)
        return self
    def predict(self,x):

        tx = self.tree_preprocess.predict(x)
        temp = []
        for model in self.rich_submodel:
            pred = model.predict(tx).tolist()
            temp.append(pred)
        for model in self.poor_submodel:
            pred = model.predict(tx).tolist()
            temp.append(pred)
        for model in self.poor_submodel:
            pred = model.predict(tx).tolist()
            temp.append(pred)
        temp = np.array(temp)
        temp = temp.T
        np.savetxt(str(self.seed)+"_class_test.csv",temp)
        return 0

    def get_params(self,deep=True):
        return {}


class stacking_model:
    def __init__(self,seed=7777):
        self.seed = seed
        self.main_model = Lasso()
        tp = tree_preprocess(seed)
        tp = tp.steps
        lp = linear_preprocess()
        lp = lp.steps
        
        self.tree_preprocess=Pipeline(steps=[
            ("pre",Pipeline(steps=tp)),
            ("dummy",dummy())
        ])
        self.linear_preprocess = Pipeline(steps=[
            ("pre",Pipeline(steps=lp)),
            ("dummy",dummy())
        ])
        self.adval_preprocess = Pipeline(steps=[
            ("pre",Pipeline(steps=lp)),
            ("dummy",dummy())
        ])
        self.tree_submodel = [
            xgb.XGBRegressor(random_state=seed,eta = 0.2,max_depth=9,min_child_weight=1,subsample=1.0),
            # CatBoostRegressor(random_state=seed,logging_level="Silent"),
            RandomForestRegressor(random_state=seed),
            lgbm.LGBMRegressor(random_state=seed,learning_rate=0.2,min_child_weight=1,max_depth=9,n_estimators=500)
        ]
        self.rich_submodel = [
            # xgb.XGBRegressor(random_state=seed,eta = 0.2,max_depth=9,min_child_weight=1,subsample=1.0),
            # CatBoostRegressor(random_state=seed,logging_level="Silent"),
            RandomForestRegressor(random_state=seed),
            lgbm.LGBMRegressor(random_state=seed,learning_rate=0.2,min_child_weight=1,max_depth=9,n_estimators=500)
        ]
        self.poor_submodel = [
            # xgb.XGBRegressor(random_state=seed,eta = 0.2,max_depth=9,min_child_weight=1,subsample=1.0),
            # CatBoostRegressor(random_state=seed,logging_level="Silent"),
            # RandomForestRegressor(random_state=seed),
            # lgbm.LGBMRegressor(random_state=seed,learning_rate=0.2,min_child_weight=1,max_depth=9,n_estimators=500)
        ]
        self.area_pre_price_submodel = [
            # xgb.XGBRegressor(random_state=seed,eta = 0.2,max_depth=9,min_child_weight=1,subsample=1.0),
            # CatBoostRegressor(random_state=seed,logging_level="Silent"),
            RandomForestRegressor(random_state=seed),
            # lgbm.LGBMRegressor(random_state=seed,learning_rate=0.2,min_child_weight=1,max_depth=9,n_estimators=500)
        ]

        self.linear_submodel = [
            # Ridge(random_state=seed),16
            Lasso(random_state=seed),
        ]

        self.knn_submodels = [
            # KNeighborsRegressor(n_neighbors=30,weights="distance"),
            # KNeighborsRegressor(n_neighbors=10,weights="distance"),
            KNeighborsRegressor(n_neighbors=5,weights="distance"),
            # KNeighborsRegressor(n_neighbors=3),
            # KNeighborsRegressor(n_neighbors=10),22
        ]

        # self.knn_app = [
        #     KNeighborsRegressor(n_neighbors=30,weights="distance"),
        #     KNeighborsRegressor(n_neighbors=10,weights="distance"),
        #     KNeighborsRegressor(n_neighbors=5,weights="distance"),
        #     KNeighborsRegressor(n_neighbors=3),
        #     KNeighborsRegressor(n_neighbors=10),
        # ]

        self.nn_submodel = [
            # make_model(100,3),
            # make_model(100,4),
            # make_model(300,3),
            make_model(300,4),
        ]
        print(self.nn_submodel)

    def reset(self):
        self.tree_submodel = [
            xgb.XGBRegressor(random_state=self.seed,eta = 0.2,max_depth=9,min_child_weight=1,subsample=1.0),
            # CatBoostRegressor(random_state=self.seed,logging_level="Silent"),
            RandomForestRegressor(random_state=self.seed),
            lgbm.LGBMRegressor(random_state=self.seed,learning_rate=0.2,min_child_weight=1,max_depth=9,n_estimators=500)
        ]
        self.rich_submodel = [
            # xgb.XGBRegressor(random_state=self.seed,eta = 0.2,max_depth=9,min_child_weight=1,subsample=1.0),
            # CatBoostRegressor(random_state=self.seed,logging_level="Silent"),
            RandomForestRegressor(random_state=self.seed),
            lgbm.LGBMRegressor(random_state=self.seed,learning_rate=0.2,min_child_weight=1,max_depth=9,n_estimators=500)
        ]
        # self.poor_submodel = [
        #     # xgb.XGBRegressor(random_state=self.seed,eta = 0.2,max_depth=9,min_child_weight=1,subsample=1.0),
        #     # CatBoostRegressor(random_state=self.seed,logging_level="Silent"),
        #     # RandomForestRegressor(random_state=self.seed),
        #     # lgbm.LGBMRegressor(random_state=self.seed,learning_rate=0.2,min_child_weight=1,max_depth=9,n_estimators=500)
        # ]
        self.area_pre_price_submodel = [
            # xgb.XGBRegressor(random_state=self.seed,eta = 0.2,max_depth=9,min_child_weight=1,subsample=1.0),
            # CatBoostRegressor(random_state=self.seed,logging_level="Silent"),
            RandomForestRegressor(random_state=self.seed),
            # lgbm.LGBMRegressor(random_state=self.seed,learning_rate=0.2,min_child_weight=1,max_depth=9,n_estimators=500)
        ]
        self.linear_submodel = [
            # Ridge(random_state=self.seed),
            Lasso(random_state=self.seed),
        ]
        self.knn_submodels = [
            # KNeighborsRegressor(n_neighbors=30,weights="distance"),
            # KNeighborsRegressor(n_neighbors=10,weights="distance"),
            KNeighborsRegressor(n_neighbors=5,weights="distance"),
            # KNeighborsRegressor(n_neighbors=3),
            # KNeighborsRegressor(n_neighbors=10),
        ]
        # self.knn_app = [
        #     KNeighborsRegressor(n_neighbors=30,weights="distance"),
        #     KNeighborsRegressor(n_neighbors=10,weights="distance"),
        #     KNeighborsRegressor(n_neighbors=5,weights="distance"),
        #     KNeighborsRegressor(n_neighbors=3),
        #     KNeighborsRegressor(n_neighbors=10),
        # ]
        self.nn_submodel = [
            # make_model(100,3),
            # make_model(100,4),
            # make_model(300,3),
            make_model(300,4),
        ]

    def fit(self,x,y):
        kf = KFold(n_splits=10,shuffle=False)
        buf = []
        self.tree_preprocess.fit(x,y)
        self.linear_preprocess.fit(x,y)
        print("preprocess_fit")
        print(datetime.datetime.now())
        for train_index, test_index in kf.split(x):
            x_train, x_test = x.iloc[train_index], x.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            tx = self.tree_preprocess.predict(x_train)
            lx = self.linear_preprocess.predict(x_train)
            sc_lx = scale(lx)

            hoge = pd.DataFrame(y_train.values)
            hoge.index = tx.index
            hoge.columns=["price_ff"]
            hoge = pd.concat([tx,hoge],axis=1)

            rich = hoge.query("price_ff > 200000")
            poor = hoge.query("price_ff <= 300000")

            rich_x = rich.drop("price_ff",axis=1)
            rich_y = rich["price_ff"]

            poor_x = poor.drop("price_ff",axis=1)
            poor_y = poor["price_ff"]

            knn_x = x_train[["ido","keido"]].values

            app_y = y_train.values/tx["mf_areasize"].values

            for model in self.tree_submodel:
                model.fit(tx,y_train)
            for model in self.linear_submodel:
                model.fit(lx,y_train)
            for model in self.rich_submodel:
                model.fit(rich_x,rich_y)
            # for model in self.poor_submodel:
            #     model.fit(poor_x,poor_y)
            for model in self.area_pre_price_submodel:
                model.fit(tx,app_y)
            print("trees_fit")
            print(datetime.datetime.now())
            for model in self.knn_submodels:
                model.fit(knn_x,y_train)
            # for model in self.knn_app:
            #     model.fit(knn_x,app_y)
            print("knn_fit")
            print(datetime.datetime.now())
            for model in self.nn_submodel:
                model.fit(sc_lx,y_train.values,epochs=100, batch_size=32,verbose=0)
            print("NN_fit")
            print(datetime.datetime.now())

            tx = self.tree_preprocess.predict(x_test)
            lx = self.linear_preprocess.predict(x_test)
            sc_lx = scale(lx)

            knn_x = x_test[["ido","keido"]].values

            temp = []
            for model in self.tree_submodel:
                pred = model.predict(tx).tolist()
                temp.append(pred)
            for model in self.linear_submodel:
                pred = model.predict(lx).tolist()
                temp.append(pred)
            for model in self.rich_submodel:
                pred = model.predict(tx).tolist()
                temp.append(pred)
            # for model in self.poor_submodel:
            #     pred = model.predict(tx).tolist()
            #     temp.append(pred)
            for model in self.area_pre_price_submodel:
                pred = model.predict(tx)
                pred = pred*tx["mf_areasize"].values
                temp.append(pred)
            print("trees_pred")
            print(datetime.datetime.now())
            for model in self.knn_submodels:
                pred = model.predict(knn_x).tolist()
                temp.append(pred)
            # for model in self.knn_app:
            #     pred = model.predict(knn_x).tolist()
            #     pred = pred*tx["mf_areasize"].values
            #     temp.append(pred)
            print("knn_pred")
            print(datetime.datetime.now())
            for model in self.nn_submodel:
                pred = model.predict(sc_lx)
                pred = pred.flatten().tolist()
                temp.append(pred)
            print("NN_pred")
            print(datetime.datetime.now())
            temp = np.array(temp)
            buf.append(temp)
            self.reset()
        first_stage = np.concatenate(buf,axis=1)
        first_stage = first_stage.T
        np.savetxt(str(self.seed)+"train.csv",first_stage)
        print("save train")
        print(datetime.datetime.now())
        self.main_model.fit(first_stage,y)
#############################################
        self.reset()
        tx = self.tree_preprocess.predict(x)
        lx = self.linear_preprocess.predict(x)
        sc_lx = scale(lx)

        hoge = pd.DataFrame(y.values)
        hoge.index = tx.index
        hoge.columns=["price_ff"]
        hoge = pd.concat([tx,hoge],axis=1)

        rich = hoge.query("price_ff > 200000")
        poor = hoge.query("price_ff <= 300000")

        rich_x = rich.drop("price_ff",axis=1)
        rich_y = rich["price_ff"]

        poor_x = poor.drop("price_ff",axis=1)
        poor_y = poor["price_ff"]

        knn_x = x[["ido","keido"]].values

        app_y = y.values/tx["mf_areasize"].values

        for model in self.tree_submodel:
            model.fit(tx,y)
        for model in self.linear_submodel:
            model.fit(lx,y)
        for model in self.rich_submodel:
            model.fit(rich_x,rich_y)
        # for model in self.poor_submodel:
        #     model.fit(poor_x,poor_y)
        for model in self.area_pre_price_submodel:
            model.fit(tx,app_y)
        for model in self.knn_submodels:
            model.fit(knn_x,y)
        # for model in self.knn_app:
        #     model.fit(knn_x,app_y)
        for model in self.nn_submodel:
            model.fit(sc_lx,y.values,epochs=150, batch_size=32,verbose=0)
        print("fit end")
        print(datetime.datetime.now())
        return self

    def predict(self,x):

        tx = self.tree_preprocess.predict(x)
        lx = self.linear_preprocess.predict(x)
        sc_lx = scale(lx)
        knn_x = x[["ido","keido"]].values

        temp = []
        for model in self.tree_submodel:
            pred = model.predict(tx).tolist()
            temp.append(pred)
        for model in self.linear_submodel:
            pred = model.predict(lx).tolist()
            temp.append(pred)
        for model in self.rich_submodel:
            pred = model.predict(tx).tolist()
            temp.append(pred)
        # for model in self.poor_submodel:
        #     pred = model.predict(tx).tolist()
        #     temp.append(pred)
        for model in self.area_pre_price_submodel:
            pred = model.predict(tx)
            pred = pred*tx["mf_areasize"].values
            temp.append(pred)
        for model in self.knn_submodels:
            pred = model.predict(knn_x).tolist()
            temp.append(pred)
        # for model in self.knn_app:
        #     pred = model.predict(knn_x).tolist()
        #     pred = pred*tx["mf_areasize"].values
        #     temp.append(pred)
        for model in self.nn_submodel:
            pred = model.predict(sc_lx)
            pred = pred.flatten().tolist()
            temp.append(pred)
        temp = np.array(temp)
        temp = temp.T
        np.savetxt(str(self.seed)+"test.csv",temp)
        return self.main_model.predict(temp)

    def get_params(self,deep=True):
        return {}




