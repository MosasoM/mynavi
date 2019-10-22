from sklearn.pipeline import Pipeline
from preprocess_block import *
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor 
import lightgbm as lgbm
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Lasso,Ridge,ElasticNet
import pandas as pd
from sklearn.model_selection import KFold

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
            # ("parking_encoder",parking_encoder()),
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
            ("knn_tika2",knn_tika2()),
            ("actual_height",actual_height()),
            ("middle_class",middle_class_centers()),
            ("high_class",heigh_class_center()),

            ("m_d_p",add_mean_dist_price()),
            ("mean_struct",add_mean_structure_price()),
            ("mean_walk",add_mean_walk_price()),
            ("mean_moyori",add_moyori_walk_price()),
            ("angle_stat",add_mean_angle_price()),

            ("drop_unnecessary",drop_unnecessary()),
            ("cross",cross_features()),

            ("NMF_train_walk",NMF_train_walk(seed)),
            ("NMF_fac",NMF_fac(seed)),
            ("NMF_kit",NMF_kit(seed)),
            ("NMF_env_dist",NMF_env_dist(seed)),
            ("NMF_env",NMF_env(seed)),


            # ("area_predictor",area_pre_predictor(seed)),
            # ("area_pre_price_predictor",area_per_price_predictor(seed)),
            # ("knn_pred",Knn_regression()),
            # ("dist_price_per_area",dist_and_price_per_area()),
            # ("only_rich",only_rich_model(seed)),
            # ("pre_cat",pre_predictor(seed))
]


class my_model:
    def __init__(self,seed=7777):
        rp = my_preprocess(seed)
        rpstep = rp.steps
        rich_step_xgb = [
            ("pre",Pipeline(steps=rpstep)),
            ("xgb",xgb.XGBRegressor(random_state=seed,max_depth=8))
        ]
        self.model = Pipeline(steps=rich_step_xgb)
    def fit(self,x,y):
        self.model.fit(x,y)
    def predict(self,x):
        pred = self.model.predict(x)
        return pred
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
        self.tree_submodel = [
            xgb.XGBRegressor(random_state=seed),
            CatBoostRegressor(random_state=seed,logging_level="Silent"),
            RandomForestRegressor(random_state=seed),
            lgbm.LGBMRegressor(random_state=seed)
        ]
        self.rich_submodel = [
            xgb.XGBRegressor(random_state=seed),
            CatBoostRegressor(random_state=seed,logging_level="Silent"),
            RandomForestRegressor(random_state=seed),
            lgbm.LGBMRegressor(random_state=seed)
        ]
        self.poor_submodel = [
            xgb.XGBRegressor(random_state=seed),
            CatBoostRegressor(random_state=seed,logging_level="Silent"),
            RandomForestRegressor(random_state=seed),
            lgbm.LGBMRegressor(random_state=seed)
        ]
        self.area_pre_price_submodel = [
            xgb.XGBRegressor(random_state=seed),
            CatBoostRegressor(random_state=seed,logging_level="Silent"),
            RandomForestRegressor(random_state=seed),
            lgbm.LGBMRegressor(random_state=seed)
        ]
        self.linear_submodel = [
            Ridge(),
            Lasso(),
        ]
        self.knn_submodel=KNeighborsRegressor(n_neighbors=30,weights="distance")

    def reset(self):
        self.tree_submodel = [
            xgb.XGBRegressor(random_state=self.seed),
            CatBoostRegressor(random_state=self.seed,logging_level="Silent"),
            RandomForestRegressor(random_state=self.seed),
            lgbm.LGBMRegressor(random_state=self.seed)
        ]
        self.rich_submodel = [
            xgb.XGBRegressor(random_state=self.seed),
            CatBoostRegressor(random_state=self.seed,logging_level="Silent"),
            RandomForestRegressor(random_state=self.seed),
            lgbm.LGBMRegressor(random_state=self.seed)
        ]
        self.poor_submodel = [
            xgb.XGBRegressor(random_state=self.seed),
            CatBoostRegressor(random_state=self.seed,logging_level="Silent"),
            RandomForestRegressor(random_state=self.seed),
            lgbm.LGBMRegressor(random_state=self.seed)
        ]
        self.area_pre_price_submodel = [
            xgb.XGBRegressor(random_state=self.seed),
            CatBoostRegressor(random_state=self.seed,logging_level="Silent"),
            RandomForestRegressor(random_state=self.seed),
            lgbm.LGBMRegressor(random_state=self.seed)
        ]
        self.linear_submodel = [
            Ridge(),
            Lasso(),
        ]
        self.knn_submodel=KNeighborsRegressor(n_neighbors=30,weights="distance")

    def fit(self,x,y):
        kf = KFold(n_splits=3,shuffle=False)
        buf = []
        self.tree_preprocess.fit(x,y)
        self.linear_preprocess.fit(x,y)
        for train_index, test_index in kf.split(x):
            x_train, x_test = x.iloc[train_index], x.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            tx = self.tree_preprocess.predict(x_train)
            lx = self.linear_preprocess.predict(x_train)

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
            for model in self.poor_submodel:
                model.fit(poor_x,poor_y)
            for model in self.area_pre_price_submodel:
                model.fit(tx,app_y)
            self.knn_submodel.fit(knn_x,y_train)


            tx = self.tree_preprocess.predict(x_test)
            lx = self.linear_preprocess.predict(x_test)

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
            for model in self.poor_submodel:
                pred = model.predict(tx).tolist()
                temp.append(pred)
            for model in self.area_pre_price_submodel:
                pred = model.predict(tx)
                pred = pred*tx["mf_areasize"].values
                temp.append(pred)
            pred = self.knn_submodel.predict(knn_x).tolist()
            temp.append(pred)
            temp = np.array(temp)
            buf.append(temp)
        first_stage = np.concatenate(buf,axis=1)
        first_stage = first_stage.T
        self.main_model.fit(first_stage,y)
#############################################
        self.reset()
        tx = self.tree_preprocess.predict(x)
        lx = self.linear_preprocess.predict(x)

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
        for model in self.poor_submodel:
            model.fit(poor_x,poor_y)
        for model in self.area_pre_price_submodel:
            model.fit(tx,app_y)
        
        self.knn_submodel.fit(knn_x,y)
        return self

    def predict(self,x):

        tx = self.tree_preprocess.predict(x)
        lx = self.linear_preprocess.predict(x)
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
        for model in self.poor_submodel:
            pred = model.predict(tx).tolist()
            temp.append(pred)
        for model in self.area_pre_price_submodel:
            pred = model.predict(tx)
            pred = pred*tx["mf_areasize"].values
            temp.append(pred)
        pred = self.knn_submodel.predict(knn_x).tolist()
        temp.append(pred)
        temp = np.array(temp)
        temp = temp.T
        return self.main_model.predict(temp)

    def get_params(self,deep=True):
        return {}
    
