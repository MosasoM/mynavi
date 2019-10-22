from agg import *
from dep import *
from indep import *
from nmf import *
from OH import *
from two_step import *

class tree_preprocess:
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


class linear_preprocess:
    def __init__(self):
        self.steps = [
            ("parse_area",parse_area_size()),
            ("parse_room",parse_rooms()),
            ("parse_old",parse_how_old()),
            ("height_enc",height_encoder()),
            ("ex_dist",extract_district()),
            ("label_dist",district_encoder()),
            ("dist2main_st",dist_to_main_station()),
            ("short_main_st",shortest2main_st()),
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

            ("m_d_p",add_mean_dist_price()),
            ("mean_walk",add_mean_walk_price()),
            ("mean_moyori",add_moyori_walk_price()),
            ("angle_stat",add_mean_angle_price()),
            ("mean_struct",add_mean_structure_price()),



            # ("dist_price_per_area",dist_and_price_per_area()),

            ("drop_for_lin",drop_for_linear()),
            ("drop_unnecessary",drop_unnecessary()),
            ("cross",cross_features()),


            ("dist_oh",district_onehot()),
            ("st_onehot",structure_onehot()),
            ("dir_oh",direction_onehot()),
        ]

