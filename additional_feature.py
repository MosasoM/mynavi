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
            ("tr_enc",train_encoder()),
            ("parking_encoder",parking_encoder()),
            ("dir_enc",direction_encoder()),#なんか精度低下した
            ("info_enc",info_encoder()),
            ("cross",cross_features()),
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
            ("tr_enc",train_encoder()),
            ("parking_encoder",parking_encoder()),
            ("dir_enc",direction_encoder()),#なんか精度低下したがバグのせいだったかも
            ("info_enc",info_encoder()),
            ("m_d_p",add_mean_dist_price()),
            ("p_con_time",parse_contract_time()),
            ("fac",fac_encoder()),
            ("bath",bath_encoder()),
            ("kit",kitchin_encoder()),
            ("env",env_encoder()),
#             ("m_c_p",add_mean_city_price()),精度低下
#             ("mdp",add_mean_dir_price()),
            ("cross",cross_features()),
            ("drop_unnecessary",drop_unnecessary())
        ]

class add_mean_city_price: #くごとの家賃平均を追加。分散とか足してもいいかも
    def __init__(self):
        self.means = {}
        self.mm = 124000
    def fit(self,x,y):
        ty = pd.DataFrame(y)
        ty.columns=["賃料"]
        ty.index = x.index
        temptemp = pd.concat([x,ty],axis = 1)
        temp = np.round(temptemp.groupby("city").mean()["賃料"].values)
        label = temptemp.groupby("city").mean().index.values
        for i in range(len(label)):
            self.means[label[i]] = temp[i]
        self.mm = round(np.mean(temp))
        return self
    def transform(self,x):
        buf = [0 for i in range(len(x.values))]
        temp = x["city"].values
        for i in range(len(x.values)):
            if temp[i] in self.means:
                buf[i] = self.means[temp[i]]
            else:
                buf[i] = self.mm
        hoge = x.assign(c_p_mean =buf)
        return hoge

          


class dummy:
    def __init__(self):
        pass
    def fit(self,x,y):
        return self
    def predict(self,x):
        return x