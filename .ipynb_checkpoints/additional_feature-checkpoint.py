
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