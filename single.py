import re
import numpy as np
import pandas as pd
import math

def area_size_(x):
    temp = x["面積"].values
    ans = [0 for i in range(len(temp))]
    for i in range(len(temp)):
        ans[i] = float(re.search(r"[0-9\.]+",temp[i])[0])
    return ans

def area_size_sq_(x):
    temp = x["面積"].values
    ans = [0 for i in range(len(temp))]
    for i in range(len(temp)):
        ans[i] = float(re.search(r"[0-9\.]+",temp[i])[0])
    return np.power(ans,2)


def rldks_(x):
    temp = x["間取り"].values
    room = [0 for i in range(len(temp))]
    head = ["L","D","K","S"]
    setubi = [[0 for i in range(len(temp))] for j in range(4)]
    for i in range(len(temp)):
        room[i] = int(temp[i][0])
        for j in range(4):
            if head[j] in temp[i]:
                setubi[j][i] = 1
    r = np.array(room)
    l = np.array(setubi[0])
    d = np.array(setubi[1])
    k = np.array(setubi[2])
    s = np.array(setubi[3])
    return r,l,d,k,s

def how_old_(x):
    temp = x["築年数"].values
    add_year = [0 for i in range(len(temp))]
    add_month = [0 for i in range(len(temp))]
    year_pat = re.compile(r"[0-9]+年")
    month_pat = re.compile(r"[0-9]+ヶ月")
    for i in range(len(temp)):
        year = year_pat.search(temp[i])
        month = month_pat.search(temp[i])
        if re.match(r"新築",temp[i]):
            year = -1
            month = 0
        else:
            if year:
                year = year[0][:-1]
            else:
                year = 3
            if month:
                month = month[0][:-2]
            else:
                month = 0
            if int(year) > 100:
                year = "15"
            add_year[i] = int(year)
            add_month[i] = int(month)
    return add_year,add_month

def height_of_it_(x):
    tmp = x["所在階"].values
    where = [0 for i in range(len(tmp))]
    what = [0 for i in range(len(tmp))]
    for i in range(len(tmp)):
        try:
            hoge =  tmp[i].split("／")
        except:
            hoge = ["2階","3階建て"]
        if len(hoge) == 2:
            if hoge[0] == "":
                hoge[0] = "2階"
            if hoge[1] == "":
                hoge[1] = "3階建て"
            x = int(re.search(r"[0-9]+",hoge[0])[0])
            y = int(re.search(r"[0-9]+",hoge[1])[0])
        else:
            x = 2
            y = 3
        where[i] = x
        what[i] = y
    return where,what

def address_of_it_(x):
    pat = re.compile(r"東京都.+区")
    p2 = re.compile(r"区.+?[０-９]丁目")
    p3 = re.compile(r"区.+?町")
    p4 = re.compile(r"区.+")
    dist = ["" for i in range(len(x.values))]
    area = ["" for i in range(len(x.values))]
    tmp = x["所在地"].values
    for i in range(len(tmp)):
        m = pat.search(tmp[i])
        dist[i] = m[0][3:]
        m = p2.search(tmp[i])
        if m:
            area[i] = m[0][1:]
        else:
            m = p3.search(tmp[i])
            if m:
                area[i] = m[0][1:]
            else:
                m = p4.search(tmp[i])
                if m:
                    area[i] = m[0][1:]
    return dist,area

def train_and_walk_(x):
    tmp = x["アクセス"].values
    train = ["" for i in range(len(tmp))]
    oth_train = [["" for i in range(len(tmp))] for j in range(2)]
    avgwalk = [0 for i in range(len(tmp))]
    oth_walk = [[30 for i in range(len(tmp))] for j in range(3)]
    for i in range(len(tmp)):
        train[i] = re.match(r".+?(線|ライン|ライナー|プレス|かもめ)",tmp[i])[0]
        avg = 0
        ind = 0
        for m in re.finditer(r"(徒歩|バス.*?)[0-9]+分",tmp[i]):
            if "バス" in m[0]:
                ind += 1
                if ind > 2:
                    break
                continue
            avg += int(m[0][2:-1])
            oth_walk[ind][i] = int(m[0][2:-1])
            ind += 1
            if ind > 2:
                break
        if ind == 0:
            ind = 1
        avg = avg/ind
        avgwalk[i] = avg
        ind = 0
        for m in re.finditer(r"\t\t.+?(線|ライン|ライナー|プレス|かもめ)",tmp[i]):
            oth_train[ind][i] = m[0][2:]
            ind += 1
            if ind > 1:
                break
    return train,oth_train,oth_walk,avgwalk


def make_freq_elem(x_column_not_value,elems):
    ans = {}
    for key in elems.keys():
        temp = x_column_not_value.str.contains(key).sum()
        ans[key] = temp
    return ans

def google_distance(lat1, lon1, lat2, lon2):
    radLat1 = math.radians(lat1) 
    radLon1 = math.radians(lon1) 
    radLat2 = math.radians(lat2) 
    radLon2 = math.radians(lon2)

    r = 6378137.0

    averageLat = (radLat1 - radLat2) / 2
    averageLon = (radLon1 - radLon2) / 2
    return r * 2 * math.asin(math.sqrt(pow(math.sin(averageLat), 2) + math.cos(radLat1) * math.cos(radLat2) * pow(math.sin(averageLon), 2)))
    
