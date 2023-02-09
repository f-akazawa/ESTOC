#!/usr/bin/env python
# coding: utf-8

# この時点で必要なデータ\n",
#####\n",
# 10日平均の9ファイル\n",
# shapeは(36,94,192) 1ヶ月を10日区切りで\n",
# dlwrf10dy.dat , dswrf10dy.dat(0列入ってる) , lhtfl10dy.dat(649728)\n",
# prate10dy.dat > デカすぎでmemory errorになる\n",
# shtfl10dy.dat , uflx10dy.dat\n",
# ulwrf10dy.dat , uswrf10dy.dat(0列入ってる） , vflx10dy.dat \n",
#####\n",
# land.06.ft.bigファイルを読み込んだ land (94,192)\n",
# prev_climate7902.nc2.bigを読み込んだ climate (12,94,192)\n",
#####\n",
# 元ソースは94 x 192 > 178 x 360に補完している"


import xarray as xr
import numpy as np
from scipy import interpolate
import pandas as pd
import datetime
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.util as cutil
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from scipy.interpolate import griddata
import gc

# 緯度経度は実際の数値を入れる必要がある\n",
# latitudeはStart-89.5、End89.5、Step数で元データの94分割になるように調整\n",
#orig_lat = np.arange(-89.5,89.5,1.905)
# 2022/1/26土居さんより依頼でStart/End-90,90に変更＞内挿後の値が180になるようにするため
orig_lat = np.arange(-90,90,1.915)

# longitudeはStart-180,End180,Stepで192分割に調整\n",
orig_lon = np.arange(-180,180,1.875)

# mgridは最初、最後、間隔を指定するので間隔1のグリッドが出来る、返り値がmeshgridと縦横は逆になるので注意\n",
# lat,lonは実際の座標値を使い、92*194 > 178*360に変更する（Stepは1）\n",
#xi,yi = np.mgrid[-180:180:1,-89.5:89.5:1]
xi,yi = np.mgrid[-180:180:1,-90:90:1]


### Fresh water fluxの計算

landft06 = '01_ESTOC_ForcingData/NCEP_NCAR_Forcing/2017/4.fwat/inc/land.ft06.big' # pythonの場合reshape(94,192)で読み込む\n",
prevclimate = '01_ESTOC_ForcingData/NCEP_NCAR_Forcing/2017/4.fwat/inc/prev_climate7902nc2.big'

# land.06.ft.big読み込み\n",
# 01_ESTOC_ForcingData/NCEP_NCAR_Forcing/2017/4.fwat/inc/land.ft06.big\n",
# 縦 x 横(94 X 192)の世界地図イメージ　0/1で表現されている\n",
# python で読むと0と32831が94*192の配列に入っている。\n",
# 0が海、1(32831)は陸\n",
# landmask用に必要
iland = np.fromfile(landft06,dtype='int32').reshape(94,192)

# prev.climate7902.nc2.big読み込み\n",
# 01_ESTOC_ForcingData/NCEP_NCAR_Forcing/2017/4.fwat/inc/prev_climate7902nc2.big\n",
# 縦ｘ横(94 x 192)が12枚あるイメージ\n",
# 12ヶ月分の何かしらの気候値\n",
clima = np.fromfile(prevclimate,dtype='float32').reshape(12,94,192)

#prate10dy.dat\n",
prate = np.fromfile('prate10dy.dat').reshape(2664,94,192)
# lthfl\n",
lhtfl = np.fromfile('lhtfl10dy.dat').reshape(2664,94,192)


# 内挿する前にilandとclima = freshを使ってマスクする必要あり\n",
# prate と lhtflで要素ごとに計算\n",
# 以下の数式とfreshの命名規則はFotran元コードから\n",
fresh = prate - lhtfl/2.5e6


# ilandで陸地だったらfreshの同位置を０にする\n",
# bool indexで配列(landmask)が返る\n",
landmask = (iland >= 1)
# True の部分（地上）を０にする\n",
index = 0
while index < 2664:
    fresh[index,:,:][landmask == True] = 0
    index += 1


resmax = 0.0

landindex =  np.where(landmask == True)
# landindexは陸地(landmask==True)の座標がタプルのndarray(配列）で返る
# landindex[0] = latitude
# landindex[1] = longitude

for index in range(36): # ループが遅いのでテストで1年分だけ出してみる、本当は74年分で2664
    for counter in range(1000):# 1000回繰り返す（又は閾値を超えたらループ終了
        for i,lat in enumerate(landindex[0]): # i = loop index
            lon = landindex[1][i]
            
            if lat > 1 and lat < 93 and lon < 191: # 画面端は処理できないので除外
                # lat,lon(landindex[1][i])が陸地なので4点計算する
                res = abs(fresh[index,lat,lon]-0.25*(fresh[index,lat-1,lon]+fresh[index,lat+1,lon]+fresh[index,lat,lon-1]+fresh[index,lat,lon+1]))
                fresh[index,lat,lon] = 0.25*(fresh[index,lat-1,lon]+fresh[index,lat+1,lon]+fresh[index,lat,lon-1]+fresh[index,lat,lon+1])
                resmax = max(resmax,res)

            # この時点でのlat,lon(landindex[1][i])が陸地なので4点計算する
            # 画面端の処理
            if lat == 0:
                for lon in range(192-1):
                    fresh[191,lat,lon] = fresh[0,lat,lon]
                    if lon == 190: # 0スタートなので1少ない
                        fresh[0,lat,lon] = fresh[191,lat,lon]

            # 差分が範囲超えたら抜ける
            if 0.0000001 > resmax:
                break
            else: # fresh > res にコピーしないとだめ
                res = fresh
        else:
            continue
        break
    else:
        continue
    break
           

# 内挿する
readdata = xr.DataArray(fresh,dims=['time','lat','lon']) # ファイルを読み込んでXArrayに入れる\n",
# dim_0 10日平均 dim_1 lat緯度 dim_2 lon経度\n",

days = 0
temp = []

while days < 2664:
    slicedays = readdata[days,:,:]
    zi = interpolate.interp2d(orig_lon,orig_lat,slicedays,kind='linear')(xi[:,0],yi[0,:])

    hozon = np.append(temp,zi).reshape([days+1,180,360])

    temp = hozon

    days += 1

hozon.tofile('fwflux10dy-1948-1000times.dat')

#ガベージコレクタ\n",
del temp,hozon
gc.collect()


### ここまでのループでうまく行ったら、ここより下のWhileループを修正してforループにする
### PythonではWhileよりforループの方が一般的とのこと。

### Momentum fluxの計算
### uflux10dy.dat vflux10dy.dataを使う


# uflx
uflx = np.fromfile('uflx10dy.dat').reshape(2664,94,192)
# vflx
vflx = np.fromfile('vflx10dy.dat').reshape(2664,94,192)

# landmask
# fresh water では計算式があったが、momentum fluxでは無い
uflux = uflx 
vflux = vflx

# ilandで陸地だったらfreshの同位置を０にする\n",
# bool indexで配列(landmask)が返る\n",
landmask = (iland >= 1)
# True の部分（地上）を０にする\n",
index = 0
while index < 2664:
    uflux[index,:,:][landmask == True] = 0
    vflux[index,:,:][landmask == True] = 0
    index += 1


resmax = 0.0

landindex =  np.where(landmask == True)
# landindexは陸地(landmask==True)の座標がタプルのndarray(配列）で返る
# landindex[0] = latitude
# landindex[1] = longitude
counter = 0

while True:
    for index in range(36): # origin 2664
        for i,lat in enumerate(landindex[0]): # i = loop index
            lon = landindex[1][i]
            
            if lat > 1 and lat < 93 and lon < 191: # 画面端は処理できないので除外
            # lat,lon(landindex[1][i])が陸地なので4点計算する
                res = abs(vflx[index,lat,lon]-0.25*(vflx[index,lat-1,lon]+vflx[index,lat+1,lon]+vflx[index,lat,lon-1]+vflx[index,lat,lon+1]))
                vflx[index,lat,lon] = 0.25*(vflx[index,lat-1,lon]+vflx[index,lat+1,lon]+vflx[index,lat,lon-1]+vflx[index,lat,lon+1])
                resmax = max(resmax,res)
                # この時点でのlat,lon(landindex[1][i])が陸地なので4点計算する
        # 画面端の処理
        if lat == 0:
            for lon in range(192-1):
                vflx[191,lat,lon] = vflx[0,lat,lon]
                if lon == 190: # 0スタートなので1少ない
                    vflx[0,lat,lon] = vflx[191,lat,lon]
        # lat,lon(landindex[1][i])が陸地なので4点計算する
            res = abs(vflx[index,lat,lon]-0.25*(vflx[index,lat-1,lon]+vflx[index,lat+1,lon]+vflx[index,lat,lon-1]+vflx[index,lat,lon+1]))
            vflx[index,lat,lon] = 0.25*(vflx[index,lat-1,lon]+vflx[index,lat+1,lon]+vflx[index,lat,lon-1]+vflx[index,lat,lon+1])
            resmax = max(resmax,res)

# 差分が範囲超えるまで、又は1000回ループしたら抜ける     
    if 0.0000001 > resmax or counter > 1000:
        break
    else: # vflx > res にコピーしないとだめ
        res = vflx
    
    counter += 1

# 内挿する
readdata = xr.DataArray(vflx,dims=['time','lat','lon']) # ファイルを読み込んでXArrayに入れる\n",
# dim_0 10日平均 dim_1 lat緯度 dim_2 lon経度\n",

days = 0
temp = []

while days < 2664:
    slicedays = readdata[days,:,:]
    zi = interpolate.interp2d(orig_lon,orig_lat,slicedays,kind='linear')(xi[:,0],yi[0,:])

    hozon = np.append(temp,zi).reshape([days+1,180,360])

    temp = hozon

    days += 1

hozon.tofile('nc1ex1deg.vflx10dy.1948-2021.dat')

#ガベージコレクタ\n",
del temp,hozon
gc.collect()

######

resmax = 0.0

landindex =  np.where(landmask == True)
# landindexは陸地(landmask==True)の座標がタプルのndarray(配列）で返る
# landindex[0] = latitude
# landindex[1] = longitude
counter = 0

while True:
    for index in range(2664):
        for i,lat in enumerate(landindex[0]): # i = loop index
            lon = landindex[1][i]
            # この時点でのlat,lon(landindex[1][i])が陸地なので4点計算する
        # 画面端の処理
        if lat == 0:
            for lon in range(192-1):
                uflx[191,lat,lon] = uflx[0,lat,lon]
                if lon == 190: # 0スタートなので1少ない
                    uflx[0,lat,lon] = uflx[191,lat,lon]
        # lat,lon(landindex[1][i])が陸地なので4点計算する
            res = abs(uflx[index,lat,lon]-0.25*(uflx[index,lat-1,lon]+uflx[index,lat+1,lon]+uflx[index,lat,lon-1]+uflx[index,lat,lon+1]))
            uflx[index,lat,lon] = 0.25*(uflx[index,lat-1,lon]+uflx[index,lat+1,lon]+uflx[index,lat,lon-1]+uflx[index,lat,lon+1])
            resmax = max(resmax,res)

# 差分が範囲超えるまで、又は1000回ループしたら抜ける     
    if 0.0000001 > resmax or counter > 1000:
        break
    else: # uflx > res にコピーしないとだめ
        res = uflx
    
    counter += 1         



# 内挿する
readdata = xr.DataArray(uflx,dims=['time','lat','lon']) # ファイルを読み込んでXArrayに入れる\n",
# dim_0 10日平均 dim_1 lat緯度 dim_2 lon経度\n",

days = 0
temp = []

while days < 2664:
    slicedays = readdata[days,:,:]
    zi = interpolate.interp2d(orig_lon,orig_lat,slicedays,kind='linear')(xi[:,0],yi[0,:])

    hozon = np.append(temp,zi).reshape([days+1,180,360])

    temp = hozon

    days += 1

hozon.tofile('nc1ex1deg.uflx10dy.1948-2021.dat')

#ガベージコレクタ\n",
del temp,hozon
gc.collect()

### Net heat fluxの計算


# dswrf
dsr = np.fromfile('dswrf10dy.dat').reshape(2664,94,192)
# ulwrf
usr = np.fromfile('uswrf10dy.dat').reshape(2664,94,192)
# dlwrf
dlr = np.fromfile('dlwrf10dy.dat').reshape(2664,94,192)
# ulwrf
ulr = np.fromfile('ulwrf10dy.dat').reshape(2664,94,192)
# shtfl
sh = np.fromfile('shtfl10dy.dat').reshape(2664,94,192)
# lhtfl
lh = np.fromfile('lhtfl10dy.dat').reshape(2664,94,192)

# landmask
# Net heat fluxは以下の式
gh = (dsr - usr + dlr - ulr) - (sh + lh)


# ilandで陸地だったらfreshの同位置を０にする\n",
# bool indexで配列(landmask)が返る\n",
landmask = (iland >= 1)
# True の部分（地上）を０にする\n",
index = 0
while index < 2664:
    gh[index,:,:][landmask == True] = 0
    index += 1


resmax = 0.0

landindex =  np.where(landmask == True)
# landindexは陸地(landmask==True)の座標がタプルのndarray(配列）で返る
# landindex[0] = latitude
# landindex[1] = longitude
counter = 0

while True:
    for index in range(2664):
        for i,lat in enumerate(landindex[0]): # i = loop index
            lon = landindex[1][i]
            # この時点でのlat,lon(landindex[1][i])が陸地なので4点計算する
        # 画面端の処理
        if lat == 0:
            for lon in range(192-1):
                gh[191,lat,lon] = gh[0,lat,lon]
                if lon == 190: # 0スタートなので1少ない
                    gh[0,lat,lon] = gh[191,lat,lon]
        # lat,lon(landindex[1][i])が陸地なので4点計算する
            res = abs(gh[index,lat,lon]-0.25*(gh[index,lat-1,lon]+gh[index,lat+1,lon]+gh[index,lat,lon-1]+gh[index,lat,lon+1]))
            gh[index,lat,lon] = 0.25*(gh[index,lat-1,lon]+gh[index,lat+1,lon]+gh[index,lat,lon-1]+gh[index,lat,lon+1])
            resmax = max(resmax,res)

# 差分が範囲超えるまで、又は1000回ループしたら抜ける     
    if 0.0000001 > resmax or counter > 1000:
        break
    else: # gh > res にコピーしないとだめ
        res = gh
    
    counter += 1         


# 内挿処理＞ファイル書き出し
# 内挿する
readdata = xr.DataArray(gh,dims=['time','lat','lon']) # ファイルを読み込んでXArrayに入れる\n",
# dim_0 10日平均 dim_1 lat緯度 dim_2 lon経度\n",

days = 0
temp = []

while days < 2664:
    slicedays = readdata[days,:,:]
    zi = interpolate.interp2d(orig_lon,orig_lat,slicedays,kind='linear')(xi[:,0],yi[0,:])

    hozon = np.append(temp,zi).reshape([days+1,180,360])

    temp = hozon

    days += 1

hozon.tofile('nc1ex1deg.heatf10dy.1948-2021.dat')

#ガベージコレクタ\n",
del temp,hozon
gc.collect()

### Net solar fluxの計算


## landmask

snr = dsr - usr


# ilandで陸地だったらfreshの同位置を０にする\n",
# bool indexで配列(landmask)が返る\n",
landmask = (iland >= 1)
# True の部分（地上）を０にする\n",
index = 0
while index < 2664:
    snr[index,:,:][landmask == True] = 0
    index += 1


resmax = 0.0

landindex =  np.where(landmask == True)
# landindexは陸地(landmask==True)の座標がタプルのndarray(配列）で返る
# landindex[0] = latitude
# landindex[1] = longitude
counter = 0

while True:
    for index in range(2664):
        for i,lat in enumerate(landindex[0]): # i = loop index
            lon = landindex[1][i]
            # この時点でのlat,lon(landindex[1][i])が陸地なので4点計算する
        # 画面端の処理
        if lat == 0:
            for lon in range(192-1):
                snr[191,lat,lon] = snr[0,lat,lon]
                if lon == 190: # 0スタートなので1少ない
                    snr[0,lat,lon] = snr[191,lat,lon]
        # lat,lon(landindex[1][i])が陸地なので4点計算する
            res = abs(snr[index,lat,lon]-0.25*(snr[index,lat-1,lon]+snr[index,lat+1,lon]+snr[index,lat,lon-1]+snr[index,lat,lon+1]))
            snr[index,lat,lon] = 0.25*(snr[index,lat-1,lon]+snr[index,lat+1,lon]+snr[index,lat,lon-1]+snr[index,lat,lon+1])
            resmax = max(resmax,res)

# 差分が範囲超えるまで、又は1000回ループしたら抜ける     
    if 0.0000001 > resmax or counter > 1000:
        break
    else: # snr > res にコピーしないとだめ
        res = snr
    
    counter += 1         


## 内挿＞ファイル書き出し
readdata = xr.DataArray(snr,dims=['time','lat','lon']) # ファイルを読み込んでXArrayに入れる\n",
# dim_0 10日平均 dim_1 lat緯度 dim_2 lon経度\n",

days = 0
temp = []

while days < 2664:
    slicedays = readdata[days,:,:]
    zi = interpolate.interp2d(orig_lon,orig_lat,slicedays,kind='linear')(xi[:,0],yi[0,:])

    hozon = np.append(temp,zi).reshape([days+1,180,360])

    temp = hozon

    days += 1

hozon.tofile('nc1ex1deg.snr10dy.1948-2021.dat')

#ガベージコレクタ\n",
del temp,hozon
gc.collect()
