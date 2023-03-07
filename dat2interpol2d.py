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
# 緯度経度は実際の数値を入れる
# 元データNCファイルの緯度経度の範囲を得る
# とりあえずprateファイルを読む、指摘されたら変えること
ncep_param = xr.open_dataset('01_ESTOC_ForcingData/NCEP_NCAR_Forcing/NCEP/prate.sfc.gauss.1948.nc')

# ncep_paramにArrayとして保存されるので以下の様に取り出して値を得る
# print(ncep_param['lon'],min()) で最小、max()で最大が取れる
# print(ncep_param['lon'].actual_range)でも取れるが、正負の順番だったりして必ずしも先に最小が来るとは限らない。。。
ncep_lat_min = ncep_param['lat'].min()
ncep_lat_max = ncep_param['lat'].max()
ncep_lon_min = ncep_param['lon'].min()
ncep_lon_max = ncep_param['lon'].max()

# latitudeはStart-89.5、End89.5、Step数で元データの94分割になるように調整\n",
#orig_lat = np.arange(-89.5,89.5,1.905)
# 2022/1/26土居さんより依頼でStart/End-90,90に変更＞内挿後の値が180になるようにするため
##orig_lat = np.arange(-90,90,1.915)
step = (abs(ncep_lat_min) + abs(ncep_lat_max)) / 94
orig_lat = np.arange(ncep_lat_min,ncep_lat_max,step)

# longitudeはStart-180,End180,Stepで192分割に調整\n",
##orig_lon = np.arange(-180,180,1.875)
step = (abs(ncep_lon_min) + abs(ncep_lon_max)) / 192
orig_lon = np.arange(ncep_lon_min,ncep_lon_max,step)


# mgridは最初、最後、間隔を指定するので間隔1のグリッドが出来る、返り値がmeshgridと縦横は逆になるので注意\n",
# lat,lonは実際の座標値を使い、92*194 > 178*360に変更する（Stepは1）\n",
#xi,yi = np.mgrid[-180:180:1,-89.5:89.5:1]
##xi,yi = np.mgrid[-180:180:1,-90:90:1]
xi,yi = np.mgrid[0:360:1,-90:90:1]



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


#ESTOCではうみだけど、NCEPでは陸地の部分
#NCEPで１どにしたときに陸地の場所
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


#ここで一度にする、陸地をダミーデータにしておく
# 一旦内挿してサイズを合わせる
# ESTOCで１とNCEPで１以外のときループさせる

#マスクの１℃１℃とデータの入った１℃１℃ができる
#NCEPのマスクデータをみて１以外のところは０にする
#ESToCが１でNCEPが１以外のところをループ
#使って良いデータestocのマスクで１のところだけ


# freshのサイズをESTOCに合わせる内挿処理
readdata = xr.DataArray(fresh,dims=['time','lat','lon']) # ファイルを読み込んでXArrayに入れる\n",
# dim_0 10日平均 dim_1 lat緯度 dim_2 lon経度\n",

days = 0
temp = []

while days < 2664:
    slicedays = readdata[days,:,:]
    zi = interpolate.interp2d(orig_lon,orig_lat,slicedays,kind='linear')(xi[:,0],yi[0,:])
    #zi = interpolate.interp2d(orig_lon,orig_lat,slicedays,kind='linear')
    #zi = zi(xi[:,0],yi[0,:])
    
    interp_fresh = np.append(temp,zi).reshape([days+1,180,360])

    temp = interp_fresh

    days += 1


# 画面端の処理をするために横に1列増やす
days = 0
temp = []

while days < 2664:
    slicedays = interp_fresh[days,:,:]
    insertzero = slicedays[:,0] # 0番目の列
    zi = ( np.hstack((slicedays,insertzero.reshape(-1,1))) ) # ここで180 x 361になっている
    fresh361 = np.append(temp,zi).reshape(days+1,180,361)
    
    temp = fresh361
    
    days += 1
    


# ESTOCランドマスク作成
# （155 x 360) 75S-80N 上下にダミーの海を足す

# 元データは75Sから80Nまでなので上下90まで陸地で埋める
topnorth = np.full((10,360),99)
bottomsouth = np.full((15,360),99)

# 読みこんで、フリップ
data = np.loadtxt('kmt_data.txt',dtype='int') # 元がテキストなのでキャストも必要
estocmask = np.flipud(data) # flipdで上下反転

# 上下にダミーの海を足す
estocmask = np.append(topnorth,estocmask,axis=0)
estocmask = np.append(estocmask,bottomsouth,axis=0)

estoclandmask = (estocmask == 0)
#plt.imshow(estoclandmask)


# NCEPランドマスク作成
# オリジナルは海が0なので陸地を0に反転させる
# NCEPデータの陸地を0にする
rev_iland = np.where(iland == 0 ,1,0)


# ESTOCランドマスクに合わせる内挿処理

ncepmask = interpolate.interp2d(orig_lon,orig_lat,rev_iland,kind='linear')(xi[:,0],yi[0,:])

######################################################
######################################################
#ESTOCで１とNCEPで１以外のときループさせる
#マスクの１℃１℃とデータの入った１℃１℃ができる
#NCEPのマスクデータをみて１以外のところは０にする
#ESToCが０以外でNCEPが１以外のところをループ
#使って良いデータestocのマスクで１のところだけ
######################################################
######################################################


#estoclandmask は陸地True,海FalseのBool配列なので海1、陸0に直す
estocweight = np.where(estoclandmask == True,0,1)

#######
mask = ((estocweight == 1) & (ncepmask != 1))
landindex =  np.where(mask == 1)
######

# landindexは陸地(landmask==True)の座標がタプルのndarray(配列）で返る
# landindex[0] = latitude
# landindex[1] = longitude

fresh361_old = fresh361 # 比較用コピー作成

for index in range(2664): # ループが遅いのでテストで1年分だけ出してみる、本当は74年分で2664
    for counter in range(1000):# 1000回繰り返す（又は閾値を超えたらループ終了
        resmax = -10000000000
        for i,lat in enumerate(landindex[0]): # i = loop index
            lon = landindex[1][i]
            if lat >= 1 and lat < 179 and lon < 359: # 画面端は処理できないので除外
                # lat,lon(landindex[1][i])が陸地なので4点計算する
                #res = abs(fresh361[index,lat,lon]-0.25*(fresh361[index,lat-1,lon]+fresh361[index,lat+1,lon]+fresh361[index,lat,lon-1]+fresh361[index,lat,lon+1]))
                fresh361[index,lat,lon] = (fresh361_old[index,lat-1,lon]*estocweight[lat-1,lon]+fresh361_old[index,lat+1,lon]*estocweight[lat+1,lon]+fresh361_old[index,lat,lon-1]*estocweight[lat,lon-1]+fresh361_old[index,lat,lon+1]*estocweight[lat,lon+1])                /(estocweight[lat-1,lon]+estocweight[lat+1,lon]+estocweight[lat,lon-1]+estocweight[lat,lon+1] + 0.0000001 )                
           
                res = abs(fresh361[index,lat,lon] - fresh361_old[index,lat,lon])
                
                resmax = max(resmax,res)

            # 差分が範囲超えたら抜ける
        if 0.0000001 > resmax:
            break
        else:
            fresh361_old[index,:,:] = fresh361[index,:,:]

fresh361[:,:,:360].tofile('fwflux10dy-1948-2021.dat')
print('end')


## ガベージコレクタ
del temp,interp_fresh,fresh361,fresh361_old,fresh
gc.collect()

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


# vfluxを内挿してESTOCサイズに合わせる。
# 内挿する
readdata = xr.DataArray(vflx,dims=['time','lat','lon']) # ファイルを読み込んでXArrayに入れる\n",
# dim_0 10日平均 dim_1 lat緯度 dim_2 lon経度\n",

days = 0
temp = []

while days < 2664:
    slicedays = readdata[days,:,:]
    zi = interpolate.interp2d(orig_lon,orig_lat,slicedays,kind='linear')(xi[:,0],yi[0,:])

    interp_vflx = np.append(temp,zi).reshape([days+1,180,360])

    temp = interp_vflx

    days += 1



# 画面端の処理をするために横に1列増やす
days = 0
temp = []

while days < 2664:
    slicedays = interp_vflx[days,:,:]
    insertzero = slicedays[:,0] # 0番目の列
    zi = ( np.hstack((slicedays,insertzero.reshape(-1,1))) ) # ここで180 x 361になっている
    vflx361 = np.append(temp,zi).reshape(days+1,180,361)
    
    temp = vflx361
    
    days += 1
    


vflx361_old = vflx361 # 比較用コピー作成

for index in range(2664): # ループが遅いのでテストで1年分だけ出してみる、本当は74年分で2664
    for counter in range(1000):# 1000回繰り返す（又は閾値を超えたらループ終了
        resmax = -10000000000
        for i,lat in enumerate(landindex[0]): # i = loop index
            lon = landindex[1][i]
            if lat >= 1 and lat < 179 and lon < 359: # 画面端は処理できないので除外
                # lat,lon(landindex[1][i])が陸地なので4点計算する
                vflx361[index,lat,lon] = (vflx361_old[index,lat-1,lon]*estocweight[lat-1,lon]+vflx361_old[index,lat+1,lon]*estocweight[lat+1,lon]+vflx361_old[index,lat,lon-1]*estocweight[lat,lon-1]+vflx361_old[index,lat,lon+1]*estocweight[lat,lon+1])                /(estocweight[lat-1,lon]+estocweight[lat+1,lon]+estocweight[lat,lon-1]+estocweight[lat,lon+1] + 0.0000001 )
                         
                res = abs(vflx361[index,lat,lon] - vflx361_old[index,lat,lon])
                
                resmax = max(resmax,res)

            # 差分が範囲超えたら抜ける
        if 0.0000001 > resmax:
            break
        else:
            vlfx361_old[index,:,:] = vflx361[index,:,:]

vflx361[:,:,:360].tofile('nc1ex1deg.vflx10dy.1948-2021.dat')

## gabage collector
del temp,interp_vflx,vflx,vflx361,vflx361_old
gc.collect()


# uflxをESTOCサイズに合わせるため内挿する
readdata = xr.DataArray(uflx,dims=['time','lat','lon']) # ファイルを読み込んでXArrayに入れる\n",
# dim_0 10日平均 dim_1 lat緯度 dim_2 lon経度\n",

days = 0
temp = []

while days < 2664:
    slicedays = readdata[days,:,:]
    zi = interpolate.interp2d(orig_lon,orig_lat,slicedays,kind='linear')(xi[:,0],yi[0,:])

    interp_uflx = np.append(temp,zi).reshape([days+1,180,360])

    temp = interp_uflx

    days += 1


# 画面端の処理をするために横に1列増やす
days = 0
temp = []

while days < 2664:
    slicedays = interp_uflx[days,:,:]
    insertzero = slicedays[:,0] # 0番目の列
    zi = ( np.hstack((slicedays,insertzero.reshape(-1,1))) ) # ここで180 x 361になっている
    uflx361 = np.append(temp,zi).reshape(days+1,180,361)
    
    temp = uflx361
    
    days += 1
    


uflx361_old = uflx361 # 比較用コピー作成

for index in range(2664): # ループが遅いのでテストで1年分だけ出してみる、本当は74年分で2664
    for counter in range(1000):# 1000回繰り返す（又は閾値を超えたらループ終了
        resmax = -10000000000
        for i,lat in enumerate(landindex[0]): # i = loop index
            lon = landindex[1][i]
            if lat >= 1 and lat < 179 and lon < 359: # 画面端は処理できないので除外
                # lat,lon(landindex[1][i])が陸地なので4点計算する
                uflx361[index,lat,lon] = (uflx361_old[index,lat-1,lon]*estocweight[lat-1,lon]+uflx361_old[index,lat+1,lon]*estocweight[lat+1,lon]+uflx361_old[index,lat,lon-1]*estocweight[lat,lon-1]+uflx361_old[index,lat,lon+1]*estocweight[lat,lon+1])                /(estocweight[lat-1,lon]+estocweight[lat+1,lon]+estocweight[lat,lon-1]+estocweight[lat,lon+1] + 0.0000001 )
                         
                res = abs(uflx361[index,lat,lon] - uflx361_old[index,lat,lon])
                
                resmax = max(resmax,res)

            # 差分が範囲超えたら抜ける
        if 0.0000001 > resmax:
            break
        else:
            ulfx361_old[index,:,:] = uflx361[index,:,:]

uflx361[:,:,:360].tofile('nc1ex1deg.uflx10dy.1948-2021.dat')

#ガベージコレクタ\n",
del temp,interp_uflx,uflx361,uflx361_old,uflx,uflux
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


# ESTOCのサイズに合わせるため内挿する
readdata = xr.DataArray(gh,dims=['time','lat','lon']) # ファイルを読み込んでXArrayに入れる\n",
# dim_0 10日平均 dim_1 lat緯度 dim_2 lon経度\n",

days = 0
temp = []

while days < 2664:
    slicedays = readdata[days,:,:]
    zi = interpolate.interp2d(orig_lon,orig_lat,slicedays,kind='linear')(xi[:,0],yi[0,:])

    interp_gh = np.append(temp,zi).reshape([days+1,180,360])

    temp = interp_gh

    days += 1


# 画面端の処理をするために横に1列増やす
days = 0
temp = []

while days < 2664:
    slicedays = interp_gh[days,:,:]
    insertzero = slicedays[:,0] # 0番目の列
    zi = ( np.hstack((slicedays,insertzero.reshape(-1,1))) ) # ここで180 x 361になっている
    gh361 = np.append(temp,zi).reshape(days+1,180,361)
    
    temp = gh361
    
    days += 1
    


gh361_old = gh361 # 比較用コピー作成

for index in range(2664): # ループが遅いのでテストで1年分だけ出してみる、本当は74年分で2664
    for counter in range(1000):# 1000回繰り返す（又は閾値を超えたらループ終了
        resmax = -10000000000
        for i,lat in enumerate(landindex[0]): # i = loop index
            lon = landindex[1][i]
            if lat >= 1 and lat < 179 and lon < 359: # 画面端は処理できないので除外
                # lat,lon(landindex[1][i])が陸地なので4点計算する
                gh361[index,lat,lon] = (gh361_old[index,lat-1,lon]*estocweight[lat-1,lon]+gh361_old[index,lat+1,lon]*estocweight[lat+1,lon]+gh361_old[index,lat,lon-1]*estocweight[lat,lon-1]+gh361_old[index,lat,lon+1]*estocweight[lat,lon+1])                /(estocweight[lat-1,lon]+estocweight[lat+1,lon]+estocweight[lat,lon-1]+estocweight[lat,lon+1] + 0.0000001 )
                         
                res = abs(gh361[index,lat,lon] - gh361_old[index,lat,lon])
                
                resmax = max(resmax,res)

            # 差分が範囲超えたら抜ける
        if 0.0000001 > resmax:
            break
        else:
            gh361_old[index,:,:] = gh361[index,:,:]

gh361[:,:,:360].tofile('nc1ex1deg.heatf10dy.1948-2021.dat')


#ガベージコレクタ\n",
del temp,interp_gh,gh361,gh361_old,gh
gc.collect()


### Net solar fluxの計算
snr = dsr - usr


# ilandで陸地だったらfreshの同位置を０にする\n",
# bool indexで配列(landmask)が返る\n",
landmask = (iland >= 1)
# True の部分（地上）を０にする\n",
index = 0
while index < 2664:
    snr[index,:,:][landmask == True] = 0
    index += 1


## ESTOCサイズにするために内挿
readdata = xr.DataArray(snr,dims=['time','lat','lon']) # ファイルを読み込んでXArrayに入れる\n",
# dim_0 10日平均 dim_1 lat緯度 dim_2 lon経度\n",

days = 0
temp = []

while days < 2664:
    slicedays = readdata[days,:,:]
    zi = interpolate.interp2d(orig_lon,orig_lat,slicedays,kind='linear')(xi[:,0],yi[0,:])

    interp_snr = np.append(temp,zi).reshape([days+1,180,360])

    temp = interp_snr

    days += 1


# 画面端の処理をするために横に1列増やす
days = 0
temp = []

while days < 2664:
    slicedays = interp_snr[days,:,:]
    insertzero = slicedays[:,0] # 0番目の列
    zi = ( np.hstack((slicedays,insertzero.reshape(-1,1))) ) # ここで180 x 361になっている
    snr361 = np.append(temp,zi).reshape(days+1,180,361)
    
    temp = snr361
    
    days += 1


snr361_old = snr361 # 比較用コピー作成

for index in range(2664): # ループが遅いのでテストで1年分だけ出してみる、本当は74年分で2664
    for counter in range(1000):# 1000回繰り返す（又は閾値を超えたらループ終了
        resmax = -10000000000
        for i,lat in enumerate(landindex[0]): # i = loop index
            lon = landindex[1][i]
            if lat >= 1 and lat < 179 and lon < 359: # 画面端は処理できないので除外
                # lat,lon(landindex[1][i])が陸地なので4点計算する
                snr361[index,lat,lon] = (snr361_old[index,lat-1,lon]*estocweight[lat-1,lon]+snr361_old[index,lat+1,lon]*estocweight[lat+1,lon]+snr361_old[index,lat,lon-1]*estocweight[lat,lon-1]+snr361_old[index,lat,lon+1]*estocweight[lat,lon+1])                /(estocweight[lat-1,lon]+estocweight[lat+1,lon]+estocweight[lat,lon-1]+estocweight[lat,lon+1] + 0.0000001 )
                         
                res = abs(snr361[index,lat,lon] - snr361_old[index,lat,lon])
                
                resmax = max(resmax,res)

            # 差分が範囲超えたら抜ける
        if 0.0000001 > resmax:
            break
        else:
            snr361_old[index,:,:] = snr361[index,:,:]

snr361[:,:,:360].tofile('nc1ex1deg.snr10dy.1948-2021.dat')
