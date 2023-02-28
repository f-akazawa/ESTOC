#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


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


# In[3]:


### Fresh water fluxの計算


# In[4]:


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


# In[5]:


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


# In[6]:


#ここで一度にする、陸地をダミーデータにしておく
# 一旦内挿してサイズを合わせる
# ESTOCで１とNCEPで１以外のときループさせる

#マスクの１℃１℃とデータの入った１℃１℃ができる
#NCEPのマスクデータをみて１以外のところは０にする
#ESToCが１でNCEPが１以外のところをループ
#使って良いデータestocのマスクで１のところだけ


# In[8]:


# freshのサイズをESTOCに合わせる内挿処理
readdata = xr.DataArray(fresh,dims=['time','lat','lon']) # ファイルを読み込んでXArrayに入れる\n",
# dim_0 10日平均 dim_1 lat緯度 dim_2 lon経度\n",

days = 0
temp = []

while days < 2664:
    slicedays = readdata[days,:,:]
    zi = interpolate.interp2d(orig_lon,orig_lat,slicedays,kind='linear')(xi[:,0],yi[0,:])

    interp_fresh = np.append(temp,zi).reshape([days+1,180,360])

    temp = interp_fresh

    days += 1


print('end')


# In[9]:


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


# In[11]:


# NCEPランドマスク作成
# オリジナルは海が0なので陸地を0に反転させる
# NCEPデータの陸地を0にする
rev_iland = np.where(iland == 0 ,1,0)


# In[12]:


# ESTOCランドマスクに合わせる内挿処理

ncepmask = interpolate.interp2d(orig_lon,orig_lat,rev_iland,kind='linear')(xi[:,0],yi[0,:])


# In[16]:


######################################################
######################################################
#ESTOCで１とNCEPで１以外のときループさせる
#マスクの１℃１℃とデータの入った１℃１℃ができる
#NCEPのマスクデータをみて１以外のところは０にする
#ESToCが０以外でNCEPが１以外のところをループ
#使って良いデータestocのマスクで１のところだけ
######################################################
######################################################


# In[45]:


resmax = 0.0

#landindex =  np.where(estoclandmask == True)
landindex =  np.where(estocmask != 0) and np.where(ncepmask != 1)
# landindexは陸地(landmask==True)の座標がタプルのndarray(配列）で返る
# landindex[0] = latitude
# landindex[1] = longitude

for index in range(2664): # ループが遅いのでテストで1年分だけ出してみる、本当は74年分で2664
    for counter in range(1000):# 1000回繰り返す（又は閾値を超えたらループ終了
        for i,lat in enumerate(landindex[0]): # i = loop index
            lon = landindex[1][i]
            if lat > 1 and lat < 180 and lon < 359: # 画面端は処理できないので除外
                # lat,lon(landindex[1][i])が陸地なので4点計算する
                res = abs(interp_fresh[index,lat,lon]-0.25*(interp_fresh[index,lat-1,lon]+interp_fresh[index,lat+1,lon]+interp_fresh[index,lat,lon-1]+interp_fresh[index,lat,lon+1]))
                interp_fresh[index,lat,lon] = 0.25*(interp_fresh[index,lat-1,lon]+interp_fresh[index,lat+1,lon]+interp_fresh[index,lat,lon-1]+interp_fresh[index,lat,lon+1])
                resmax = max(resmax,res)

            # この時点でのlat,lon(landindex[1][i])が陸地なので4点計算する
            # 画面端の処理
            if lat == 0:
                for lon in range(360-1):
                    interp_fresh[360,lat,lon] = interp_fresh[0,lat,lon]
                    if lon == 358: # 0スタートなので1少ない
                        interp_fresh[0,lat,lon] = interp_fresh[359,lat,lon]

            # 差分が範囲超えたら抜ける
            if 0.0000001 > resmax:
                break
            else: # fresh > res にコピーしないとだめ
                res = interp_fresh
        else:
            continue
        break
    else:
        continue
    break

interp_fresh.tofile('fwflux10dy-1948-2021.dat')
print('end')


# In[46]:


## ガベージコレクタ
del temp,interp_fresh
gc.collect()

### Momentum fluxの計算
### uflux10dy.dat vflux10dy.dataを使う


# In[47]:


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


# In[48]:


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



#ガベージコレクタ\n",
#del temp,hozon
#gc.collect()

print('end')


# In[50]:


resmax = 0.0

#landindex =  np.where(landmask == True)
landindex =  np.where(estocmask != 0) and np.where(ncepmask != 1)
# landindexは陸地(landmask==True)の座標がタプルのndarray(配列）で返る
# landindex[0] = latitude
# landindex[1] = longitude
counter = 0

for index in range(2664): # origin 2664(74years 1948-2021)
    for counter in range(1000):
        for i,lat in enumerate(landindex[0]): # i = loop index
            lon = landindex[1][i]
            if lat > 1 and lat < 180 and lon < 359: # 画面端は処理できないので除外
                # lat,lon(landindex[1][i])が陸地なので4点計算する
                res = abs(interp_vflx[index,lat,lon]-0.25*(interp_vflx[index,lat-1,lon]+interp_vflx[index,lat+1,lon]+interp_vflx[index,lat,lon-1]+interp_vflx[index,lat,lon+1]))
                interp_vflx[index,lat,lon] = 0.25*(interp_vflx[index,lat-1,lon]+interp_vflx[index,lat+1,lon]+interp_vflx[index,lat,lon-1]+interp_vflx[index,lat,lon+1])
                resmax = max(resmax,res)
            # この時点でのlat,lon(landindex[1][i])が陸地なので4点計算する
            # 画面端の処理
            if lat == 0:
                for lon in range(360-1):
                    interp_vflx[360,lat,lon] = interp_vflx[0,lat,lon]
                    if lon == 358: # 0スタートなので1少ない
                        interp_vflx[0,lat,lon] = interp_vflx[359,lat,lon]

            # 差分が範囲超えるまで、又は1000回ループしたら抜ける     
            if 0.0000001 > resmax:
                break
            else: # vflx > res にコピーしないとだめ
                res = interp_vflx
        else:
            continue
        break
    else:
        continue
    break
    
interp_vflx.tofile('nc1ex1deg.vflx10dy.1948-2021.dat')
print('end')


# In[51]:


## gabage collector
del temp,interp_vflx
gc.collect()


# In[52]:


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



print('end')


# In[53]:


resmax = 0.0

#landindex =  np.where(landmask == True)
landindex =  np.where(estocmask != 0) and np.where(ncepmask != 1)
# landindexは陸地(landmask==True)の座標がタプルのndarray(配列）で返る
# landindex[0] = latitude
# landindex[1] = longitude
counter = 0

for index in range(2664):
    for counter in range (1000):
        for i,lat in enumerate(landindex[0]): # i = loop index
            lon = landindex[1][i]
            if lat >1 and lat < 180 and lon < 359: #画面端は除外して別処理
                # lat,lon(landindex[1][i])が陸地なので4点計算する
                res = abs(interp_uflx[index,lat,lon]-0.25*(interp_uflx[index,lat-1,lon]+interp_uflx[index,lat+1,lon]+interp_uflx[index,lat,lon-1]+interp_uflx[index,lat,lon+1]))
                interp_uflx[index,lat,lon] = 0.25*(interp_uflx[index,lat-1,lon]+interp_uflx[index,lat+1,lon]+interp_uflx[index,lat,lon-1]+interp_uflx[index,lat,lon+1])
                resmax = max(resmax,res)
            # 画面端の処理
            if lat == 0:
                for lon in range(360-1):
                    interp_uflx[360,lat,lon] = interp_uflx[0,lat,lon]
                    if lon == 358: # 0スタートなので1少ない
                        interp_uflx[0,lat,lon] = interp_uflx[359,lat,lon]

            # 差分が範囲超えるまで、又は1000回ループしたら抜ける     
            if 0.0000001 > resmax:
                break
            else: # uflx > res にコピーしないとだめ
                res = interp_uflx
        else:
            continue
        break
    else:
        continue
    break
    
interp_uflx.tofile('nc1ex1deg.uflx10dy.1948-2021.dat')
print('end')


# In[54]:


#ガベージコレクタ\n",
del temp,interp_uflx
gc.collect()

### Net heat fluxの計算


# In[55]:


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


# In[57]:


# landmask
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


# In[58]:


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



print('end')


# In[61]:


resmax = 0.0

#landindex =  np.where(landmask == True)
landindex =  np.where(estocmask != 0) and np.where(ncepmask != 1)
# landindexは陸地(landmask==True)の座標がタプルのndarray(配列）で返る
# landindex[0] = latitude
# landindex[1] = longitude
counter = 0

for index in range(2664):
    for counter in range(1000):
        for i,lat in enumerate(landindex[0]): # i = loop index
            lon = landindex[1][i]
            if lat > 1 and lat < 180 and lon < 359: # 画面端は除外して別処理
                # lat,lon(landindex[1][i])が陸地なので4点計算する
                res = abs(interp_gh[index,lat,lon]-0.25*(interp_gh[index,lat-1,lon]+interp_gh[index,lat+1,lon]+interp_gh[index,lat,lon-1]+interp_gh[index,lat,lon+1]))
                interp_gh[index,lat,lon] = 0.25*(interp_gh[index,lat-1,lon]+interp_gh[index,lat+1,lon]+interp_gh[index,lat,lon-1]+interp_gh[index,lat,lon+1])
                resmax = max(resmax,res)
                
                # 画面端の処理
            if lat == 0:
                for lon in range(360-1):
                    interp_gh[360,lat,lon] = interp_gh[0,lat,lon]
                    if lon == 358: # 0スタートなので1少ない
                        interp_gh[0,lat,lon] = interp_gh[359,lat,lon]

            # 差分が範囲超えるまで、又は1000回ループしたら抜ける     
            if 0.0000001 > resmax:
                break
            else: # gh > res にコピーしないとだめ
                res = interp_gh
        else:
            continue
        break
    else:
        continue
    break
    
interp_gh.tofile('nc1ex1deg.heatf10dy.1948-2021.dat')
print('end')


# In[62]:


#ガベージコレクタ\n",
del temp,interp_gh
gc.collect()
### Net solar fluxの計算


# In[63]:


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


# In[64]:


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


print('end')


# In[66]:


resmax = 0.0

landindex =  np.where(landmask == True)
# landindexは陸地(landmask==True)の座標がタプルのndarray(配列）で返る
# landindex[0] = latitude
# landindex[1] = longitude
counter = 0

for index in range(2664):
    for counter in range(1000):
        for i,lat in enumerate(landindex[0]): # i = loop index
            lon = landindex[1][i]
            if lat > 1 and lat < 180 and lon < 359: # 画面端は除外して別処理
                # lat,lon(landindex[1][i])が陸地なので4点計算する
                res = abs(interp_snr[index,lat,lon]-0.25*(interp_snr[index,lat-1,lon]+interp_snr[index,lat+1,lon]+interp_snr[index,lat,lon-1]+interp_snr[index,lat,lon+1]))
                interp_snr[index,lat,lon] = 0.25*(interp_snr[index,lat-1,lon]+interp_snr[index,lat+1,lon]+interp_snr[index,lat,lon-1]+interp_snr[index,lat,lon+1])
                resmax = max(resmax,res)
                
                # この時点でのlat,lon(landindex[1][i])が陸地なので4点計算する
                # 画面端の処理
            if lat == 0:
                for lon in range(360-1):
                    interp_snr[360,lat,lon] = interp_snr[0,lat,lon]
                    if lon == 358: # 0スタートなので1少ない
                        interp_snr[0,lat,lon] = interp_snr[359,lat,lon]

            # 差分が範囲超えるまで、又は1000回ループしたら抜ける     
            if 0.0000001 > resmax:
                break
            else: # snr > res にコピーしないとだめ
                res = interp_snr
        else:
            continue
        break
    else:
        continue
    break

    
interp_snr.tofile('nc1ex1deg.snr10dy.1948-2021.dat')

#ガベージコレクタ\n",
#del interp_snr
#gc.collect()

print('end')


# In[ ]:




