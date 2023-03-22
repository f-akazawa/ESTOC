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

# 緯度経度は実際の数値を入れる
# 元データNCファイルの緯度経度の範囲を得る
ncep_param = xr.open_dataset('01_ESTOC_ForcingData/NCEP_NCAR_Forcing/NCEP/land.sfc.gauss.nc')

# ncep_paramにArrayとして保存されるので以下の様に取り出して値を得る

# データも反転させないといけない南北反転
# 10日平均作成の方も確認する

orig_lat = np.array(ncep_param['lat'])[::-1] # マイナス値が後になっているので反転する
orig_lon = np.append(np.array(ncep_param['lon']),360) # 360を追加しておく(3/16)

# mgridは最初、最後、間隔を指定するので間隔1のグリッドが出来る、返り値がmeshgridと縦横は逆になるので注意\n",
xi,yi = np.mgrid[0.5:360:1,-89.5:90:1]


# In[3]:


#landft06 = '01_ESTOC_ForcingData/NCEP_NCAR_Forcing/2017/4.fwat/inc/land.ft06.big' # pythonの場合reshape(94,192)で読み込む\n",
prevclimate = '01_ESTOC_ForcingData/NCEP_NCAR_Forcing/2017/4.fwat/inc/prev_climate7902nc2.big'

# land.06.ft.big読み込み\n",
# 01_ESTOC_ForcingData/NCEP_NCAR_Forcing/2017/4.fwat/inc/land.ft06.big\n",
# 縦 x 横(94 X 192)の世界地図イメージ　0/1で表現されている\n",
# python で読むと0と32831が94*192の配列に入っている。\n",
# 0が海、1(32831)は陸\n",
# landmask用に必要
# 過去の遺物なので今後はlandft06は使わないように変更する
#iland = np.fromfile(landft06,dtype='int32').reshape(94,192)

iland = np.array(ncep_param['land'])[0,:,:]
ilandzero = iland[:,0]
iland = np.hstack((iland,ilandzero.reshape(-1,1))) # 画面端の処理のため、経度193番目に経度0番を入れる

# prev.climate7902.nc2.big読み込み\n",
# 01_ESTOC_ForcingData/NCEP_NCAR_Forcing/2017/4.fwat/inc/prev_climate7902nc2.big\n",
# 縦ｘ横(94 x 192)が12枚あるイメージ\n",
# 12ヶ月分の何かしらの気候値\n",
clima = np.fromfile(prevclimate,dtype='float32').reshape(12,94,192)

#prate10dy.dat\n",
prate = np.fromfile('prate10dy.dat').reshape(2664,94,192)
# lthfl\n",
lhtfl = np.fromfile('lhtfl10dy.dat').reshape(2664,94,192)


# In[4]:



#ESTOCでは海だけど、NCEPでは陸地の部分
#NCEPで１度にしたときに陸地の場所
# 内挿する前にilandとclima = freshを使ってマスクする必要あり",
# prate と lhtflで要素ごとに計算\n",
# 以下の数式とfreshの命名規則はFotran元コードから"
fresh = (prate - lhtfl/2.5e6)*0.1 # ESTOC用に単位換算

# fresh 193を作り0番の値を入れておく（他も同様、画面端処理のため）
days = 0
temp = []

while days < 2664:
    sliceone = fresh[days,:,:]
    addzero = fresh[days,:,0]
    zi = np.hstack((sliceone,addzero.reshape(-1,1)))
    fresh193 = np.append(temp,zi).reshape([days+1,94,193])
    temp = fresh193
    
    days += 1


# In[5]:


# ilandで陸地だったらfreshの同位置を０にする\n",
# bool indexで配列(landmask)が返る\n",
landmask = (iland >= 1)
# True の部分（地上）を０にする\n",
index = 0
while index < 2664:
    fresh193[index,:,:][landmask == True] = 0
    index += 1


# In[6]:


### この時点でlandmaskは(94*193)
### 94*192だが、画面端の処理の為、193番目に0番のデータを入れる


# In[7]:


# freshのサイズをESTOCに合わせる内挿処理
readdata = xr.DataArray(fresh193,dims=['time','lat','lon']) # ファイルを読み込んでXArrayに入れる\n",
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


# In[8]:


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
    


# In[11]:


# ESTOCランドマスク作成
# （155 x 360) 75S-80N 上下にダミーの海を足す

# 元データは75Sから80Nまでなので上下90まで陸地0で埋める
topnorth = np.full((10,360),0)
bottomsouth = np.full((15,360),0)

# ESTOCマスクデータを読みこんで、フリップ
data = np.loadtxt('kmt_data.txt',dtype='int') # 元がテキストなのでキャストも必要
estocmask = np.flipud(data) # flipdで上下反転

# 上下にダミーの海を足す
estocmask = np.append(topnorth,estocmask,axis=0)
estocmask = np.append(estocmask,bottomsouth,axis=0)

estoclandmask = (estocmask == 0)


# In[12]:


# NCEPランドマスク作成
# オリジナルは海が0なので陸地を0に反転させる
# NCEPデータの陸地を0にする
rev_iland = np.where(iland == 0 ,1,0)


# In[13]:


# ESTOCランドマスクに合わせる内挿処理

ncepmask = interpolate.interp2d(orig_lon,orig_lat,rev_iland,kind='linear')(xi[:,0],yi[0,:])


# In[15]:


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
mask = ((estocweight == 1) & (ncepmask < 0.9999))
landindex =  np.where(mask == 1)
######

fresh361_old = fresh361.copy() # 比較用コピー作成 pythonは＝だけだとメモリ共通なので元も代わってしまう

#####
##### 重み計算用のランドマスクを作成

maskflag = estocweight.copy()


# In[13]:


# landindexは陸地(landmask==True)の座標がタプルのndarray(配列）で返る
# landindex[0] = latitude
# landindex[1] = longitude

###
# fresh361_oldは0度の前に360の値、360度の先に0度の値を入れる(3/16)
###

for index in range(2664): # ループが遅いのでテストで1年分だけ出してみる、本当は74年分で2664
     for counter in range(1000):# 1000回繰り返す（又は閾値を超えたらループ終了
        resmax = -10000000000
        for i in enumerate(landindex[0]): # i = loop index
            lat = (landindex[0][i[0]])
            lon = (landindex[1][i[0]])
            
            if lon < 359: # estocweight out of bounds 対策　lon=360になることは無いのだが、エラーが出る為
                calcflag = estocweight[lat-1,lon] + estocweight[lat+1,lon] + estocweight[lat,lon-1] + estocweight[lat,lon+1] 
                calcdata = (fresh361_old[index,lat-1,lon]*estocweight[lat-1,lon] + fresh361_old[index,lat+1,lon]*estocweight[lat+1,lon]                            +fresh361_old[index,lat,lon-1]*estocweight[lat,lon-1] + fresh361_old[index,lat,lon+1]*estocweight[lat,lon+1])
                if calcflag > 0:
                    calcweight = calcdata/(calcflag+0.0000001)
                    res = abs(fresh361_old[index,lat,lon]-calcweight)
                    fresh361_old[index,lat,lon] = calcweight
                    maskflag[lat,lon] = 1
                    resmax = max(resmax,res)
        
        estocweight = maskflag
        
        if 0.0000001 > resmax:
            break
            
            
            #if lat >= 1 and lat < 179 and lon < 359: # 画面端は処理できないので除外 if入らないがlon+1するひつようあり
                # lat,lon(landindex[1][i])が陸地なので4点計算する
                #res = abs(fresh361[index,lat,lon]-0.25*(fresh361[index,lat-1,lon]+fresh361[index,lat+1,lon]+fresh361[index,lat,lon-1]+fresh361[index,lat,lon+1]))
                #fresh361[index,lat,lon] = (fresh361_old[index,lat-1,lon]*estocweight[lat-1,lon]+fresh361_old[index,lat+1,lon]*estocweight[lat+1,lon]+fresh361_old[index,lat,lon-1]*estocweight[lat,lon-1]+fresh361_old[index,lat,lon+1]*estocweight[lat,lon+1])\
                #/(estocweight[lat-1,lon]+estocweight[lat+1,lon]+estocweight[lat,lon-1]+estocweight[lat,lon+1] + 0.0000001 )\
                         
                #res = abs(fresh361[index,lat,lon] - fresh361_old[index,lat,lon])
                
                #resmax = max(resmax,res)
            # 差分が範囲超えたら抜ける
        #if 0.0000001 > resmax:
            #break
        #else:
            #fresh361_old[index,:,:] = fresh361[index,:,:]


fresh361[:,:,:360].tofile('fwflux10dy-1948-2021.dat')
print('end')


# In[14]:


## ガベージコレクタ
del temp,interp_fresh,fresh361,fresh361_old,fresh,fresh193
gc.collect()

### Momentum fluxの計算
### uflux10dy.dat vflux10dy.dataを使う


# In[4]:


## ESTOCマスク専用を用意するxi,yiが変わる
## uflx,vflxはESTOCマスクを変更する
## mgridを公差ではなく項数にするためにjをつける
xi,yi = np.mgrid[1:360:360j,-89:90:180j]


# uflx
uflx = np.fromfile('uflx10dy.dat').reshape(2664,94,192)
# vflx
vflx = np.fromfile('vflx10dy.dat').reshape(2664,94,192)

# landmask
# fresh water では計算式があったが、momentum fluxでは無い
# *10は単位換算
uflux = uflx * 10
vflux = vflx * 10


# In[5]:


## uflux,vfluxの193番目に0番を追加する、画面端の処理のため

days = 0
temp = []

while days < 2664:
    sliceone = uflux[days,:,:]
    addzero = uflux[days,:,0]
    zi = np.hstack((sliceone,addzero.reshape(-1,1)))
    uflux193 = np.append(temp,zi).reshape([days+1,94,193])
    temp = uflux193
    
    days += 1


days = 0
temp = []

while days < 2664:
    sliceone = vflux[days,:,:]
    addzero = vflux[days,:,0]
    zi = np.hstack((sliceone,addzero.reshape(-1,1)))
    vflux193 = np.append(temp,zi).reshape([days+1,94,193])
    temp = vflux193
    
    days += 1


# In[6]:



# ilandで陸地だったらfreshの同位置を０にする\n",
# bool indexで配列(landmask)が返る\n",
landmask = (iland >= 1)
# True の部分（地上）を０にする\n",
index = 0
while index < 2664:
    uflux193[index,:,:][landmask == True] = 0
    vflux193[index,:,:][landmask == True] = 0
    index += 1


# In[20]:


# vfluxを内挿してESTOCサイズに合わせる。
# 内挿する
readdata = xr.DataArray(vflux193,dims=['time','lat','lon']) # ファイルを読み込んでXArrayに入れる\n",
# dim_0 10日平均 dim_1 lat緯度 dim_2 lon経度\n",

days = 0
temp = []

while days < 2664:
    slicedays = readdata[days,:,:]
    zi = interpolate.interp2d(orig_lon,orig_lat,slicedays,kind='linear')(xi[:,0],yi[0,:])

    interp_vflx = np.append(temp,zi).reshape([days+1,180,360])

    temp = interp_vflx

    days += 1


# In[21]:


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
    


# In[25]:


## 明示的にコピーを作成
vflx361_old = vflx361.copy() # 計算用コピー作成
estocweight = np.where(estoclandmask == True,0,1) # estocweight初期化（他でも使っているので一応）
maskflag = estocweight.copy() # maskflag初期化（他でも使っているので一応）


# In[26]:


# landindexは陸地(landmask==True)の座標がタプルのndarray(配列）で返る
# landindex[0] = latitude
# landindex[1] = longitude



for index in range(2664): # 74年分で2664
    for counter in range(1000):# 1000回繰り返す（又は閾値を超えたらループ終了
        resmax = -10000000000
        for i in enumerate(landindex[0]): # i = loop index
            lat = (landindex[0][i[0]])
            lon = (landindex[1][i[0]])
            if lon < 359: # lon=360は無いのだが、out of boundsが出るのでその対策。画面端は処理できないので除外
                calcflag = estocweight[lat-1,lon] + estocweight[lat+1,lon] + estocweight[lat,lon-1] + estocweight[lat,lon+1]
                calddata = (vflx361_old[index,lat-1,lon]*estocweight[lat-1,lon]+vflx361_old[index,lat+1,lon]*estocweight[lat+1,lon]                            +vflx361_old[index,lat,lon-1]*estocweight[lat,lon-1]+vflx361_old[index,lat,lon+1]*estocweight[lat,lon+1])
                
                if calcflag > 0:
                    calcweight = calcdata/(calcflag+0.0000001)
                    res = abs(vflx361_old[index,lat,lon] - calcweight)
                    vflx361_old[index,lat,lon] = calcweight
                    maskflag[lat,lon] = 1
                    resmax = max(resmax,res)
        estocweight = maskflag

            # 差分が範囲超えたら抜ける
            # 閾値は要調整
        if 0.001 > resmax:
            break

vflx361[:,:,:360].tofile('nc1ex1deg.vflx10dy.1948-2021.dat')
print('end')


# In[7]:


# uflxをESTOCサイズに合わせるため内挿する
readdata = xr.DataArray(uflux193,dims=['time','lat','lon']) # ファイルを読み込んでXArrayに入れる\n",
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


# In[8]:


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
    


# In[ ]:


uflx361_old = uflx361.copy() # 計算用コピー作成
estocweight = np.where(estoclandmask == True,0,1) # estocweight初期化（他でも使っているので一応）
maskflag = estocweight.copy() # maskflag初期化（他でも使っているので一応）


# In[ ]:


# landindexは陸地(landmask==True)の座標がタプルのndarray(配列）で返る
# landindex[0] = latitude
# landindex[1] = longitude

uflx361_old = uflx361.copy() # 計算用コピー作成

for index in range(2664): # ループが遅いのでテストで1年分だけ出してみる、本当は74年分で2664
    for counter in range(1000):# 1000回繰り返す（又は閾値を超えたらループ終了
        resmax = -10000000000
        for i in enumerate(landindex[0]): # i = loop index
            lat = (landindex[0][i[0]])
            lon = (landindex[1][i[0]])
            if lon < 359: # lon=360はlandindexい無いのだが、out of bounds対策
                # lat,lon(landindex[1][i])が陸地なので4点計算する
                calcflag = estocweight[lat-1,lon] + estocweight[lat+1,lon] + estocweight[lat,lon-1] + estocweight[lat,lon+1] 
                calcdata = (uflx361_old[index,lat-1,lon]*estocweight[lat-1,lon]+uflx361_old[index,lat+1,lon]*estocweight[lat+1,lon]                            +uflx361_old[index,lat,lon-1]*estocweight[lat,lon-1]+uflx361_old[index,lat,lon+1]*estocweight[lat,lon+1])
                
                if calcflag > 0:
                    calcweight = calcdata/(calcflag+0.0000001)
                    res = abs(uflx361_old[index,lat,lon] - calcweight)
                    uflx361_old[index,lat,lon] = calcweight
                    maskflag[lat,lon] = 1
                    resmax = max(resmax,res)
        
        estocweight = maskflag
        #print(resmax)
        # 差分が範囲超えたら抜ける
        if 0.001 > resmax:
            print('break counter=',counter,'index=',index)
            break
        #else:
            #print(resmax)

uflx361[:,:,:360].tofile('nc1ex1deg.uflx10dy.1948-2021.dat')
print('end')


# In[44]:


#ガベージコレクタ\n",
del temp,interp_uflx,uflx361,uflx361_old,uflx,vflux,uflux193,vflux193,vflx361,vflx361_old
gc.collect()

### Net heat fluxの計算


# In[45]:


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


# In[58]:


# landmask
# landmask
# Net heat fluxは以下の式
## ESTOC用に単位換算
gh = ((dsr - usr + dlr - ulr) - (sh + lh))



# 193番目を作り0番目の値を入れておく（他も同様、画面端処理のため）
days = 0
temp = []

while days < 2664:
    sliceone = gh[days,:,:]
    addzero = gh[days,:,0]
    zi = np.hstack((sliceone,addzero.reshape(-1,1)))
    gh193 = np.append(temp,zi).reshape([days+1,94,193])
    temp = gh193
    
    days += 1

# ilandで陸地だったらfreshの同位置を０にする\n",
# bool indexで配列(landmask)が返る\n",
landmask = (iland >= 1)
# True の部分（地上）を０にする\n",
index = 0
while index < 2664:
    gh193[index,:,:][landmask == True] = 0
    index += 1


# In[48]:


## 内挿の座標値を元に戻す（vflux,ufluxだけ座標値が変わる）
xi,yi = np.mgrid[0.5:360:1,-89.5:90:1]


# In[49]:


# ESTOCのサイズに合わせるため内挿する
readdata = xr.DataArray(gh193,dims=['time','lat','lon']) # ファイルを読み込んでXArrayに入れる\n",
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


# In[50]:


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
    


# In[51]:


gh361_old = gh361.copy() # 計算用コピー作成
estocweight = np.where(estoclandmask == True,0,1) # estocweight初期化（他でも使っているので一応）
maskflag = estocweight.copy() # maskflag初期化（他でも使っているので一応）


# In[61]:


# landindexは陸地(landmask==True)の座標がタプルのndarray(配列）で返る
# landindex[0] = latitude
# landindex[1] = longitude


for index in range(2664): # ループが遅いのでテストで1年分だけ出してみる、本当は74年分で2664
    for counter in range(1000):# 1000回繰り返す（又は閾値を超えたらループ終了
        resmax = -10000000000
        for i in enumerate(landindex[0]): # i = loop index
            lat = (landindex[0][i[0]])
            lon = (landindex[1][i[0]])
            if lon < 359: # out of bounds対策
                # lat,lon(landindex[1][i])が陸地なので4点計算する
                calcflag = estocweight[lat-1,lon] + estocweight[lat+1,lon] + estocweight[lat,lon-1] + estocweight[lat,lon+1] 
                calcdata= (gh361_old[index,lat-1,lon]*estocweight[lat-1,lon]+gh361_old[index,lat+1,lon]*estocweight[lat+1,lon]                           +gh361_old[index,lat,lon-1]*estocweight[lat,lon-1]+gh361_old[index,lat,lon+1]*estocweight[lat,lon+1])
                if calcflag > 0:
                    calcweight = calcdata/(calcflag+0.00000001)
                    res = abs(gh361_old[index,lat,lon] - calcweight)
                    gh361_old[index,lat,lon] = calcweight
                    maskflag[lat,lon] = 1
                    resmax = max(resmax,res)
            
        estocweight = maskflag

            # 差分が範囲超えたら抜ける
        if 0.1 > resmax:
            print('index=',index,'counter=',counter)
            break
        
gh361[:,:,:360].tofile('nc1ex1deg.heatf10dy.1948-2021.dat')
print('end')


# In[63]:


#ガベージコレクタ\n",
del temp,interp_gh,gh361,gh361_old,gh,gh193
gc.collect()
### Net solar fluxの計算


# In[64]:


## net solar flux
snr = dsr - usr

## 単位換算


# 193番目を作り0番目の値を入れておく（他も同様、画面端処理のため）
days = 0
temp = []

while days < 2664:
    sliceone = snr[days,:,:]
    addzero = snr[days,:,0]
    zi = np.hstack((sliceone,addzero.reshape(-1,1)))
    snr193 = np.append(temp,zi).reshape([days+1,94,193])
    temp = snr193
    
    days += 1

# ilandで陸地だったらfreshの同位置を０にする\n",
# bool indexで配列(landmask)が返る\n",
landmask = (iland >= 1)
# True の部分（地上）を０にする\n",
index = 0
while index < 2664:
    snr193[index,:,:][landmask == True] = 0
    index += 1


# In[65]:


## ESTOCサイズにするために内挿
readdata = xr.DataArray(snr193,dims=['time','lat','lon']) # ファイルを読み込んでXArrayに入れる\n",
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


# In[67]:


snr361_old = snr361.copy() # 計算用コピー作成
estocweight = np.where(estoclandmask == True,0,1) # estocweight初期化（他でも使っているので一応）
maskflag = estocweight.copy() # maskflag初期化（他でも使っているので一応）


# In[69]:


# landindexは陸地(landmask==True)の座標がタプルのndarray(配列）で返る
# landindex[0] = latitude
# landindex[1] = longitude



for index in range(2664): # ループが遅いのでテストで1年分だけ出してみる、本当は74年分で2664
    for counter in range(1000):# 1000回繰り返す（又は閾値を超えたらループ終了
        resmax = -10000000000
        for i in enumerate(landindex[0]): # i = loop index
            lat = (landindex[0][i[0]])
            lon = (landindex[1][i[0]])
            if lon < 359: # out of bounds対策
                calcflag = estocweight[lat-1,lon] + estocweight[lat+1,lon] + estocweight[lat,lon-1] + estocweight[lat,lon+1] 
                calcdata = snr361_old[index,lat-1,lon]*estocweight[lat-1,lon]+snr361_old[index,lat+1,lon]*estocweight[lat+1,lon]                +snr361_old[index,lat,lon-1]*estocweight[lat,lon-1]+snr361_old[index,lat,lon+1]*estocweight[lat,lon+1]
                
                if calcflag > 0:
                    calcweight = calcdata/(calcflag+0.0000001)
                    res = abs(snr361_old[index,lat,lon] - calcweight)
                    snr361_old[index,lat,lon] = calcweight
                    maskflag[lat,lon] = 1
                    resmax = max(resmax,res)
                    
        estocweight = maskflag
        
        # 差分が範囲超えたら抜ける
        if 0.5 > resmax:
            print('index=',index,'counter=',counter)
            break

snr361[:,:,:360].tofile('nc1ex1deg.snr10dy.1948-2021.dat')

print('end')


# In[ ]:


### ファイル変換ができたら、単位を合わせるNCEP > ESTOC
### SNR,VFLX,UFLXはグリーンランドの上はデータ更新されているか？
### 南極の近くはブロック状になっているので元データを見てみる。
#ヒートフラックがみやすい、秋～11月くらいを見てみる

