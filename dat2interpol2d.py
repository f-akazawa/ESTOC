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
import os

# 緯度経度は実際の数値を入れる
# 元データNCファイルの緯度経度の範囲を得る
ncep_param = xr.open_dataset('01_ESTOC_ForcingData/NCEP_NCAR_Forcing/NCEP/land.sfc.gauss.nc')

# ncepからのデータは読んだら必ずフリップ(4/4)
iland = np.array(ncep_param['land'])[0,:,:]
iland = np.flipud(iland)
ilandzero = iland[:,0]
iland = np.hstack((iland,ilandzero.reshape(-1,1))) # 画面端の処理のため、経度193番目に経度0番を入れる

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



# prev.climate7902.nc2.big読み込み\n",
# 01_ESTOC_ForcingData/NCEP_NCAR_Forcing/2017/4.fwat/inc/prev_climate7902nc2.big\n",
# 縦ｘ横(94 x 192)が12枚あるイメージ\n",
# 12ヶ月分の何かしらの気候値\n",
#clima = np.fromfile(prevclimate,dtype='float32').reshape(12,94,192)
## 使わない判断をした(4/4)


# 読み込んだデータはフリップさせる(4/4)


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
addzero = fresh[:,:,0].reshape(2664,94,1) # 0番目の列を取り出して2次元＞3次元に変換
fresh193 = np.append(fresh,addzero,axis=2) # axis=0奥行き方向、1行方向、2列方向


# In[5]:


## NCEPデータは読んだら必ずフリップ(4/4)
## 必要とされるデータは南が0で上になっているため
index = 0
while index < 2664:
    fresh193[index,:,:] = np.flipud(fresh193[index,:,:])
    index += 1


# In[6]:


# ilandで陸地だったらfreshの同位置を０にする\n",
# bool indexで配列(landmask)が返る\n",
landmask = (iland >= 1)
# True の部分（地上）を０にする\n",
index = 0
while index < 2664:
    fresh193[index,:,:][landmask == True] = 0
    index += 1

### この時点でlandmaskは(94*193)


# In[7]:


# freshのサイズをESTOCに合わせる内挿処理
readdata = xr.DataArray(fresh193,dims=['time','lat','lon']) # ファイルを読み込んでXArrayに入れる\n",
# dim_0 10日平均 dim_1 lat緯度 dim_2 lon経度\n",

days = 0
temp = []

while days < 2664: # 2664
    slicedays = readdata[days,:,:]
    
    zi = interpolate.interp2d(orig_lon,orig_lat,slicedays,kind='linear')(xi[:,0],yi[0,:])
    
    interp_fresh = np.append(temp,zi).reshape([days+1,180,360])

    temp = interp_fresh

    days += 1
    # debug print
    if days % 100 == 0:
        print(days)

print('end')


# In[8]:


## debug用にinterp_freshはファイル保存する
## 内挿処理に時間がかかるので
np.save('./interp_fresh.npy',interp_fresh)


# In[7]:


## debug用interp_freshからロードして以降で使用する
interp_fresh = np.load('./interp_fresh.npy')


# In[8]:


# 画面端の処理をするために横に1列増やす、左右必要
addzero = interp_fresh[:,:,0].reshape(2664,180,1) # 0番目の列の値を抜き出して3次元に直す
add359 = interp_fresh[:,:,359].reshape(2664,180,1) # 360番目の値を抜き出して3次元に直す

fresh362 = np.append(add359,(np.append(interp_fresh,addzero,axis=2)),axis=2)


# In[9]:


# ESTOCランドマスク作成
# （155 x 360) 75S-80N 上下にダミーの海を足す

# 元データは75Sから80Nまでなので上下90まで陸地0で埋める
top = np.full((15,360),0)
bottom = np.full((10,360),0)

# ESTOCマスクデータを読みこむ（元データは天地逆だがフリップさせてはいない）
data = np.loadtxt('kmt_data.txt',dtype='int') # 元がテキストなのでキャストも必要
estocmask = data
# 上下にダミーの海を足す
estocmask = np.append(top,estocmask,axis=0)
estocmask = np.append(estocmask,bottom,axis=0)

estoclandmask = (estocmask == 0)


# In[10]:


# NCEPランドマスク作成
# オリジナルは海が0なので陸地を0に反転させる
# NCEPデータの陸地を0にする
# iland９２＊１９３

rev_iland = np.where(iland == 0 ,1,0)


# In[11]:


# ESTOCランドマスクに合わせる内挿処理

ncepmask = interpolate.interp2d(orig_lon,orig_lat,rev_iland,kind='linear')(xi[:,0],yi[0,:])


# In[12]:


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
## estocweighも０番と
estocweight = np.where(estoclandmask == True,0,1)
checkweight = estocweight
#######
## mask の中身はBool
mask = ((estocweight == 1) & (ncepmask < 1-10**-10)) 
landindex =  np.where(mask == 1)
######

fresh362_old = fresh362.copy() # 比較用コピー作成 pythonは＝だけだとメモリ共通なので元も代わってしまう

###

####
## landindexの中身０から(mask==0として確認　5/30)
####
for i in enumerate(landindex[0]): # i = loop index
    lat = (landindex[0][i[0]])
    lon = (landindex[1][i[0]])
    estocweight[lat,lon] = 0 # 6/26 =0に戻す
    checkweight[lat,lon] = 0 # estocweight確認用コピー
    ## ここのESTOCWeightを確認する6/26
    ## この時点ではmask == checkweight(estocwaight)になっている。
####
####
####
    

##estocweightを拡大する。
##0の左に３５９の値　３６０の右に元の０の値それぞれ値を入れておく（３・２３）
left = estocweight[:,0]
right = estocweight[:,359]

estocweight = np.hstack([(right.reshape(-1,1)),estocweight]) # 0の前に360の値を足す
estocweight = np.hstack([estocweight,(left.reshape(-1,1))]) #360の先に0の値を足す

# この時点でestocweight 180＊362


# In[13]:


## この時点のestocweightをファイル保存
np.save('./estocweight_orig',estocweight)


# In[14]:


# 計算前のfresh362を保存
np.save('./fresh362_orig',fresh362)


# In[13]:


# maskの図
plt.imshow(np.flipud(mask[153:160,283:300]),vmin=-10**-6 ,vmax=10**-6)


# In[14]:


# maskと同位置のestocweightの図
plt.imshow(np.flipud(estocweight[153:160,284:301]),vmin=-10**-6 ,vmax=10**-6)


# In[15]:


check362 = fresh362.copy()
## 計算前のFres362 を出すcheck362


# In[16]:


# 初回穴埋め前の図
plt.imshow((check362[33,153:160,283:300]),vmin=-10**-6 ,vmax=10**-6)


# In[ ]:





# In[17]:


# ファイル作る前にあったらとりあえず消しておく
##ファイルがあるときは一旦削除する
if os.path.isfile('fwflux10dy-1948-2021.dat'):
    os.remove('fwflux10dy-1948-2021.dat')


# In[ ]:


# landindexは陸地(landmask==True)の座標がタプルのndarray(配列）で返る
# landindex[0] = latitude
# landindex[1] = longitude

estocweight_old = np.load('./estocweight_orig.npy')
# 更新されないestocweight必要
#estocweight_zero = np.load('./estocweight_orig.npy')

for index in range(2664): # ループが遅いのでテストで1年分だけ出してみる、本当は74年分で2664

    estocweight = np.load('./estocweight_orig.npy')# ループを抜けたら毎回初期状態に戻す。
    estocweight_old = np.load('./estocweight_orig.npy') # こっちも毎回初期状態に戻す

    #np.save(str(index),fresh362)
    if index % 100 == 0:
        print(index)
    
    for counter in range(1000):# 1000回繰り返す（又は閾値を超えたらループ終了
        resmax = -10000000000
        fresh362_old[index,:,:] = fresh362[index,:,:] ## コレも必要
        estocweight_old[:,:] = estocweight[:,:] # フラグ１の場所を更新
        
        ## estocweight0,360 に更新（右と左両方追加、3/23)
        left = estocweight[:,1]
        right = estocweight[:,360]
        estocweight_old[:,0] = right # 0の前に360の値を更新
        estocweight_old[:,361] = left #360の先に1の値を更新
        ## 同じことをFresh362でもやる
        fresh362_old[index,:,0] = fresh362[index,:,360]
        fresh362_old[index,:,361] = fresh362[index,:,1]
        
        ## ここがズレてる
        for i in enumerate(landindex[0]): # i = loop index
            lat = (landindex[0][i[0]])
            lon = (landindex[1][i[0]])+1 # 6/9
            ## ループの1回目は０であるはず、ESTOCWEIGHT lat,lon
            # 1箇所だけlon=360がある
            
            ## latも値を確認0、179があるとだめ
            calcflag = estocweight_old[lat-1,lon] + estocweight_old[lat+1,lon] + estocweight_old[lat,lon-1] + estocweight_old[lat,lon+1] 
            #if lon != 360:
            calcdata = (fresh362_old[index,lat-1,lon]*estocweight_old[lat-1,lon] + fresh362_old[index,lat+1,lon]*estocweight_old[lat+1,lon]                            +fresh362_old[index,lat,lon-1]*estocweight_old[lat,lon-1] + fresh362_old[index,lat,lon+1]*estocweight_old[lat,lon+1])
                
            if calcflag > 0:
                calcweight = calcdata/(calcflag)
                res = abs(fresh362_old[index,lat,lon]-calcweight)
                fresh362[index,lat,lon] = calcweight
                estocweight[lat,lon] = 1
                resmax = max(resmax,res)
                
        if 0.0000001 > resmax:
            break

fresh362[:,:,1:361].tofile('fwflux10dy-1948-2021.dat')
print('end')

# In[19]:


plt.imshow((fresh362[33,153:160,283:300]),vmin=-10**-6 ,vmax=10**-6)


# In[37]:


## ガベージコレクタ
del interp_fresh,fresh,fresh193,fresh362,fresh362_old,slicedays
gc.collect()

### Momentum fluxの計算
### uflux10dy.dat vflux10dy.dataを使う


# In[17]:


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


# In[18]:


## uflux,vfluxの193番目に0番を追加する、画面端の処理のため

addzero = uflux[:,:,0].reshape(2664,94,1)
uflux193 = np.append(uflux,addzero,axis=2)

addzero = vflux[:,:,0].reshape(2664,94,1)
vflux193 = np.append(vflux,addzero,axis=2)


# In[19]:


## NCEPデータは読んだら必ずフリップ(4/4)
## 必要とされるデータは南が0で上になっている
index = 0
while index < 2664:
    uflux193[index,:,:] = np.flipud(uflux193[index,:,:])
    index += 1

index = 0
while index < 2664:
    vflux193[index,:,:] = np.flipud(vflux193[index,:,:])
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


# ファイル作る前にあったらとりあえず消しておく
##ファイルがあるときは一旦削除する
if os.path.isfile('nc1ex1deg.vflx10dy.1948-2021.dat'):
    os.remove('nc1ex1deg.vflx10dy.1948-2021.dat')


# In[22]:


## 風応力のランドシーマスクでの穴埋めは不必要と判断されたので書き出し(4/26)
## 以下の1000回ループ計算を削除して書き出し
interp_vflx.tofile('nc1ex1deg.vflx10dy.1948-2021.dat')
print('end')


# In[23]:


## gabege collector
del add359,addzero,slicedays,vflux,vflux193
gc.collect()


# In[24]:


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


# In[25]:


# ファイル作る前にあったらとりあえず消しておく
##ファイルがあるときは一旦削除する
if os.path.isfile('nc1ex1deg.uflx10dy.1948-2021.dat'):
    os.remove('nc1ex1deg.uflx10dy.1948-2021.dat')


# In[26]:


## 風応力の場合はランドシーマスクでの穴埋めはやらない（4/26)
## 1000回ループ計算がなくなるので書き出して終了
interp_uflx.tofile('nc1ex1deg.uflx10dy.1948-2021.dat')
print('end')


# In[27]:


#ガベージコレクタ\n",
del interp_uflx,uflux193,uflux,uflx,slicedays
gc.collect()

### Net heat fluxの計算


# In[28]:


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


# In[29]:


# landmask

# Net heat fluxは以下の式
## ESTOC用に単位換算
gh = ((dsr - usr + dlr - ulr) - (sh + lh)) / 4.184 / 1*10**4

# 193番目を作り0番目の値を入れておく（他も同様、画面端処理のため）
addzero = gh[:,:,0].reshape(2664,94,1)
gh193 = np.append(gh,addzero,axis=2)

## NCEPデータは読んだら必ずフリップ(4/4)
## 必要とされるデータは南が0で上になっている
index = 0
while index < 2664:
    gh193[index,:,:] = np.flipud(gh193[index,:,:])
    index += 1


# ilandで陸地だったらfreshの同位置を０にする\n",
# bool indexで配列(landmask)が返る\n",
landmask = (iland >= 1)
# True の部分（地上）を０にする\n",
index = 0
while index < 2664:
    gh193[index,:,:][landmask == True] = 0
    index += 1


# In[30]:


## 内挿の座標値を元に戻す（vflux,ufluxだけ座標値が変わる）
xi,yi = np.mgrid[0.5:360:1,-89.5:90:1]


# In[31]:


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


# In[32]:


# 画面端の処理をするために360番目の先に0番目を足す
addzero = interp_gh[:,:,0].reshape(2664,180,1)
add359 = interp_gh[:,:,359].reshape(2664,180,1)

gh362 = np.append(add359,(np.append(interp_gh,addzero,axis=2)),axis=2)
   


# In[33]:


gh362_old = gh362.copy() # 計算用コピー作成
# estocweightを初期化
estocweight = np.where(estoclandmask == True,0,1)
##estocweightを拡大する。
##0の左に３５９の値　３６０の右に元の０の値それぞれ値を入れておく（３・２３）
left = estocweight[:,0]
right = estocweight[:,359]

estocweight = np.hstack([(right.reshape(-1,1)),estocweight]) # 0の前に360の値を足す
estocweight = np.hstack([estocweight,(left.reshape(-1,1))]) #360の先に0の値を足す


# In[36]:


# ファイル作る前にあったらとりあえず消しておく
##ファイルがあるときは一旦削除する
if os.path.isfile('nc1ex1deg.heatf10dy.1948-2021.dat'):
    os.remove('nc1ex1deg.heatf10dy.1948-2021.dat')


# In[ ]:


# landindexは陸地(landmask==True)の座標がタプルのndarray(配列）で返る
# landindex[0] = latitude
# landindex[1] = longitude

estocweight_old = estocweight.copy()

for index in range(2664): # ループが遅いのでテストで1年分だけ出してみる、本当は74年分で2664
    for counter in range(1000):# 1000回繰り返す（又は閾値を超えたらループ終了
        resmax = -10000000000
        gh362_old[index,:,:] = gh362[index,:,:] ## コレも必要
        estocweight_old[:,:] = estocweight[:,:]
        ## estocweight0,360 に更新（右と左両方追加、3/23)
        left = estocweight[:,1]
        right = estocweight[:,360]
        estocweight[:,0] = right # 0の前に360の値を更新
        estocweight[:,361] = left #360の先に1の値を更新
        # 同じことをgh362でもやる
        gh362_old[index,:,0] = gh362[index,:,360]
        gh362_old[index,:,361] = gh362[index,:,1]
                
        
        for i in enumerate(landindex[0]): # i = loop index
            lat = (landindex[0][i[0]])
            lon = (landindex[1][i[0]])
            
            calcflag = estocweight_old[lat-1,lon] + estocweight_old[lat+1,lon] + estocweight_old[lat,lon-1] + estocweight_old[lat,lon+1] 
            
            calcdata= (gh362_old[index,lat-1,lon]*estocweight_old[lat-1,lon]+gh362_old[index,lat+1,lon]*estocweight_old[lat+1,lon]                       +gh362_old[index,lat,lon-1]*estocweight_old[lat,lon-1]+gh362_old[index,lat,lon+1]*estocweight_old[lat,lon+1])
                
                
            if calcflag > 0:
                calcweight = calcdata/calcflag
                res = abs(gh362_old[index,lat,lon] - calcweight)
                gh362[index,lat,lon] = calcweight
                estocweight[lat,lon] = 1
                resmax = max(resmax,res)
            
            # 差分が範囲超えたら抜ける
        if 0.1 > resmax:
            print('index=',index,'counter=',counter)
            break
        
gh362[:,:,1:361].tofile('nc1ex1deg.heatf10dy.1948-2021.dat')
print('end')


# In[54]:


#ガベージコレクタ\n",
del dlr,ulr,sh,lh,interp_gh,gh362,gh362_old,gh,gh193
gc.collect()
### Net solar fluxの計算


# In[15]:


## net solar flux
## 単位換算
snr = (dsr - usr) / 4.186 / 1*10**4

# 193番目を作り0番目の値を入れておく（他も同様、画面端処理のため）
addzero = snr[:,:,0].reshape(2664,94,1)
snr193 = np.append(snr,addzero,axis=2)

## NCEPデータは読んだら必ずフリップ(4/4)
## 必要とされるデータは南が0で上になっている
index = 0
while index < 2664:
    snr193[index,:,:] = np.flipud(snr193[index,:,:])
    index += 1


# ilandで陸地だったらfreshの同位置を０にする\n",
# bool indexで配列(landmask)が返る\n",
landmask = (iland >= 1)
# True の部分（地上）を０にする\n",
index = 0
while index < 2664:
    snr193[index,:,:][landmask == True] = 0
    index += 1


# In[16]:


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


# In[17]:


# 画面端の処理をするために360番目の先に0番目の値を足す
# 0番の前に359番目の値を足す
addzero = interp_snr[:,:,0].reshape(2664,180,1)
add359 = interp_snr[:,:,359].reshape(2664,180,1)

snr362 = np.append(add359,(np.append(interp_snr,addzero,axis=2)),axis=2)


# In[18]:


snr362_old = snr362.copy() # 計算用コピー作成
# estocweightを初期化
estocweight = np.where(estoclandmask == True,0,1)
##estocweightを拡大する。
##0の左に３５９の値　３６０の右に元の０の値それぞれ値を入れておく（３・２３）
left = estocweight[:,0]
right = estocweight[:,359]

estocweight = np.hstack([(right.reshape(-1,1)),estocweight]) # 0の前に360の値を足す
estocweight = np.hstack([estocweight,(left.reshape(-1,1))]) #360の先に0の値を足す


# In[19]:


# ファイル作る前にあったらとりあえず消しておく
##ファイルがあるときは一旦削除する
if os.path.isfile('nc1ex1deg.snr10dy.1948-2021.dat'):
    os.remove('nc1ex1deg.snr10dy.1948-2021.dat')


# In[ ]:


# landindexは陸地(landmask==True)の座標がタプルのndarray(配列）で返る
# landindex[0] = latitude
# landindex[1] = longitude

estocweight_old = estocweight.copy()

for index in range(2664): # ループが遅いのでテストで1年分だけ出してみる、本当は74年分で2664
    for counter in range(1000):# 1000回繰り返す（又は閾値を超えたらループ終了
        resmax = -10000000000
        snr362_old[index,:,:] = snr362[index,:,:] ## コレも必要
        estocweight_old[:,:] = estocweight[:,:]
        ## estocweight0,360 に更新（右と左両方追加、3/23)
        left = estocweight[:,1]
        right = estocweight[:,360]
        estocweight[:,0] = right # 0の前に360の値を更新
        estocweight[:,361] = left #360の先に1の値を更新
        # 同じことをsnr362でもやる
        snr362_old[index,:,0] = snr362[index,:,360]
        snr362_old[index,:,361] = snr362[index,:,1]
        
        for i in enumerate(landindex[0]): # i = loop index
            lat = (landindex[0][i[0]])
            lon = (landindex[1][i[0]])
            
            calcflag = estocweight_old[lat-1,lon] + estocweight_old[lat+1,lon] + estocweight_old[lat,lon-1] + estocweight_old[lat,lon+1] 
            calcdata = snr362_old[index,lat-1,lon]*estocweight_old[lat-1,lon]+snr362_old[index,lat+1,lon]*estocweight_old[lat+1,lon]                       +snr362_old[index,lat,lon-1]*estocweight_old[lat,lon-1]+snr362_old[index,lat,lon+1]*estocweight_old[lat,lon+1]
                           
                
            if calcflag > 0:
                calcweight = calcdata/calcflag
                res = abs(snr362_old[index,lat,lon] - calcweight)
                snr362[index,lat,lon] = calcweight
                estocweight[lat,lon] = 1
                resmax = max(resmax,res)
                    

        # 差分が範囲超えたら抜ける
        if 0.5 > resmax:
            print('index=',index,'counter=',counter)
            break

snr362[:,:,:360].tofile('nc1ex1deg.snr10dy.1948-2021.dat')

print('end')


# In[ ]:


### ファイル変換ができたら、単位を合わせるNCEP > ESTOC
### SNR,VFLX,UFLXはグリーンランドの上はデータ更新されているか？
### 南極の近くはブロック状になっているので元データを見てみる。
#ヒートフラックがみやすい、秋～11月くらいを見てみる


# In[ ]:


## 以下デバック用セル


# In[1]:





fig,ax = plt.subplots()
aximg = ax.imshow(np.flipud(mask[130:170,275:315]),vmin=-10**-6 ,vmax=10**-6)
fig.colorbar(aximg,ax = ax)
plt.show()


# In[ ]:


plt.imshow(np.flipud(fresh362[32,130:170,275:315]),vmin=-10**-6 ,vmax=10**-6)


# In[ ]:


print(fresh362[32,130:170,300]*10**6)


# In[ ]:


print(interp_fresh[32,130:170,299]*10**6)


# In[ ]:


plt.imshow(np.flipud((1-mask[130:170,275:315])*fresh362[32,130:170,275:315]))


# In[ ]:


plt.imshow(np.flipud(mask[130:170,275:315]*fresh362[32,130:170,275:315]))
plt.savefig('1.png')


# In[ ]:


plt.imshow(np.flipud(fresh362[32,130:170,275:315]))


# In[ ]:


plt.imshow(np.flipud((ncepmask[130:170,275:315] < 1-10**-10)))


# In[ ]:


plt.imshow(np.flipud(estoclandmask[130:170,275:315]))


# In[ ]:


estocweight_old = estocweight.copy()
index = 32
resmax = -10000000000
fresh362_old[index,:,:] = fresh362[index,:,:] ## コレも必要
estocweight_old[:,:] = estocweight[:,:]
        
## estocweight0,360 に更新（右と左両方追加、3/23)
left = estocweight[:,1]
right = estocweight[:,360]
estocweight_old[:,0] = right # 0の前に360の値を更新
estocweight_old[:,361] = left #360の先に1の値を更新
## 同じことをFresh362でもやる
fresh362_old[index,:,0] = fresh362[index,:,360]
fresh362_old[index,:,361] = fresh362[index,:,1]
        
for i in enumerate(landindex[0]): # i = loop index
    lat = (landindex[0][i[0]])
    lon = (landindex[1][i[0]])+1
            
# 1箇所だけlon=360がある
## latも値を確認0、179があるとだめ
    calcflag = estocweight_old[lat-1,lon] + estocweight_old[lat+1,lon] + estocweight_old[lat,lon-1] + estocweight_old[lat,lon+1] 
    #if lon != 360:
    calcdata = (fresh362_old[index,lat-1,lon]*estocweight_old[lat-1,lon] + fresh362_old[index,lat+1,lon]*estocweight_old[lat+1,lon]                +fresh362_old[index,lat,lon-1]*estocweight_old[lat,lon-1] + fresh362_old[index,lat,lon+1]*estocweight_old[lat,lon+1])
                
    if calcflag > 0:
        calcweight = calcdata/(calcflag)
        res = abs(fresh362_old[index,lat,lon]-calcweight)
        fresh362[index,lat,lon] = calcweight
        estocweight[lat,lon] = 1
        resmax = max(resmax,res)
    

plt.imshow(np.flipud(fresh362[32,130:170,275:315]))


# In[ ]:


plt.imshow(np.flipud(ncepmask[130:170,275:315]))


# In[22]:


plt.imshow(np.flipud(estocweight))


# In[ ]:


plt.imshow(np.flipud(mask[130:170,275:315]*fresh362[32,130:170,275:315]))
plt.savefig('2.png')


# In[19]:


first = np.load('./0.npy')
plt.imshow((first[33,153:160,283:300]),vmin=-10**-6 ,vmax=10**-6)


# In[20]:


hoge = np.load('./33.npy')
plt.imshow((hoge[33,153:160,283:300]),vmin=-10**-6 ,vmax=10**-6)


# In[22]:


hoe = np.load('./34.npy')
plt.imshow((hoe[33,153:160,283:300]),vmin=-10**-6 ,vmax=10**-6)


# In[21]:



print(np.all(orig == estocweight_old))


# In[20]:


orig = np.load('./estocweight_orig.npy')
plt.imshow(orig)


# In[21]:


plt.imshow(estocweight)


# In[ ]:




