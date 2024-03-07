#!/usr/bin/env python
# coding: utf-8

# In[1]:


## freshwaterの流れとは微妙に違うところがあったので、まるっきりFreshwaterに処理をあわせて作成してみるテスト

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
import multiprocessing as mp

# 緯度経度は実際の数値を入れる
# 元データNCファイルの緯度経度の範囲を得る
ncep_param = xr.open_dataset('01_ESTOC_ForcingData/NCEP_NCAR_Forcing/NCEP/land.sfc.gauss.nc')

# ncepからのデータは読んだら必ずフリップ(4/4)
# 見た目、上下逆になるが提出データは上が０、上が南になるのが正しい
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

##   データの範囲、１９４８年１月〜２０２３年１月、１０日平均で2703期間
yearnum = 2703


# In[2]:


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

# NCEPランドマスク作成
# オリジナルは海が0なので陸地を0に反転させる
# NCEPデータの陸地を0にする
# iland９２＊１９３

rev_iland = np.where(iland == 0 ,1,0)


# NCEPランドマスクをESTOCランドマスクに合わせる内挿処理

ncepmask = interpolate.interp2d(orig_lon,orig_lat,rev_iland,kind='linear')(xi[:,0],yi[0,:])


# In[6]:


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
#######
## mask の中身はBool
mask = ((estocweight == 1) & (ncepmask < 1-10**-10)) 
landindex =  np.where(mask == 1)
######

## landindexの中身０から(mask==0として確認　5/30)
####
for i in enumerate(landindex[0]): # i = loop index
    lat = (landindex[0][i[0]])
    lon = (landindex[1][i[0]])
    estocweight[lat,lon] = 0 # 6/26 =0に戻す
    ## ここのESTOCWeightを確認する6/26
    ## この時点ではmask == estocwaightになっている。
####

##estocweightを拡大する。
##0の左に３５９の値　３６０の右に元の０の値それぞれ値を入れておく（３・２３）
left = estocweight[:,0]
right = estocweight[:,359]

estocweight = np.hstack([(right.reshape(-1,1)),estocweight]) # 0の前に360の値を足す
estocweight = np.hstack([estocweight,(left.reshape(-1,1))]) #360の先に0の値を足す

# この時点でestocweight 180＊362
estocweight_old = estocweight.copy() # 計算用に必要なので作っておく
estocweight_orig = estocweight.copy()


# In[7]:


ncep_param = xr.open_dataset('01_ESTOC_ForcingData/NCEP_NCAR_Forcing/NCEP/land.sfc.gauss.nc')

# NCEPの海陸マスクで陸の値で海の値を埋める処理
#　UFlux/VFluxは穴埋め計算のかわりにこれを行ってからESTOCサイズに合わせる
# 必要なNCEPの海陸マスク作成
lsmsk = np.array(ncep_param['land'])[0,:,:]
lsmsk = np.flipud(lsmsk)

# 陸１、海０になっているので反転させる
lsmsk = 1 - lsmsk

## この時点でデータは上が南


# In[ ]:





# In[8]:


## ESTOCマスク専用を用意するxi,yiが変わる
## uflx,vflxはESTOCマスクを変更する
## mgridを公差ではなく項数にするためにjをつける
xi,yi = np.mgrid[1:360:360j,-89:90:180j]


# uflx
uflx = np.fromfile('uflx10dy.dat').reshape(yearnum,94,192)
# vflx
vflx = np.fromfile('vflx10dy.dat').reshape(yearnum,94,192)

# landmask
# fresh water では計算式があったが、momentum fluxでは無い


# In[9]:


# *10は単位換算
uflux = uflx * 10
vflux = vflx * 10

#　０番目の列をコピーして１９３番目に追加する、画面端の処理
addzero = uflux[:,:,0].reshape(yearnum,94,1) # 0番目の列を取り出して2次元＞3次元に変換
uflux193 = np.append(uflux,addzero,axis=2) # axis=0奥行き方向、1行方向、2列方向

#同上
addzero = vflux[:,:,0].reshape(yearnum,94,1) # 0番目の列を取り出して2次元＞3次元に変換
vflux193 = np.append(vflux,addzero,axis=2) # axis=0奥行き方向、1行方向、2列方向


# In[ ]:





# In[ ]:





# In[10]:


## uflx10dy,vflx10dyデータは北が上なので読んだらフリップ
## 提出データは南が0で上になっている

## NCEPデータは読んだら必ずフリップ(4/4)
## 提出データは南が0で上になっているため
index = 0
while index < yearnum:
    vflux193[index,:,:] = np.flipud(vflux193[index,:,:])
    index += 1
    


# In[11]:


## NCEPデータは読んだら必ずフリップ(4/4)
## 提出データは南が0で上になっているため
index = 0
while index < yearnum:
    uflux193[index,:,:] = np.flipud(uflux193[index,:,:])
    index += 1


# In[ ]:





# In[12]:


# ilandで陸地だったらfreshの同位置を０にする\n",
# bool indexで配列(landmask)が返る\n",
landmask = (iland >= 1)
# True の部分（地上）を０にする\n",
index = 0
while index < yearnum:
    vflux193[index,:,:][landmask == True] = 0
    index += 1

### この時点でlandmaskは(94*193)



# In[13]:


# vfluxを内挿してESTOCサイズに合わせる。
readdata = xr.DataArray(vflux193,dims=['time','lat','lon']) # ファイルを読み込んでXArrayに入れる\n",
# dim_0 10日平均 dim_1 lat緯度 dim_2 lon経度\n",

def process_day(day):
    slicedays = readdata[day,:,:]
    zi = interpolate.interp2d(orig_lon,orig_lat,slicedays,kind='linear')(xi[:,0],yi[0,:])
    return zi

if __name__ == '__main__':
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.map(process_day, range(yearnum)) # 2703(1948/01-2023/01)

    interp_vflx = np.array(results).reshape([yearnum,180,360])

print('end')


# In[14]:


# debug用の中間ファイル保存

if os.path.isfile('interp_vflx.npy'):
    os.remove('interp_vflx.npy')

np.save('interp_vflx',interp_vflx)


# In[15]:


# 画面端の処理をするために横に1列増やす、左右必要
addzero = interp_ｖｆｌｘ[:,:,0].reshape(yearnum,180,1) # 0番目の列の値を抜き出して3次元に直す
add359 = interp_ｖｆｌｘ[:,:,359].reshape(yearnum,180,1) # 360番目の値を抜き出して3次元に直す

vflux362 = np.append(add359,(np.append(interp_vflx,addzero,axis=2)),axis=2)


# In[16]:


# landindexは陸地(landmask==True)の座標がタプルのndarray(配列）で返る
# landindex[0] = latitude
# landindex[1] = longitude
vflux362_old = vflux362.copy() # 比較用コピー作成 pythonは＝だけだとメモリ共通なので元も代わってしまうため.copy()をつけて別のメモリにコピー


for index in range(2703): # ループが遅いのでテストで1年分だけ出してみる、本当は75年分で2703

    estocweight = estocweight_orig.copy() # 10/2
    
    

    for counter in range(1000):# 1000回繰り返す（又は閾値を超えたらループ終了 

        resmax = -10000000000
        vflux362_old[index,:,:] = vflux362[index,:,:] ## コレも必要
        
        estocweight_old[:,:] = estocweight[:,:] # フラグ１の場所を更新
        
        ## estocweight0,360 に更新（右と左両方追加、3/23)
        left = estocweight[:,1]
        right = estocweight[:,360]
        estocweight_old[:,0] = right # 0の前に360の値を更新
        estocweight_old[:,361] = left #360の先に1の値を更新
        ## 同じことをFresh362でもやる
        vflux362_old[index,:,0] = vflux362[index,:,360]
        vflux362_old[index,:,361] = vflux362[index,:,1]
        
        ## ここがズレてる
        for i in enumerate(landindex[0]): # i = loop index
            lat = (landindex[0][i[0]])
            lon = (landindex[1][i[0]])+1 # 6/9
            ## ループの1回目は０であるはず、ESTOCWEIGHT lat,lon
            # 1箇所だけlon=360がある
            
            ## latも値を確認0、179があるとだめ
            calcflag = estocweight_old[lat-1,lon] + estocweight_old[lat+1,lon] + estocweight_old[lat,lon-1] + estocweight_old[lat,lon+1] 
            #if lon != 360:
            calcdata = (vflux362_old[index,lat-1,lon]*estocweight_old[lat-1,lon] + vflux362_old[index,lat+1,lon]*estocweight_old[lat+1,lon]\
                            +vflux362_old[index,lat,lon-1]*estocweight_old[lat,lon-1] + vflux362_old[index,lat,lon+1]*estocweight_old[lat,lon+1] )

                
            if calcflag > 0:
                calcweight = calcdata/(calcflag)
                res = abs(vflux362_old[index,lat,lon]-calcweight)
                vflux362[index,lat,lon] = calcweight
                estocweight[lat,lon] = 1
                resmax = max(resmax,res)

            
        if 0.0000001 > resmax:
            print('break count =',counter)
            break

## ファイル書き出しを別セルにする（遅いので）
print('end')


# In[17]:


## ファイル出来上がり、陸地に−１，０E33を入れてビッグエンディアンバイナリで書き出して提出用になる
vflux_finish = vflux362[:,:,1:361]

# 　-1.0e33とビッグエンディアンバイナリは下のセルでやっている


# In[18]:





# In[19]:


if os.path.isfile('vflux_finish.npy'):
    os.remove('vflux_finish.npy')

np.save('vflux_finish.npy',vflux_finish)

## vflux終わり




## 以下uflux


# In[23]:


# ilandで陸地だったらfreshの同位置を０にする\n",
# bool indexで配列(landmask)が返る\n",
landmask = (iland >= 1)
# True の部分（地上）を０にする\n",
index = 0
while index < yearnum:
    uflux193[index,:,:][landmask == True] = 0
    index += 1

### この時点でlandmaskは(94*193)


# In[24]:


# uflxをESTOCサイズに合わせるため内挿する
readdata = xr.DataArray(uflux193,dims=['time','lat','lon']) # ファイルを読み込んでXArrayに入れる\n",
# dim_0 10日平均 dim_1 lat緯度 dim_2 lon経度\n",

def process_day(day):
    slicedays = readdata[day,:,:]
    zi = interpolate.interp2d(orig_lon,orig_lat,slicedays,kind='linear')(xi[:,0],yi[0,:])
    return zi

if __name__ == '__main__':
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.map(process_day, range(yearnum)) # 2703(1948/01-2023/01)

    interp_uflx = np.array(results).reshape([yearnum,180,360])

print('end')


# In[25]:


# debug用の中間ファイル保存

if os.path.isfile('interp_uflx.npy'):
    os.remove('interp_uflx.npy')

np.save('interp_uflx',interp_uflx)


# In[26]:


# 画面端の処理をするために横に1列増やす、左右必要
addzero = interp_uｆｌｘ[:,:,0].reshape(yearnum,180,1) # 0番目の列の値を抜き出して3次元に直す
add359 = interp_uｆｌｘ[:,:,359].reshape(yearnum,180,1) # 360番目の値を抜き出して3次元に直す

uflux362 = np.append(add359,(np.append(interp_uflx,addzero,axis=2)),axis=2)


# In[29]:





# In[ ]:





# In[ ]:





# In[ ]:


## 一応estocweightフラグを初期化する


# In[30]:


# landindexは陸地(landmask==True)の座標がタプルのndarray(配列）で返る
# landindex[0] = latitude
# landindex[1] = longitude
uflux362_old = uflux362.copy() # 比較用コピー作成 pythonは＝だけだとメモリ共通なので元も代わってしまうため.copy()をつけて別のメモリにコピー


for index in range(2703): # ループが遅いのでテストで1年分だけ出してみる、本当は75年分で2703

    estocweight = estocweight_orig.copy() # 10/2

    
    for counter in range(1000):# 1000回繰り返す（又は閾値を超えたらループ終了 
        resmax = -10000000000
        uflux362_old[index,:,:] = uflux362[index,:,:] ## コレも必要
        
        estocweight_old[:,:] = estocweight[:,:] # フラグ１の場所を更新
        
        ## estocweight0,360 に更新（右と左両方追加、3/23)
        left = estocweight[:,1]
        right = estocweight[:,360]
        estocweight_old[:,0] = right # 0の前に360の値を更新
        estocweight_old[:,361] = left #360の先に1の値を更新
        ## 同じことをFresh362でもやる
        uflux362_old[index,:,0] = uflux362[index,:,360]
        uflux362_old[index,:,361] = uflux362[index,:,1]
        
        ## ここがズレてる
        for i in enumerate(landindex[0]): # i = loop index
            lat = (landindex[0][i[0]])
            lon = (landindex[1][i[0]])+1 # 6/9
            ## ループの1回目は０であるはず、ESTOCWEIGHT lat,lon
            # 1箇所だけlon=360がある
            
            ## latも値を確認0、179があるとだめ
            calcflag = estocweight_old[lat-1,lon] + estocweight_old[lat+1,lon] + estocweight_old[lat,lon-1] + estocweight_old[lat,lon+1] 
            #if lon != 360:
            calcdata = (uflux362_old[index,lat-1,lon]*estocweight_old[lat-1,lon] + uflux362_old[index,lat+1,lon]*estocweight_old[lat+1,lon]\
                            +uflux362_old[index,lat,lon-1]*estocweight_old[lat,lon-1] + uflux362_old[index,lat,lon+1]*estocweight_old[lat,lon+1] )

                
            if calcflag > 0:
                calcweight = calcdata/(calcflag)
                res = abs(uflux362_old[index,lat,lon]-calcweight)
                uflux362[index,lat,lon] = calcweight
                estocweight[lat,lon] = 1
                resmax = max(resmax,res)

                
        if 0.0000001 > resmax:
            break

## ファイル書き出しを別セルにする（遅いので）
print('end')


# In[31]:


## 1:361で抜き出して、陸地に-1.0e33を入れてビッグエンディアンバイナリで書き出して終わり

uflux_finish = uflux362[:,:,1:361]


# In[13]:





# In[ ]:





# In[20]:





# In[ ]:





# In[32]:


## estocのmaskデータを作る
# ESTOCランドマスク作成
# （155 x 360) 75S-80N 上下にダミーの海を足す

# 元データは75Sから80Nまでなので上下90まで陸地0で埋める
top = np.full((15,360),0)
bottom = np.full((10,360),0)

# ESTOCマスクデータを読みこむ（元データは天地逆だがフリップさせてはいない）
data = np.loadtxt('kmt_data.txt',dtype='int') # 元がテキストなのでキャストも必要
mask = data
# 上下にダミーの海を足す
mask = np.append(top,mask,axis=0)
mask = np.append(mask,bottom,axis=0)

landmask = (mask == 0)

#landmask は陸地True,海FalseのBool配列なので海1、陸0に直す
estocmask = np.where(landmask == True,0,1)
#######


# In[ ]:





# In[33]:


uflux_finish[:,estocmask == 0] = -1.0e33
vflux_finish[:,estocmask == 0] = -1.0e33


# In[ ]:





# In[34]:


temp = uflux_finish.byteswap()

if os.path.isfile('uflux_big.bin'):
    os.remove('uflux_big.bin')
    
with open('uflux_big.bin','wb') as f:
    temp.tofile(f)


# In[ ]:





# In[22]:


temp = vflux_finish.byteswap()

if os.path.isfile('vflux_big.bin'):
    os.remove('vflux_big.bin')
    
with open('vflux_big.bin','wb') as f:
    temp.tofile(f)


# In[11]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


## 以下デバッグセル


# In[ ]:





# In[ ]:




