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


## プロットして見やすいようにファイルを読み込んだらflipudで上下反転させている
##　あとから追加読み込みしたファイルでは忘れがちなので注意すること


# In[ ]:


##　vflux,ufluxは処理が違うので別ファイルでやることにした（２０２３・１２・１９）


# In[ ]:





# In[3]:


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


# In[ ]:





# In[ ]:





# In[4]:


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




# In[6]:


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



# In[ ]:





# In[5]:


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


# In[6]:


#prate10dy.dat",
prate = np.fromfile('prate10dy.dat').reshape(yearnum,94,192)
# lthfl10dy.dat",
lhtfl = np.fromfile('lhtfl10dy.dat').reshape(yearnum,94,192)


# In[7]:


#ESTOCでは海だけど、NCEPでは陸地の部分
#NCEPで１度にしたときに陸地の場所
# 内挿する前にilandとclima = freshを使ってマスクする必要あり",
# prate と lhtflで要素ごとに計算\n",
# 以下の数式とfreshの命名規則はFotran元コードから"
fresh = (prate - lhtfl/2.5e6)*0.1 # ESTOC用に単位換算

# fresh 193を作り0番の値を入れておく（他も同様、画面端処理のため）
addzero = fresh[:,:,0].reshape(yearnum,94,1) # 0番目の列を取り出して2次元＞3次元に変換
fresh193 = np.append(fresh,addzero,axis=2) # axis=0奥行き方向、1行方向、2列方向


# In[8]:


## NCEPデータは読んだら必ずフリップ(4/4)
## 提出データは南が0で上になっているため
index = 0
while index < yearnum:
    fresh193[index,:,:] = np.flipud(fresh193[index,:,:])
    index += 1


# In[9]:


# ilandで陸地だったらfreshの同位置を０にする\n",
# bool indexで配列(landmask)が返る\n",
landmask = (iland >= 1)
# True の部分（地上）を０にする\n",
index = 0
while index < yearnum:
    fresh193[index,:,:][landmask == True] = 0
    index += 1

### この時点でlandmaskは(94*193)


# In[ ]:





# In[10]:


#各日のデータ処理を個別の関数process_dayに分け、multiprocessing.Pool.mapを使用してそれらを並列に実行する。
#これにより、マルチコアプロセッサの全てのコアを利用して高速化することができる。

# freshのサイズをESTOCに合わせる内挿処理
readdata = xr.DataArray(fresh193,dims=['time','lat','lon']) # ファイルを読み込んでXArrayに入れる\n",
# dim_0 10日平均 dim_1 lat緯度 dim_2 lon経度\n",

## time(year)毎に内挿するので

def process_day(day):
    slicedays = readdata[day,:,:]
    zi = interpolate.interp2d(orig_lon,orig_lat,slicedays,kind='linear')(xi[:,0],yi[0,:])
    return zi

if __name__ == '__main__':
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.map(process_day, range(yearnum)) # 2703(1948/01-2023/01)

    interp_fresh = np.array(results).reshape([yearnum,180,360])

print('end')


# In[ ]:





# In[ ]:





# In[11]:


## debug用にinterp_freshはファイル保存する
## 内挿処理に時間がかかるので
if os.path.isfile('./interp_fresh.npy'):
    os.remove('./interp_fresh.npy')
    
np.save('./interp_fresh.npy',interp_fresh)


# In[ ]:





# In[11]:


# 画面端の処理をするために横に1列増やす、左右必要
# 7/4 全部思惑通りコピーされているか確認
addzero = interp_fresh[:,:,0].reshape(yearnum,180,1) # 0番目の列の値を抜き出して3次元に直す
add359 = interp_fresh[:,:,359].reshape(yearnum,180,1) # 360番目の値を抜き出して3次元に直す

fresh362 = np.append(add359,(np.append(interp_fresh,addzero,axis=2)),axis=2)


# In[ ]:





# In[13]:


# ファイル作る前にあったらとりあえず消しておく
##ファイルがあるときは一旦削除する
if os.path.isfile('fwflux10dy-1948-2023.dat'):
    os.remove('fwflux10dy-1948-2023.dat')

if os.path.isfile('fwflux10dy-10.dat'):
    os.remove('fwflux10dy-10.dat')


# In[17]:


# landindexは陸地(landmask==True)の座標がタプルのndarray(配列）で返る
# landindex[0] = latitude
# landindex[1] = longitude
fresh362_old = fresh362.copy() # 比較用コピー作成 pythonは＝だけだとメモリ共通なので元も代わってしまうため.copy()をつけて別のメモリにコピー


for index in range(2703): # ループが遅いのでテストで1年分だけ出してみる、本当は75年分で2703

    estocweight = estocweight_orig.copy() # 10/2

    
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
            calcdata = (fresh362_old[index,lat-1,lon]*estocweight_old[lat-1,lon] + fresh362_old[index,lat+1,lon]*estocweight_old[lat+1,lon]\
                            +fresh362_old[index,lat,lon-1]*estocweight_old[lat,lon-1] + fresh362_old[index,lat,lon+1]*estocweight_old[lat,lon+1])
                
            if calcflag > 0:
                calcweight = calcdata/(calcflag)
                res = abs(fresh362_old[index,lat,lon]-calcweight)
                fresh362[index,lat,lon] = calcweight
                estocweight[lat,lon] = 1
                resmax = max(resmax,res)

                
        if 0.0000001 > resmax:
            break

## ファイル書き出しを別セルにする（遅いので）
print('end')


# In[ ]:


## 出来たファイルを保存して別ファイルでESTOCの陸に-1.0e33 入れて、ビッグエンディアン変換、バイナリで書き出しをする


# In[14]:


## write file
fwflux = fresh362[:,:,1:361]

if os.path.isfile('fresh_anaume.npy'):
    os.remove('fresh_anaume.npy')

np.save('fresh_anaume',fwflux)


# In[ ]:





# In[26]:


## ガベージコレクタ
del interp_fresh,fresh,fresh193,fresh362,fresh362_old,slicedays
gc.collect()

### Momentum fluxの計算
### uflux10dy.dat vflux10dy.dataを使う


# In[ ]:


##　uflux, vfluxは別ファイルで作成する


# In[6]:


yearnum = 2703
# dswrf
dsr = np.fromfile('dswrf10dy.dat').reshape(yearnum,94,192)
# ulwrf
usr = np.fromfile('uswrf10dy.dat').reshape(yearnum,94,192)
# dlwrf
dlr = np.fromfile('dlwrf10dy.dat').reshape(yearnum,94,192)
# ulwrf
ulr = np.fromfile('ulwrf10dy.dat').reshape(yearnum,94,192)
# shtfl
sh = np.fromfile('shtfl10dy.dat').reshape(yearnum,94,192)
# lhtfl
lh = np.fromfile('lhtfl10dy.dat').reshape(yearnum,94,192)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[7]:


# landmask

# Net heat fluxは以下の式
## ESTOC用に単位換算
gh = ((dsr - usr + dlr - ulr) - (sh + lh)) / 4.186 / 1e4

# 193番目を作り0番目の値を入れておく（他も同様、画面端処理のため）
addzero = gh[:,:,0].reshape(yearnum,94,1)
gh193 = np.append(gh,addzero,axis=2)

## NCEPデータは読んだら必ずフリップ(4/4)
## 必要とされるデータは南が0で上になっている
index = 0
while index < yearnum:
    gh193[index,:,:] = np.flipud(gh193[index,:,:])
    index += 1


# ilandで陸地だったらfreshの同位置を０にする\n",
# bool indexで配列(landmask)が返る\n",
landmask = (iland >= 1)
# True の部分（地上）を０にする\n",
index = 0
while index < yearnum:
    gh193[index,:,:][landmask == True] = 0
    index += 1


# In[8]:


## 内挿の座標値を元に戻す（vflux,ufluxだけ座標値が変わる）
xi,yi = np.mgrid[0.5:360:1,-89.5:90:1]


# In[9]:


# ESTOCのサイズに合わせるため内挿する
readdata = xr.DataArray(gh193,dims=['time','lat','lon']) # ファイルを読み込んでXArrayに入れる\n",
# dim_0 10日平均 dim_1 lat緯度 dim_2 lon経度\n",

def process_day(day):
    slicedays = readdata[day,:,:]
    zi = interpolate.interp2d(orig_lon,orig_lat,slicedays,kind='linear')(xi[:,0],yi[0,:])
    return zi

if __name__ == '__main__':
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.map(process_day, range(yearnum)) # 2703(1948/01-2023/01)

    interp_gh = np.array(results).reshape([yearnum,180,360])

print('end')


# In[10]:


if os.path.isfile('interp_gh.npy'):
    os.remove('interp_gh.npy')

np.save('interp_gh',interp_gh)


# In[ ]:


# debug load
interp_gh = np.fromfile('interp_gh.npy').reshape(2703,180,360)


# In[21]:


# 画面端の処理をするために360番目の先に0番目を足す
addzero = interp_gh[:,:,0].reshape(yearnum,180,1)
add359 = interp_gh[:,:,359].reshape(yearnum,180,1)

gh362 = np.append(add359,(np.append(interp_gh,addzero,axis=2)),axis=2)
   


# In[22]:


gh362_old = gh362.copy() # 計算用コピー作成
# estocweightを初期化
estocweight = np.where(estoclandmask == True,0,1)
##estocweightを拡大する。
##0の左に３５９の値　３６０の右に元の０の値それぞれ値を入れておく（３・２３）
left = estocweight[:,0]
right = estocweight[:,359]

estocweight = np.hstack([(right.reshape(-1,1)),estocweight]) # 0の前に360の値を足す
estocweight = np.hstack([estocweight,(left.reshape(-1,1))]) #360の先に0の値を足す


# In[23]:


# ファイル作る前にあったらとりあえず消しておく
##ファイルがあるときは一旦削除する
if os.path.isfile('nc1ex1deg.heatf10dy.1948-2023.dat'):
    os.remove('nc1ex1deg.heatf10dy.1948-2023.dat')
    
if os.path.isfile('heatf10dy.dat'):
    os.remove('heat10dy.bin')
    


# In[24]:


# landindexは陸地(landmask==True)の座標がタプルのndarray(配列）で返る
# landindex[0] = latitude
# landindex[1] = longitude

for index in range(yearnum): # ループが遅いのでテストで1年分だけ出してみる、本当は75年分で2703
    estocweight = estocweight_orig.copy() # 10/2
    
    
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
            lon = (landindex[1][i[0]])+1
            
            calcflag = estocweight_old[lat-1,lon] + estocweight_old[lat+1,lon] + estocweight_old[lat,lon-1] + estocweight_old[lat,lon+1] 
            
            calcdata= (gh362_old[index,lat-1,lon]*estocweight_old[lat-1,lon]+gh362_old[index,lat+1,lon]*estocweight_old[lat+1,lon]\
                       +gh362_old[index,lat,lon-1]*estocweight_old[lat,lon-1]+gh362_old[index,lat,lon+1]*estocweight_old[lat,lon+1])
                
                
            if calcflag > 0:
                calcweight = calcdata/calcflag
                res = abs(gh362_old[index,lat,lon] - calcweight)
                gh362[index,lat,lon] = calcweight
                estocweight[lat,lon] = 1
                resmax = max(resmax,res)
            
            # 差分が範囲超えたら抜ける
        if 0.0001 > resmax:
            break
        

print('end')


# In[33]:


temp = gh362[:,:,1:361]

if os.path.isfile('./heat_anaume.npy'):
    os.remove('./heat_anaume.npy')

np.save('heat_anaume',temp)
    


# In[54]:


#ガベージコレクタ\n",
del dlr,ulr,sh,lh,interp_gh,gh362,gh362_old,gh,gh193
gc.collect()
### Net solar fluxの計算


# In[43]:


## net solar flux
## 単位換算
snr = (dsr - usr) / 4.186 / 1e4

# 193番目を作り0番目の値を入れておく（他も同様、画面端処理のため）
addzero = snr[:,:,0].reshape(yearnum,94,1)
snr193 = np.append(snr,addzero,axis=2)

## NCEPデータは読んだら必ずフリップ(4/4)
## 必要とされるデータは南が0で上になっている
index = 0
while index < yearnum:
    snr193[index,:,:] = np.flipud(snr193[index,:,:])
    index += 1


# ilandで陸地だったらfreshの同位置を０にする\n",
# bool indexで配列(landmask)が返る\n",
landmask = (iland >= 1)
# True の部分（地上）を０にする\n",
index = 0
while index < yearnum:
    snr193[index,:,:][landmask == True] = 0
    index += 1


# In[44]:


## ESTOCサイズにするために内挿
readdata = xr.DataArray(snr193,dims=['time','lat','lon']) # ファイルを読み込んでXArrayに入れる\n",
# dim_0 10日平均 dim_1 lat緯度 dim_2 lon経度\n",

def process_day(day):
    slicedays = readdata[day,:,:]
    zi = interpolate.interp2d(orig_lon,orig_lat,slicedays,kind='linear')(xi[:,0],yi[0,:])
    return zi

if __name__ == '__main__':
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.map(process_day, range(yearnum)) # 2703(1948/01-2023/01)

    interp_snr = np.array(results).reshape([yearnum,180,360])

print('end')


# In[45]:


if os.path.isfile('interp_snr.npy'):
    os.remove('interp_snr.npy')
    
np.save('interp_snr',interp_snr)


# In[46]:


# 画面端の処理をするために360番目の先に0番目の値を足す
# 0番の前に359番目の値を足す
addzero = interp_snr[:,:,0].reshape(yearnum,180,1)
add359 = interp_snr[:,:,359].reshape(yearnum,180,1)

snr362 = np.append(add359,(np.append(interp_snr,addzero,axis=2)),axis=2)


# In[47]:


snr362_old = snr362.copy() # 計算用コピー作成
# estocweightを初期化
estocweight = np.where(estoclandmask == True,0,1)
##estocweightを拡大する。
##0の左に３５９の値　３６０の右に元の０の値それぞれ値を入れておく（３・２３）
left = estocweight[:,0]
right = estocweight[:,359]

estocweight = np.hstack([(right.reshape(-1,1)),estocweight]) # 0の前に360の値を足す
estocweight = np.hstack([estocweight,(left.reshape(-1,1))]) #360の先に0の値を足す


# In[48]:


# ファイル作る前にあったらとりあえず消しておく
##ファイルがあるときは一旦削除する
if os.path.isfile('snr10dy_1948-2023.dat'):
    os.remove('snr10dy_1948-2023.dat')


# In[49]:


# landindexは陸地(landmask==True)の座標がタプルのndarray(配列）で返る
# landindex[0] = latitude
# landindex[1] = longitude



for index in range(yearnum): # ループが遅いのでテストで1年分だけ出してみる、本当は74年分で2664
    estocweight = estocweight_orig.copy() # 10/2
    
    
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
            lon = (landindex[1][i[0]])+1
            
            calcflag = estocweight_old[lat-1,lon] + estocweight_old[lat+1,lon] + estocweight_old[lat,lon-1] + estocweight_old[lat,lon+1] 
            calcdata = snr362_old[index,lat-1,lon]*estocweight_old[lat-1,lon]+snr362_old[index,lat+1,lon]*estocweight_old[lat+1,lon]\
                       +snr362_old[index,lat,lon-1]*estocweight_old[lat,lon-1]+snr362_old[index,lat,lon+1]*estocweight_old[lat,lon+1]
                           
                
            if calcflag > 0:
                calcweight = calcdata/calcflag
                res = abs(snr362_old[index,lat,lon] - calcweight)
                snr362[index,lat,lon] = calcweight
                estocweight[lat,lon] = 1
                resmax = max(resmax,res)
                    

        # 差分が範囲超えたら抜ける
        if 0.0001 > resmax:
            break

snr362[:,:,1:361].tofile('snr10dy_1948-2023.dat')

print('end')


# In[55]:


temp = snr362[:,:,1:361]

if os.path.isfile('./snr_anaume.npy'):
    os.remove('./snr_anaume.npy')

np.save('snr_anaume',temp)
    


# In[57]:





# In[ ]:


## 以下デバック用セル
