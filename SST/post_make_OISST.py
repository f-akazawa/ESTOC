#!/usr/bin/env python
# coding: utf-8

# In[82]:


## OISST V2.1のデイリーデータを必要な期間分前処理する
# plt.は簡易的なのでaxでグラフを書くように色々やっておく
get_ipython().run_line_magic('matplotlib', 'inline')
# %matplotlib nbagg # anime用 
#　描画系インポートは確認用

import xarray as xr
import numpy as np
import pandas as pd
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.util as cutil
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from scipy.interpolate import griddata
import os
import gc


# In[2]:


# 最初に空のリストを作成します。
sst_list = []

for year in range(1981, 1992):
    fname = './downloads/OISSTV2.1/sst.day.mean.'+ str(year) + '.nc'
    # xarrayでは_FillValueを適切に処理してNanにしてしまうので後の計算で困る
    #　　sst:missing_value となっているが、おそらく陸地の部分-9.96921e+36f がNanになる
    #　mask_and_scale=Falseのオプション設定でNanに置換するのを回避する
    data = xr.open_dataset(fname , mask_and_scale=False)
    sst = data['sst']
    
    sst_list.append(sst)
    
# リスト内の全てのDataArrayを時間軸で結合します。
total_sst = xr.concat(sst_list, dim='time')

# 一月毎に上旬、中旬、下旬で平均を取ります。
def ten_days_average(x):
    if x.day <= 10:
        return '上旬'
    elif x.day <= 20:
        return '中旬'
    else:
        return '下旬'

# 新しい座標を作成します。
total_sst['year'] = total_sst['time.year']
total_sst['month'] = total_sst['time.month']
total_sst['ten_days'] = xr.DataArray(total_sst.time.to_series().apply(ten_days_average), dims='time')

# MultiIndexを使用して新しい座標を作成します。
grouper = xr.DataArray(
    pd.MultiIndex.from_arrays([total_sst['year'].values, total_sst['month'].values, total_sst['ten_days'].values], names=['year', 'month', 'ten_days']),
    dims='time',
)

# 新しい座標を使用してgroupbyします。
total_sst_avg = total_sst.groupby(grouper).mean(dim='time')


# In[ ]:





# In[12]:


hozon = total_sst_avg.values
hozon[hozon == -9.9692100e+36] == -1.0e33


# In[13]:


print(hozon[0,0:100,:])


# In[14]:


plt.imshow(hozon[0,:,:])


# In[ ]:





# In[ ]:





# In[6]:


## 一旦保存してガベージコレクタ
if os.path.isfile('./OISST1981-1991.npy'):
    os.remove('./OISST1981-1991.npy')
    
np.save('OISST1981-1991',hozon)

gc.collect()


# In[7]:


# 最初に空のリストを作成します。
sst_list = []

for year in range(1992, 2003):
    fname = './downloads/OISSTV2.1/sst.day.mean.'+ str(year) + '.nc'
    data = xr.open_dataset(fname , mask_and_scale=False)
    sst = data['sst']
    
    sst_list.append(sst)
    
# リスト内の全てのDataArrayを時間軸で結合します。
total_sst = xr.concat(sst_list, dim='time')

# 一月毎に上旬、中旬、下旬で平均を取ります。
def ten_days_average(x):
    if x.day <= 10:
        return '上旬'
    elif x.day <= 20:
        return '中旬'
    else:
        return '下旬'

# 新しい座標を作成します。
total_sst['year'] = total_sst['time.year']
total_sst['month'] = total_sst['time.month']
total_sst['ten_days'] = xr.DataArray(total_sst.time.to_series().apply(ten_days_average), dims='time')

# MultiIndexを使用して新しい座標を作成します。
grouper = xr.DataArray(
    pd.MultiIndex.from_arrays([total_sst['year'].values, total_sst['month'].values, total_sst['ten_days'].values], names=['year', 'month', 'ten_days']),
    dims='time',
)

# 新しい座標を使用してgroupbyします。
total_sst_avg = total_sst.groupby(grouper).mean(dim='time')


# In[8]:


hozon = total_sst_avg.values


# In[9]:


## 一旦保存してガベージコレクタ
if os.path.isfile('./OISST1992-2002.npy'):
    os.remove('./OISST1992-2002.npy')
    
np.save('OISST1992-2002',hozon)

gc.collect()


# In[10]:


# 最初に空のリストを作成します。
sst_list = []

for year in range(2003, 2013):
    fname = './downloads/OISSTV2.1/sst.day.mean.'+ str(year) + '.nc'
    data = xr.open_dataset(fname , mask_and_scale=False)
    sst = data['sst']
    
    sst_list.append(sst)
    
# リスト内の全てのDataArrayを時間軸で結合します。
total_sst = xr.concat(sst_list, dim='time')

# 一月毎に上旬、中旬、下旬で平均を取ります。
def ten_days_average(x):
    if x.day <= 10:
        return '上旬'
    elif x.day <= 20:
        return '中旬'
    else:
        return '下旬'

# 新しい座標を作成します。
total_sst['year'] = total_sst['time.year']
total_sst['month'] = total_sst['time.month']
total_sst['ten_days'] = xr.DataArray(total_sst.time.to_series().apply(ten_days_average), dims='time')

# MultiIndexを使用して新しい座標を作成します。
grouper = xr.DataArray(
    pd.MultiIndex.from_arrays([total_sst['year'].values, total_sst['month'].values, total_sst['ten_days'].values], names=['year', 'month', 'ten_days']),
    dims='time',
)

# 新しい座標を使用してgroupbyします。
total_sst_avg = total_sst.groupby(grouper).mean(dim='time')


# In[11]:


hozon = total_sst_avg.values


# In[12]:


## 一旦保存してガベージコレクタ
if os.path.isfile('./OISST2003-2012.npy'):
    os.remove('./OISST2003-2012.npy')
    
np.save('OISST2003-2012',hozon)

gc.collect()


# In[13]:


# 最初に空のリストを作成します。
sst_list = []

for year in range(2013, 2024):
    fname = './downloads/OISSTV2.1/sst.day.mean.'+ str(year) + '.nc'
    data = xr.open_dataset(fname,mask_and_scale=False)
    sst = data['sst']
    
    sst_list.append(sst)
    
# リスト内の全てのDataArrayを時間軸で結合します。
total_sst = xr.concat(sst_list, dim='time')

# 一月毎に上旬、中旬、下旬で平均を取ります。
def ten_days_average(x):
    if x.day <= 10:
        return '上旬'
    elif x.day <= 20:
        return '中旬'
    else:
        return '下旬'

# 新しい座標を作成します。
total_sst['year'] = total_sst['time.year']
total_sst['month'] = total_sst['time.month']
total_sst['ten_days'] = xr.DataArray(total_sst.time.to_series().apply(ten_days_average), dims='time')

# MultiIndexを使用して新しい座標を作成します。
grouper = xr.DataArray(
    pd.MultiIndex.from_arrays([total_sst['year'].values, total_sst['month'].values, total_sst['ten_days'].values], names=['year', 'month', 'ten_days']),
    dims='time',
)

# 新しい座標を使用してgroupbyします。
total_sst_avg = total_sst.groupby(grouper).mean(dim='time')


# In[14]:


## 2023.2まで利用するので後ろを削除する
## 2013-2022 = 10years > 3*12*10 = 360
## 2023.1-2023.2 > 6 
##　上記計算とおり先頭から３６６までを利用する。
total_sst_avg = total_sst_avg[:366,:,:]


# In[15]:


hozon = total_sst_avg.values


# In[16]:


## 一旦保存してガベージコレクタ
if os.path.isfile('./OISST2013-2023.npy'):
    os.remove('./OISST2013-2023.npy')
    
np.save('OISST2013-2023' , hozon)

gc.collect()


# In[2]:





# In[ ]:





# In[2]:


## OISSTの海（１）、陸（０）データを作る
##　元データではNanが陸地
## lsmask.oisst.ncというのがあるので海陸マスクはそれを使う
## xr.DataArrayの720*1440、海１，陸０のデータが取り出せる
oisst_landmask_HD = xr.open_dataset('./lsmask.oisst.nc')['lsmask'][0,:,:]



# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[3]:


first = np.load('OISST1981-1991.npy')
second = np.load('OISST1992-2002.npy')
third = np.load('OISST2003-2012.npy')
forth = np.load('OISST2013-2023.npy')
hozon = np.concatenate((first,second,third,forth),axis=0)


# In[4]:


# 使わないのでメモリから削除
del first,second,third,forth


# In[20]:


del sst,total_sst,total_sst_avg


# In[44]:


print(hozon.dtype)


# In[5]:


##　OISSTデータをESTOCサイズに内挿する
##土居さんに教えてもらった方法でやる
##1/4度と１度の格子境界がきっちり揃っていることが条件
##OISSTの海陸マスクを使ってそれを使い、海域のデータだけを抽出して４＊４グリッド毎に平均を取る

from scipy import ndimage


# ここまででhozonとoisst_landmask_HDは既に読み込まれている

# oisst_landmask_HDをhozonと同じ形状にブロードキャストします。
# sea=1 , land=0
oisst_landmask_HD_broadcasted = np.broadcast_to(oisst_landmask_HD, hozon.shape)

# 海域のデータだけを抽出します。
masked_hozon = hozon * oisst_landmask_HD_broadcasted

# 新しい配列を作成して、4x4グリッドごとに平均値を計算します。
# Pythonでは a/b は浮動小数点除算になるので 720 / 4 は180.0の浮動小数点数が返る
# a // bは整数除算になるので 720 // 4 は180という整数値が返る
# 配列の形状指定なので整数値が必要
new_shape = (hozon.shape[0], hozon.shape[1]//4,hozon.shape[2]//4)
resized_hozon = np.zeros(new_shape)

# 4x4グリッド内の海域データの平均値を計算します。
for i in range(new_shape[1]):
    for j in range(new_shape[2]):
        # 元のグリッドから4x4ブロックを取り出します
        block_hozon = hozon[:, i*4:(i+1)*4, j*4:(j+1)*4]
        block_landmask = oisst_landmask_HD_broadcasted[:, i*4:(i+1)*4, j*4:(j+1)*4]
        # ブロック内が全て陸（値が0）でなければ平均を計算します
        if not np.all(block_landmask == 0):
            ## block_landmaskに０があると平均を計算する分母も変わるので０以外のグリッド数をカウントする
            num_sea_grids = np.count_nonzero(block_landmask == 1 , axis=(1,2))
            resized_hozon[:, i, j] = np.sum(block_hozon * (block_landmask == 1), axis=(1,2)) / num_sea_grids


print('end')


# In[22]:


## 完成
if os.path.isfile('./OISST_result.npy'):
    os.remove('./OISST_result.npy')

np.save('OISST_result.npy',resized_hozon)


# In[3]:


# debug load
resized_hozon = np.load('./OISST_result.npy')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[5]:


# oisst_landmask_HDをESTOCのサイズに変換する

# 全部海は１６、陸と海が混じっている所は１５〜１，全部陸なら０
#　穴埋めはoisst_maskが０のところとESTOCで海１になっている所で行う

# 4x4グリッドごとに合計値を計算します。
coarsened_mask = oisst_landmask_HD.coarsen(lat=4, lon=4, boundary='trim').sum()

# 元の格子で全部陸だった場合は0(land)、それ以外は1(sea)に変換します。
oisst_mask = xr.where(coarsened_mask == 0, 0, 1)

# numpy配列に変換
oisst_mask = oisst_mask.values


# In[79]:


# ESTOCのランドマスクを用意
# ESTOCランドマスク作成
# （155 x 360) 75S-80N 上下にダミーの海を足す

# 元データは75Sから80Nまでなので上下90まで陸地を埋める
#温度データなので０は含まれてしまうので陸地は０よりもものすごく小さくする
top = np.full((15,360),0)
bottom = np.full((10,360),0)

# ESTOCマスクデータを読みこむ（元データは天地逆だがフリップさせてはいない）
data = np.loadtxt('../kmt_data.txt',dtype='int') # 元がテキストなのでキャストも必要
estocmask = data
# 上下にダミーの海を足す
estocmask = np.append(top,estocmask,axis=0)
estocmask = np.append(estocmask,bottom,axis=0)

estoclandmask = (estocmask == 0)


## Boolマスクを変換して陸0，　海1でESTOCマスクを作る
estocweight = np.where(estoclandmask == True,0,1)


# In[7]:


## mask の中身はBool
#
mask = ((estocweight == 1) & (oisst_mask == 0)) 
landindex =  np.where(mask == 1)

# estocweightで1であり、かつcoarsened_maskで0の場所を抜き出します。
selected_positions = np.where((estocweight == 1) & (coarsened_mask == 0))

# 結果を表示します。
print("Selected positions (lat, lon):")
for lat, lon in zip(*selected_positions):
    #print(f"({lat}, {lon})")
    estocweight[lat,lon] = 0
    
 # estocweightで、このlat,lonに対応した場所を０にする。これが計算フラグとなる


# In[80]:


plt.imshow(estocweight)


# In[9]:


#　estocweightを横に拡大する
##estocweightを拡大する。
##0の左に３５９の値　３６０の右に元の０の値それぞれ値を入れておく（３・２３）
left = estocweight[:,0]
right = estocweight[:,359]

estocweight = np.hstack([(right.reshape(-1,1)),estocweight]) # 0の前に360の値を足す
estocweight = np.hstack([estocweight,(left.reshape(-1,1))]) #360の先に0の値を足す

# この時点でestocweight 180＊362
estocweight_old = estocweight.copy() # 計算用に必要なので作っておく 
estocweight_orig = estocweight.copy() # 同上


# In[10]:


# OISSTも横に拡大する
## 画面端の処理
# 画面端の処理をするために横に1列増やす、左右必要
addzero = resized_hozon[:,:,0].reshape(resized_hozon.shape[0],resized_hozon.shape[1],1) # 0番目の列の値を抜き出して3次元に直す
add359 = resized_hozon[:,:,359].reshape(resized_hozon.shape[0],resized_hozon.shape[1],1) # 360番目の値を抜き出して3次元に直す

oisst362 = np.append(add359,(np.append(resized_hozon,addzero,axis=2)),axis=2)


# In[11]:


oisst362_old = oisst362.copy()


# In[12]:


## 畳み込み計算部

for index in range(oisst362.shape[0]): # oisst362.shape[0] = 1487
#for index in range(10): # for debug

    estocweight = estocweight_orig.copy()

    
    for counter in range(1000):# 1000回繰り返す（又は閾値を超えたらループ終了
        
        
        resmax = -10000000000
        oisst362_old[index,:,:] = oisst362[index,:,:] ## コレも必要
        
        estocweight_old[:,:] = estocweight[:,:] # フラグ１の場所を更新
        
        ## estocweight0,360 に更新（右と左両方追加、323)
        left = estocweight[:,1]
        right = estocweight[:,360]
        estocweight_old[:,0] = right # 0の前に360の値を更新
        estocweight_old[:,361] = left #360の先に1の値を更新
        ## 同じことをersst362でもやる
        oisst362_old[index,:,0] = oisst362[index,:,360]
        oisst362_old[index,:,361] = oisst362[index,:,1]
        
        ## ここがズレてるので注意
        for i in enumerate(landindex[0]): # i = loop index
            lat = (landindex[0][i[0]])
            lon = (landindex[1][i[0]])+1 # 6/9
            ## ループの1回目は０であるはず、ESTOCWEIGHT lat,lon
            # 1箇所だけlon=360がある
            
            ## latも値を確認0、179があるとだめ
            if lat > 0 and lat < estocweight_old.shape[0]-1:
                calcflag = estocweight_old[lat-1,lon] + estocweight_old[lat+1,lon] + estocweight_old[lat,lon-1] + estocweight_old[lat,lon+1] 
                calcdata = (oisst362_old[index,lat-1,lon]*estocweight_old[lat-1,lon] + oisst362_old[index,lat+1,lon]*estocweight_old[lat+1,lon]\
                                +oisst362_old[index,lat,lon-1]*estocweight_old[lat,lon-1] + oisst362_old[index,lat,lon+1]*estocweight_old[lat,lon+1])
                if calcflag > 0:
                    calcweight = calcdata/(calcflag)
                    res = abs(oisst362_old[index,lat,lon]-calcweight)
                    oisst362[index,lat,lon] = calcweight
                    estocweight[lat,lon] = 1
                    resmax = max(resmax,res)

        if  0.0000001 > resmax:
            print('index=',index , 'counter=',counter,'resmax=',resmax)
            break
            
        


## ファイル書き出しを別セルにする（遅いので）
oisst362[:,:,1:361].tofile('OISST_only.dat')
print('end')


# In[13]:


only_oisst = oisst362[:,:,1:361]


# In[41]:


plt.imshow(only_oisst[33,:,:])


# In[23]:


print(only_oisst[33,0:30,0])


# In[ ]:





# In[ ]:





# In[24]:


only_oisst = np.fromfile('./OISST_only.dat').reshape(1494,180,360)


# In[ ]:





# In[ ]:





# In[73]:





# In[62]:


## 海の値0を
## ESTOCの海に変換する-1.0e+33
only_oisst[only_oisst == 0] = -1.0e33


# In[ ]:





# In[ ]:





# In[ ]:





# In[38]:


only_ersst = np.fromfile('./ERSST_only.dat').reshape(1221,180,360)


# In[39]:


# ersst 1948.01 - 1981.11
# １年は３６個（上中下＊１２ヶ月）　
#　1981／８月までを使うことにしたのでうしろから９個削除する
only_ersst = only_ersst[:1212,:,:]


# In[67]:


## 海の値を調べた所-9.999999944957273e+32,これを
## ESTOCの海に変換する-1.0e+33

only_ersst[only_ersst == -9.99999994e+32] = -1.0e33


# In[68]:


# ERSST とOISSTをマージ、土居さんに教えてもらった。
#ersst:1981.7-1
#ersst:1981.7-2
#ersst:1981.7-3
#ersst:1981.8-1 (A)
#ersst:1981.8-2 = A*2.0/3.0 + B*1.0/3.0
#ersst:1981.8-3 = A*1.0/3.0 + B*2.0/3.0
#oisst:1981.9-1 (B)
#oisst:1981.9-2
#oisst:1981.9-3
#oisst:1981.10-1
#oisst:1981.10-2
#oisst:1981.10-3

## ersstの最後２つ（１９８１年８月中旬と下旬）はOISSTの最初のデータを使って重み付け平均を作って入れ替える
bigA = only_ersst[1209,:,:]
bigB = only_oisst[0,:,:]

aug2nd = bigA*2.0/3.0 + bigB*1.0/3.0
aug3rd = bigA*1.0/3.0 + bigB*2.0/3.0


# In[69]:


## ersst の後ろ２つを消す
only_ersst = only_ersst[:1210,:,:]

# 重み付け平均したやつに軸を追加して合わせる
aug2nd3D= aug2nd[np.newaxis,:,:]
aug3rd3D = aug3rd[np.newaxis,:,:]

#　１軸目に合わせてマージ
only_ersst = np.concatenate((only_ersst,aug2nd3D,aug3rd3D),axis=0)


# In[70]:


merged = np.concatenate((only_ersst,only_oisst),axis=0)


# In[84]:


#  ERSST , OISSTが元の地形なのでESTOCの地形でマスクする

#　kmt.dataから読み直して陸0海１にしたestocweightをブロードキャストする
lastmask = np.tile(estocweight,(merged.shape[0],1,1))


# In[ ]:





# In[ ]:





# In[98]:


## ESTOCの地形でマスクして提出データになる
hoge = merged * lastmask


# In[102]:


#　陸が０になってしまうのでもう一度ESTOCの陸地を入れる
hoge[hoge == 0] = -1.0e33


# In[ ]:





# In[103]:


plt.imshow(hoge[1211,:,:],vmin=-10,vmax=35)
plt.savefig('oisst.png')


# In[105]:


plt.imshow(hoge[1210,:,:],vmin=-10,vmax=35)
plt.savefig('ersst.png')


# In[106]:


#　ビッグエンディアンに変換
merged_big = hoge.byteswap()


# In[107]:


##　提出ファイルはビックエンディアンで書き出す
if os.path.isfile('SST_big.bin'):
    os.remove('SST_big.bin')

with open('SST_big.bin' , 'wb') as f:
    merged_big.tofile(f)


# In[59]:


print(only_ersst[0,0:30,0])


# In[58]:





# In[ ]:





# In[ ]:





# In[73]:


plt.imshow(only_oisst[0,:,:],vmin=-10,vmax=30)


# In[77]:


plt.imshow(merged[1211,:,:],vmin=-10,vmax=30)
plt.savefig('ersst.png')


# In[78]:


plt.imshow(merged[1212,:,:],vmin=-10,vmax=30)
plt.savefig('oisst.png')


# In[65]:


only_oisst[0,0:30,0]


# In[66]:


only_ersst[200,:,:]


# In[26]:


print(only_oisst.dtype)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


##  上手く回ったら、出来上がったのは2013から2023年2月までの分である。
np.save('ES１９８１-１９９１',new_hozon)


# In[ ]:





# In[ ]:


if sst[0,:,:].isnull().any():
    print("There are NaN values in the DataArray.")
else:
    print("There are no NaN values in the DataArray.")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


original_grid_x


# In[ ]:


plt.imshow(average_hozon[0,:,:])


# In[ ]:





# In[ ]:


kakunin = xr.open_dataset('./downloads/OISSTV2.1/sst.day.mean.1982.nc')


# In[ ]:


kakunin


# In[ ]:


original_grid_x


# In[ ]:


original_grid_y


# In[ ]:


gc.collect()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


＃#ESTOCへのサイズ変更


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


#　元のグリッドと新しいグリッドの座標を1次元配列に変換

estoc_list = []

#for index in range(first.shape[0]):
estoc_grid_x,estoc_grid_y = np.mgrid[-89.5:90:1,0.5:360:1]
points = np.vstack((original_grid_x.ravel() , original_grid_y.ravel())).T
values = hozon[0,:,:].ravel()
new_points = np.vstack((estoc_grid_x.ravel() , estoc_grid_y.ravel())).T
    
#　griddata で変換する
new_data = griddata(points , values , new_points, method='linear')



# In[ ]:


plt.imshow(new_data.reshape(180,360))


# In[ ]:




