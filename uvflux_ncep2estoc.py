#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


ncep_param = xr.open_dataset('01_ESTOC_ForcingData/NCEP_NCAR_Forcing/NCEP/land.sfc.gauss.nc')

# NCEPの海陸マスクで陸の値で海の値を埋める処理
#　UFlux/VFluxは穴埋め計算のかわりにこれを行ってからESTOCサイズに合わせる
# 必要なNCEPの海陸マスク作成
lsmsk = np.array(ncep_param['land'])[0,:,:]
lsmsk = np.flipud(lsmsk)

# 陸１、海０になっているので反転させる
lsmsk = 1 - lsmsk

## この時点でデータは上が南


# In[3]:


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
# *10は単位換算
uflux = uflx * 10
vflux = vflx * 10


# In[ ]:





# In[4]:


## データは読んだらフリップ
## 提出データは南が0で上になっている

# Y軸（緯度）だけを上下反転します。
uflux = np.flip(uflux, axis=1)
vflux = np.flip(vflux, axis=1)
    

##　フリップさせて、このデータは上が南になる


# In[ ]:





# In[5]:


# scとlsmskの形状を確認
assert uflux.shape == (2703, 94, 192)
assert vflux.shape ==(2703,94,192)
assert lsmsk.shape == (94, 192)

## lsmskのサイズも94＊194にする
lsmsk = np.pad(lsmsk,((0,0),(1,1)),mode='wrap')

# 経度方向に対して360度側と0度側の値を追加
vflux = np.pad(vflux, ((0, 0), (0, 0), (1, 1)), mode='wrap')

for l in range(1000):  # 1000回で打ち切り(本当はもっと必要かも)
    resmax = -1.e10  # 収束判定の初期値
    for lon in range(1, 193):  # 経度方向のループ for 文は終了値を含まないので注意
        for lat in range(1, 94):  # 緯度方向のループ
            if lsmsk[lat, lon] == 0:  # NCEP海陸マスクで陸の場合実行
                if 0< lat <  93 and 0 < lon < 192:
                    avgf = 0.25 * (vflux[:, lat, lon-1] + vflux[:, lat, lon+1] + vflux[:, lat-1, lon] + vflux[:, lat+1, lon])  # 4点平均
                    res = np.abs(vflux[:, lat, lon] - avgf)  # 元の値と平均値の差
                    vflx[:, lat, lon] = avgf  # 元の値を平均値に更新
                    resmax = max(resmax, np.max(res))  # 収束判定値を更新
    vflux[:, :, 0] = vflux[:, :, 192]  # 経度方向360度側の値を更新
    vflux[:, :, 193] = vflux[:, :, 1]  # 経度方向0度側の値を更新
    if resmax < 1.e-6:  # 収束判定
        break

# 最後に経度方向の範囲(0～360)を取り出す
vflux = vflux[:, :, 1:193]
print('end')


# In[ ]:





# In[7]:


# 経度方向に対して360度側と0度側の値を追加
uflux = np.pad(uflux, ((0, 0), (0, 0), (1, 1)), mode='wrap')

for l in range(1000):  # 1000回で打ち切り(本当はもっと必要かも)
    resmax = -1.e10  # 収束判定の初期値
    for lon in range(1, 193):  # 経度方向のループ for 文は終了値を含まないので注意
        for lat in range(1, 94):  # 緯度方向のループ
            if lsmsk[lat, lon] == 0:  # NCEP海陸マスクで陸の場合実行
                if 0< lat <  93 and 0 < lon < 192:
                    avgf = 0.25 * (uflux[:, lat, lon-1] + uflux[:, lat, lon+1] + uflux[:, lat-1, lon] + uflux[:, lat+1, lon])  # 4点平均
                    res = np.abs(uflux[:, lat, lon] - avgf)  # 元の値と平均値の差
                    uflux[:, lat, lon] = avgf  # 元の値を平均値に更新
                    resmax = max(resmax, np.max(res))  # 収束判定値を更新
    uflux[:, :, 0] = uflux[:, :, 192]  # 経度方向360度側の値を更新
    uflux[:, :, 193] = uflux[:, :, 1]  # 経度方向0度側の値を更新
    if resmax < 1.e-6:  # 収束判定
        break

# 最後に経度方向の範囲(0～360)を取り出す
uflux = uflux[:, :, 1:193]
print('end')


# In[ ]:





# In[8]:


## uflux,vfluxの193番目に0番を追加する、画面端の処理のため

addzero = uflux[:,:,0].reshape(yearnum,94,1)
uflux193 = np.append(uflux,addzero,axis=2)

addzero = vflux[:,:,0].reshape(yearnum,94,1)
vflux193 = np.append(vflux,addzero,axis=2)



# In[ ]:





# In[9]:


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


# In[10]:


if os.path.isfile('interp_vflx.npy'):
    os.remove('interp_vflx.npy')

np.save('interp_vflx',interp_vflx)


# In[11]:


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


# In[12]:


if os.path.isfile('ingerp_uflx.npy'):
    os.remove('interp_uflx.npy')
    
np.save('interp_uflx',interp_uflx)


# In[ ]:





# In[13]:


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





# In[14]:


interp_uflx[:,estocmask == 0] = -1.0e33
interp_vflx[:,estocmask == 0] = -1.0e33


# In[ ]:





# In[19]:


temp = interp_uflx.byteswap()

if os.path.isfile('uflux_big.bin'):
    os.remove('uflux_big.bin')
    
with open('uflux_big.bin','wb') as f:
    temp.tofile(f)


# In[ ]:





# In[20]:


temp = interp_vflx.byteswap()

if os.path.isfile('vflux_big.bin'):
    os.remove('vflux_big.bin')
    
with open('vflux_big.bin','wb') as f:
    temp.tofile(f)


# In[ ]:





# In[ ]:




