#!/usr/bin/env python
# coding: utf-8

# In[1]:


import xarray as xr
import inspect
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.util as cutil
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import os
import gc

month = [1,2,3,4,5,6,7,8,9,10,11,12]


# In[2]:


prateyear = xr.DataArray()
#month10dy = xr.DataArray(np.zeros((3,94,192))) # ループの中でやらないとどんどん増える
if os.path.isfile('./prate10dy.dat'):
    os.remove('./prate10dy.dat')


# prate.sfc.gauss.XXXX.nc読み込み、１０日毎に単純平均を取ったらバイナリファイルprate10dy.datに保存
# 他にdlwrf,lhtfl,shtfl,ulwrf,uflx,uswrf,vflxがある
# lhtflは次のFreshWaterFlux作成でも使う
with open('prate10dy.dat',mode='a') as f:
    for year in range(1948,2024): # 1948-2023
        month10dy = xr.DataArray(np.zeros((3,94,192))) # コピー用の配列を初期化
        pratenc = '01_ESTOC_ForcingData/NCEP_NCAR_Forcing/NCEP/prate.sfc.gauss.'+str(year)+'.nc' # ncファイルはすべてlat94,lon192,Time365

        pratenc = xr.open_dataset(pratenc)['prate'].load() # パタメータ名はファイルから読める。load()は配列をメモリに一度全部読み込む

    
        #dlwrf = np.empty([366,94,192]) # 10日区切りを保存する変数、命名は上と同じくファイルから読むべき
        for x in month:
            # 範囲指定用の文字列構成
            start1 = str(year) + '-' + str(x) + '-1'
            end1 = str(year) + '-' + str(x) + '-10'
            start2 = str(year) + '-' + str(x) + '-11'
            end2 = str(year) + '-' + str(x) + '-20'                
            start3 = str(year) + '-' + str(x) + '-21'
    
            if x == 12:
                nextmonth = 1
                lastday = str(year+1) + '-' + str(nextmonth) + '-1'
            else:
                nextmonth = x + 1
                lastday = str(year) + '-' + str(nextmonth) + '-1'

            end3 = datetime.datetime.strptime(lastday,'%Y-%m-%d') - datetime.timedelta(days = 1)                
            end3 = end3.strftime('%Y-%m-%d')

              # 上中下旬に分けて平均を取る（とりあえずmean重みなし）
            data_mean_10dy1 = pratenc.loc[start1:end1].mean(dim='time')
            data_mean_10dy2 = pratenc.loc[start2:end2].mean(dim='time')
            data_mean_10dy3 = pratenc.loc[start3:end3].mean(dim='time')

            #月毎に10日区切りで3つ作ったら保存用の配列に代入
            temp = np.stack([data_mean_10dy1,data_mean_10dy2,data_mean_10dy3],0) # tempは1ヶ月分
            month10dy = np.append(month10dy,temp,axis=0) # 1ヶ月分のtempを順番にappendしていく（70年分）
                
                
        prateyear = month10dy[3:] # 最初の3つは0なので消しておく

# どんどん遅くなるので一旦バイナリファイルとして保存
# 最後の4つが空列になってる?
        prateyear.tofile(f)
#        from IPython.core.debugger import Pdb; Pdb().set_trace()


print('end')


# In[3]:


prate = np.fromfile('prate10dy.dat').reshape(2736,94,192) # 2736(1948 - 2023 76 years)

## 2023年末までの数になっているので11ヶ月分後ろから削除する（２７０３期間）
## 1948 - 2022 (75年 * 12ヶ月 * 3個 )
## 2023.01(3個)
prate = prate[:2703,:,:]

if os.path.isfile('./prate10dy.dat'):
    os.remove('./prate10dy.dat')

with open('prate10dy.dat' , mode='a') as f:
    prate.tofile(f)


# In[4]:


# 要素削除とガベージコレクター
import gc

del data_mean_10dy1,data_mean_10dy3,data_mean_10dy2
del month10dy,pratenc,prateyear,temp

gc.collect()


# In[5]:


# 上と同様にlhtfl.sfc.gauss.XXXX.ncを読み込んでバイナリファイルlhtfl10dy.datに保存
if os.path.isfile('./lhtfl10dy.dat'):
    os.remove('./lhtfl10dy.dat')

lhtflyear = xr.DataArray()
with open('lhtfl10dy.dat',mode='a') as f:
    for year in range(1948,2024): # 1948-2023
        month10dy = xr.DataArray(np.zeros((3,94,192))) # コピー用の配列を初期化
        lhtflnc = '01_ESTOC_ForcingData/NCEP_NCAR_Forcing/NCEP/lhtfl.sfc.gauss.'+str(year)+'.nc'
        lhtflnc = xr.open_dataset(lhtflnc)['lhtfl'].load()
        

        for x in month:
            # 範囲指定用の文字列構成
            start1 = str(year) + '-' + str(x) + '-1'
            end1 = str(year) + '-' + str(x) + '-10'
            start2 = str(year) + '-' + str(x) + '-11'
            end2 = str(year) + '-' + str(x) + '-20'
            start3 = str(year) + '-' + str(x) + '-21'
    
            if x == 12:
                nextmonth = 1
                lastday = str(year+1) + '-' + str(nextmonth) + '-1'
            else:
                nextmonth = x + 1
                lastday = str(year) + '-' + str(nextmonth) + '-1'

            end3 = datetime.datetime.strptime(lastday,'%Y-%m-%d') - datetime.timedelta(days = 1)
            end3 = end3.strftime('%Y-%m-%d')

            # 上中下旬に分けて平均を取る（とりあえずmean重みなし）
            data_mean_10dy1 = lhtflnc.loc[start1:end1].mean(dim='time')
            data_mean_10dy2 = lhtflnc.loc[start2:end2].mean(dim='time')
            data_mean_10dy3 = lhtflnc.loc[start3:end3].mean(dim='time')

            #月毎に10日区切りで3つ作ったら保存用の配列に代入
            temp = np.stack([data_mean_10dy1,data_mean_10dy2,data_mean_10dy3],0) # tempは1ヶ月分
            month10dy = np.append(month10dy,temp,axis=0) # 1ヶ月分のtempを順番にappendしていく（70年分）
                            
        lhtflyear = month10dy[3:] # 最初の3つは0なので消しておく
# どんどん遅くなるので一旦バイナリファイルとして保存
# インデント注意
        lhtflyear.tofile(f)


print ('end')


# In[6]:


## 2023年も12月までループ回っているので後ろ11ヶ月分は削除して保存する
lhtfl = np.fromfile('./lhtfl10dy.dat').reshape(2736,94,192)

lhtfl = lhtfl[:2703,:,:]

if os.path.isfile('./lhtfl10dy.dat'):
    os.remove('./lhtfl10dy.dat')

with open('lhtfl10dy.dat' , mode='a') as f:
    lhtfl.tofile(f)


# In[7]:


# gabage collcter
del data_mean_10dy1,data_mean_10dy3,data_mean_10dy2
del lhtflyear
del month10dy,temp
gc.collect()


# In[9]:


if os.path.isfile('./dswrf10dy.dat'):
    os.remove('./dswrf10dy.dat')

# dswrf.sfc.gauss.XXXX.ncを読み込んで１０日平均を取ってバイナリファイルに一度保存
dswrfyear = dlwrfyear = uswrfyear = ulwrfyear = shtflyear = uflxyear = vflxyear = xr.DataArray()
with open('dswrf10dy.dat',mode='a') as f:
    for year in range(1948,2024): # 1948-2023
        month10dy = xr.DataArray(np.zeros((3,94,192))) # コピー用の配列を初期化
        dswrfnc = '01_ESTOC_ForcingData/NCEP_NCAR_Forcing/NCEP/dswrf.sfc.gauss.'+str(year)+'.nc'
        dswrfnc = xr.open_dataset(dswrfnc)['dswrf'].load()

        for x in month:
            # 範囲指定用の文字列構成
            start1 = str(year) + '-' + str(x) + '-1'
            end1 = str(year) + '-' + str(x) + '-10'
            start2 = str(year) + '-' + str(x) + '-11'
            end2 = str(year) + '-' + str(x) + '-20'
            start3 = str(year) + '-' + str(x) + '-21'
    
            if x == 12:
                nextmonth = 1
                lastday = str(year+1) + '-' + str(nextmonth) + '-1'
            else:
                nextmonth = x + 1
                lastday = str(year) + '-' + str(nextmonth) + '-1'

            end3 = datetime.datetime.strptime(lastday,'%Y-%m-%d') - datetime.timedelta(days = 1)
            end3 = end3.strftime('%Y-%m-%d')

            # 上中下旬に分けて平均を取る（とりあえずmean重みなし）
            data_mean_10dy1 = dswrfnc.loc[start1:end1].mean(dim='time')
            data_mean_10dy2 = dswrfnc.loc[start2:end2].mean(dim='time')
            data_mean_10dy3 = dswrfnc.loc[start3:end3].mean(dim='time')

            #月毎に10日区切りで3つ作ったら保存用の配列に代入
            temp = np.stack([data_mean_10dy1,data_mean_10dy2,data_mean_10dy3],0) # tempは1ヶ月分
            month10dy = np.append(month10dy,temp,axis=0) # 1ヶ月分のtempを順番にappendしていく（70年分）
            
            
        dswrfyear= month10dy[3:] # 最初の3つは0なので消しておく

        dswrfyear.tofile(f)

print('end')


# In[10]:


## 2023.2~12 delete
dswrf = np.fromfile('./dswrf10dy.dat').reshape(2736,94,192)

dswrf = dswrf[:2703,:,:]

if os.path.isfile('./dswrf10dy.dat'):
    os.remove('./dswrf10dy.dat')

with open('dswrf10dy.dat' , mode='a') as f:
    dswrf.tofile(f)


# In[11]:


# gabage collecter
del data_mean_10dy1,data_mean_10dy3,data_mean_10dy2
del dswrfyear
del month10dy,temp
gc.collect()


# In[8]:





# In[11]:





# In[ ]:





# In[12]:


if os.path.isfile('./dlwrf10dy.dat'):
    os.remove('./dlwrf10dy.dat')

# dlwrf.sfc.gauss.XXXX.ncファイルを読んで１０日平均を取ってバイナリファイルに保存
with open('dlwrf10dy.dat',mode='a') as f:
    for year in range(1948,2024): # 1948-2023
        month10dy = xr.DataArray(np.zeros((3,94,192))) # コピー用の配列を初期化
        dlwrfnc = '01_ESTOC_ForcingData/NCEP_NCAR_Forcing/NCEP/dlwrf.sfc.gauss.'+str(year)+'.nc'
        dlwrfnc = xr.open_dataset(dlwrfnc)['dlwrf'].load()
    
        for x in month:
           # 範囲指定用の文字列構成
            start1 = str(year) + '-' + str(x) + '-1'
            end1 = str(year) + '-' + str(x) + '-10'
            start2 = str(year) + '-' + str(x) + '-11'
            end2 = str(year) + '-' + str(x) + '-20'
            start3 = str(year) + '-' + str(x) + '-21'
    
            if x == 12:
                nextmonth = 1
                lastday = str(year+1) + '-' + str(nextmonth) + '-1'
            else:
                nextmonth = x + 1
                lastday = str(year) + '-' + str(nextmonth) + '-1'

            end3 = datetime.datetime.strptime(lastday,'%Y-%m-%d') - datetime.timedelta(days = 1)
            end3 = end3.strftime('%Y-%m-%d')

            # 上中下旬に分けて平均を取る（とりあえずmean重みなし）
            data_mean_10dy1 = dlwrfnc.loc[start1:end1].mean(dim='time')
            data_mean_10dy2 = dlwrfnc.loc[start2:end2].mean(dim='time')
            data_mean_10dy3 = dlwrfnc.loc[start3:end3].mean(dim='time')

            #月毎に10日区切りで3つ作ったら保存用の配列に代入
            temp = np.stack([data_mean_10dy1,data_mean_10dy2,data_mean_10dy3],0) # tempは1ヶ月分
            month10dy = np.append(month10dy,temp,axis=0) # 1ヶ月分のtempを順番にappendしていく（70年分）

                
        dlwrfyear= month10dy[3:] # 最初の3つは0なので消しておく
        dlwrfyear.tofile(f)

print('end')


# In[13]:


## delete 2023.2~2023.12

dlwrf = np.fromfile('./dlwrf10dy.dat').reshape(2736,94,192)

dlwrf = dlwrf[:2703,:,:]

if os.path.isfile('./dlwrf10dy.dat'):
    os.remove('./dlwrf10dy.dat')

with open('dlwrf10dy.dat' , mode='a') as f:
    dlwrf.tofile(f)


# In[14]:


# gabage collecter
del data_mean_10dy1,data_mean_10dy3,data_mean_10dy2
del dlwrfyear
del month10dy,temp
gc.collect()


# In[15]:


if os.path.isfile('./ulwrf10dy.dat'):
    os.remove('./ulwrf10dy.dat')

# ulwrf.sfc.gauss.XXXX.ncファイルを読んで１０日平均を取ってバイナリファイルに保存
with open('ulwrf10dy.dat',mode='a') as f:
    for year in range(1948,2024): # 1948-2023
        month10dy = xr.DataArray(np.zeros((3,94,192))) # コピー用の配列を初期化
        ulwrfnc = '01_ESTOC_ForcingData/NCEP_NCAR_Forcing/NCEP/ulwrf.sfc.gauss.'+str(year)+'.nc'
        ulwrfnc = xr.open_dataset(ulwrfnc)['ulwrf'].load()
    
        for x in month:
            # 範囲指定用の文字列構成
            start1 = str(year) + '-' + str(x) + '-1'
            end1 = str(year) + '-' + str(x) + '-10'
            start2 = str(year) + '-' + str(x) + '-11'
            end2 = str(year) + '-' + str(x) + '-20'
            start3 = str(year) + '-' + str(x) + '-21'
    
            if x == 12:
                nextmonth = 1
                lastday = str(year+1) + '-' + str(nextmonth) + '-1'
            else:
                nextmonth = x + 1
                lastday = str(year) + '-' + str(nextmonth) + '-1'

            end3 = datetime.datetime.strptime(lastday,'%Y-%m-%d') - datetime.timedelta(days = 1)
            end3 = end3.strftime('%Y-%m-%d')

            # 上中下旬に分けて平均を取る（とりあえずmean重みなし）
            data_mean_10dy1 = ulwrfnc.loc[start1:end1].mean(dim='time')
            data_mean_10dy2 = ulwrfnc.loc[start2:end2].mean(dim='time')
            data_mean_10dy3 = ulwrfnc.loc[start3:end3].mean(dim='time')

            #月毎に10日区切りで3つ作ったら保存用の配列に代入
            temp = np.stack([data_mean_10dy1,data_mean_10dy2,data_mean_10dy3],0) # tempは1ヶ月分
            month10dy = np.append(month10dy,temp,axis=0) # 1ヶ月分のtempを順番にappendしていく（70年分）
            
    
        ulwrfyear= month10dy[3:] # 最初の3つは0なので消しておく
        ulwrfyear.tofile(f)

print('end')


# In[16]:


ulwrf = np.fromfile('./ulwrf10dy.dat').reshape(2736,94,192)

## delete 2023.2-2023.12

ulwrf = ulwrf[:2703,:,:]

if os.path.isfile('./ulwrf10dy.dat'):
    os.remove('./ulwrf10dy.dat')

with open('ulwrf10dy.dat' , mode='a') as f:
    ulwrf.tofile(f)


# In[17]:


# gabage collecter
del data_mean_10dy1,data_mean_10dy3,data_mean_10dy2
del ulwrfyear
del month10dy,temp
gc.collect()


# In[18]:


if os.path.isfile('./uswrf10dy.dat'):
    os.remove('./uswrf10dy.dat')
    
# uswrf.sfc.gauss.XXXX.ncフィアルを読んで１０日平均を取ってバイナリファイルに保存
with open('uswrf10dy.dat',mode='a') as f:
    for year in range(1948,2024): # 1948-2023
        month10dy = xr.DataArray(np.zeros((3,94,192))) # コピー用の配列を初期化
        uswrfnc = '01_ESTOC_ForcingData/NCEP_NCAR_Forcing/NCEP/uswrf.sfc.gauss.'+str(year)+'.nc'
        uswrfnc = xr.open_dataset(uswrfnc)['uswrf'].load()
    
        for x in month:
            # 範囲指定用の文字列構成
            start1 = str(year) + '-' + str(x) + '-1'
            end1 = str(year) + '-' + str(x) + '-10'
            start2 = str(year) + '-' + str(x) + '-11'
            end2 = str(year) + '-' + str(x) + '-20'
            start3 = str(year) + '-' + str(x) + '-21'
    
            if x == 12:
                nextmonth = 1
                lastday = str(year+1) + '-' + str(nextmonth) + '-1'
            else:
                nextmonth = x + 1
                lastday = str(year) + '-' + str(nextmonth) + '-1'

            end3 = datetime.datetime.strptime(lastday,'%Y-%m-%d') - datetime.timedelta(days = 1)
            end3 = end3.strftime('%Y-%m-%d')

            # 上中下旬に分けて平均を取る（とりあえずmean重みなし）
            data_mean_10dy1 = uswrfnc.loc[start1:end1].mean(dim='time')
            data_mean_10dy2 = uswrfnc.loc[start2:end2].mean(dim='time')
            data_mean_10dy3 = uswrfnc.loc[start3:end3].mean(dim='time')

            #月毎に10日区切りで3つ作ったら保存用の配列に代入
            temp = np.stack([data_mean_10dy1,data_mean_10dy2,data_mean_10dy3],0) # tempは1ヶ月分
            month10dy = np.append(month10dy,temp,axis=0) # 1ヶ月分のtempを順番にappendしていく（70年分）
            
    
        uswrfyear= month10dy[3:] # 最初の3つは0なので消しておく
        uswrfyear.tofile(f)

print('end')


# In[19]:


uswrf = np.fromfile('./uswrf10dy.dat').reshape(2736,94,192)

## delete 2023.2-2023.12
uswrf = uswrf[:2703,:,:]

if os.path.isfile('./uswrf10dy.dat'):
    os.remove('./uswrf10dy.dat')

with open('uswrf10dy.dat' , mode='a') as f:
    uswrf.tofile(f)


# In[20]:


# gabage collecter
del data_mean_10dy1,data_mean_10dy3,data_mean_10dy2
del uswrfyear
del month10dy,temp
gc.collect()


# In[ ]:





# In[ ]:





# In[21]:


if os.path.isfile('./shtfl10dy.dat'):
    os.remove('./shtfl10dy.dat')

# shtfl.sfc.gauss.XXXX.ncファイルを読んで１０日平均を取ってバイナリファイルに保存
with open('shtfl10dy.dat',mode='a') as f:
    for year in range(1948,2024): # 1948-2023
        month10dy = xr.DataArray(np.zeros((3,94,192))) # コピー用の配列を初期化
        shtflnc = '01_ESTOC_ForcingData/NCEP_NCAR_Forcing/NCEP/shtfl.sfc.gauss.'+str(year)+'.nc'
        shtflnc = xr.open_dataset(shtflnc)['shtfl'].load()
    
        for x in month:
            # 範囲指定用の文字列構成
            start1 = str(year) + '-' + str(x) + '-1'
            end1 = str(year) + '-' + str(x) + '-10'
            start2 = str(year) + '-' + str(x) + '-11'
            end2 = str(year) + '-' + str(x) + '-20'
            start3 = str(year) + '-' + str(x) + '-21'
    
            if x == 12:
                nextmonth = 1
                lastday = str(year+1) + '-' + str(nextmonth) + '-1'
            else:
                nextmonth = x + 1
                lastday = str(year) + '-' + str(nextmonth) + '-1'

            end3 = datetime.datetime.strptime(lastday,'%Y-%m-%d') - datetime.timedelta(days = 1)
            end3 = end3.strftime('%Y-%m-%d')

            # 上中下旬に分けて平均を取る（とりあえずmean重みなし）
            data_mean_10dy1 = shtflnc.loc[start1:end1].mean(dim='time')
            data_mean_10dy2 = shtflnc.loc[start2:end2].mean(dim='time')
            data_mean_10dy3 = shtflnc.loc[start3:end3].mean(dim='time')

            #月毎に10日区切りで3つ作ったら保存用の配列に代入
            temp = np.stack([data_mean_10dy1,data_mean_10dy2,data_mean_10dy3],0) # tempは1ヶ月分
            month10dy = np.append(month10dy,temp,axis=0) # 1ヶ月分のtempを順番にappendしていく（70年分）
                        
        shtflyear= month10dy[3:] # 最初の3つは0なので消しておく
        shtflyear.tofile(f)

print('end')


# In[22]:


shtfl = np.fromfile('./shtfl10dy.dat').reshape(2736,94,192)

## delete 2023.2-2023.12
shtfl= shtfl[:2703,:,:]

if os.path.isfile('./shtfl10dy.dat'):
    os.remove('./shtfl10dy.dat')

with open('shtfl10dy.dat' , mode='a') as f:
    shtfl.tofile(f)


# In[23]:


# gabage collecter
del data_mean_10dy1,data_mean_10dy3,data_mean_10dy2
del shtflyear
del month10dy,temp
gc.collect()


# In[24]:


if os.path.isfile('./uflx10dy.dat'):
    os.remove('./uflx10dy.dat')

# uflx.sfc.gauss.XXXX.ncファイルを読んで１０日平均を取ってバイナリファイルに保存
with open('uflx10dy.dat',mode='a') as f:
    for year in range(1948,2024): # 1948-2023
        month10dy = xr.DataArray(np.zeros((3,94,192))) # コピー用の配列を初期化
        uflxnc = '01_ESTOC_ForcingData/NCEP_NCAR_Forcing/NCEP/uflx.sfc.gauss.'+str(year)+'.nc'
        uflxnc = xr.open_dataset(uflxnc)['uflx'].load()
    
        for x in month:
            # 範囲指定用の文字列構成
            start1 = str(year) + '-' + str(x) + '-1'
            end1 = str(year) + '-' + str(x) + '-10'
            start2 = str(year) + '-' + str(x) + '-11'
            end2 = str(year) + '-' + str(x) + '-20'
            start3 = str(year) + '-' + str(x) + '-21'
    
            if x == 12:
                nextmonth = 1
                lastday = str(year+1) + '-' + str(nextmonth) + '-1'
            else:
                nextmonth = x + 1
                lastday = str(year) + '-' + str(nextmonth) + '-1'

            end3 = datetime.datetime.strptime(lastday,'%Y-%m-%d') - datetime.timedelta(days = 1)
            end3 = end3.strftime('%Y-%m-%d')

            # 上中下旬に分けて平均を取る（とりあえずmean重みなし）
            data_mean_10dy1 = uflxnc.loc[start1:end1].mean(dim='time')
            data_mean_10dy2 = uflxnc.loc[start2:end2].mean(dim='time')
            data_mean_10dy3 = uflxnc.loc[start3:end3].mean(dim='time')

            #月毎に10日区切りで3つ作ったら保存用の配列に代入
            temp = np.stack([data_mean_10dy1,data_mean_10dy2,data_mean_10dy3],0) # tempは1ヶ月分
            month10dy = np.append(month10dy,temp,axis=0) # 1ヶ月分のtempを順番にappendしていく（70年分）
                        

        uflxyear= month10dy[3:] # 最初の3つは0なので消しておく
        uflxyear.tofile(f)

print('end')


# In[25]:


uflx = np.fromfile('./uflx10dy.dat').reshape(2736,94,192)

## delete 2023.2-2023.12
uflx = uflx[:2703,:,:]

if os.path.isfile('./uflx10dy.dat'):
    os.remove('./uflx10dy.dat')

with open('uflx10dy.dat' , mode='a') as f:
    uflx.tofile(f)


# In[26]:


# gabage collecter
del data_mean_10dy1,data_mean_10dy3,data_mean_10dy2
del uflxyear
del month10dy,temp
gc.collect()


# In[27]:


# vflx.sfc.gauss.XXXX.ncファイルを読んで１０日平均を取ってバイナリファイルに保存
if os.path.isfile('./vflx10dy.dat'):
    os.remove('./vflx10dy.dat')

with open('vflx10dy.dat',mode='a') as f:
    for year in range(1948,2024): # 1948-2023
        month10dy = xr.DataArray(np.zeros((3,94,192))) # コピー用の配列を初期化
        vflxnc = '01_ESTOC_ForcingData/NCEP_NCAR_Forcing/NCEP/vflx.sfc.gauss.'+str(year)+'.nc'
        vflxnc = xr.open_dataset(vflxnc)['vflx'].load()
    
        for x in month:
            # 範囲指定用の文字列構成
            start1 = str(year) + '-' + str(x) + '-1'
            end1 = str(year) + '-' + str(x) + '-10'
            start2 = str(year) + '-' + str(x) + '-11'
            end2 = str(year) + '-' + str(x) + '-20'
            start3 = str(year) + '-' + str(x) + '-21'
    
            if x == 12:
                nextmonth = 1
                lastday = str(year+1) + '-' + str(nextmonth) + '-1'
            else:
                nextmonth = x + 1
                lastday = str(year) + '-' + str(nextmonth) + '-1'

            end3 = datetime.datetime.strptime(lastday,'%Y-%m-%d') - datetime.timedelta(days = 1)
            end3 = end3.strftime('%Y-%m-%d')

            # 上中下旬に分けて平均を取る（とりあえずmean重みなし）
            data_mean_10dy1 = vflxnc.loc[start1:end1].mean(dim='time')
            data_mean_10dy2 = vflxnc.loc[start2:end2].mean(dim='time')
            data_mean_10dy3 = vflxnc.loc[start3:end3].mean(dim='time')

            #月毎に10日区切りで3つ作ったら保存用の配列に代入
            temp = np.stack([data_mean_10dy1,data_mean_10dy2,data_mean_10dy3],0) # tempは1ヶ月分
            month10dy = np.append(month10dy,temp,axis=0) # 1ヶ月分のtempを順番にappendしていく（70年分）
                        

    
        vflxncyear= month10dy[3:] # 最初の3つは0なので消しておく
        vflxncyear.tofile(f)

print('end')


# In[28]:


vflx = np.fromfile('./vflx10dy.dat').reshape(2736,94,192)

## delete 2023.2-2023.12
vflx = vflx[:2703,:,:]

if os.path.isfile('./vflx10dy.dat'):
    os.remove('./vflx10dy.dat')

with open('vflx10dy.dat' , mode='a') as f:
    vflx.tofile(f)


# In[29]:


# gabage collecter
del data_mean_10dy1,data_mean_10dy3,data_mean_10dy2
del vflxyear
del month10dy,temp
gc.collect()


# In[28]:


# この時点で必要なデータ
#####
# 10日平均の9ファイル
# shapeは(36,94,192) 1ヶ月を10日区切りで
# dlwrf10dy.dat , dswrf10dy.dat , lhtfl10dy.dat
# prate10dy.dat , shtfl10dy.dat , uflx10dy.dat
# ulwrf10dy.dat , uswrf10dy.dat , vflx10dy.dat 
#####
# land.06.ft.bigファイルを読み込んだ land (94,192)
# prev_climate7902.nc2.bigを読み込んだ climate (12,94,192)



# In[ ]:


## 以下デバック用セル


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[16]:


hoge = np.fromfile('dlwrf10dy.dat').reshape(36,94,192)


# In[2]:


# 全期間を１ヶ月毎にスライスしておく
# 後ほど作画のために必要な部分
Jan = xr.DataArray(np.zeros((3,94,192)))
Feb=Mer=Apr=May=Jun=Jul=Aug=Sep=Oct=Nov=Dec = xr.DataArray(np.zeros((3,94,192)))

for i in range(70):
    if i == 0:
        temp = lhtflyear[0:3]
        temp2 = lhtflyear[3:3]
        temp3 = lhtflyear[6:3]
        temp4 = lhtflyear[9:3]
        temp5 = lhtflyear[12:3]
        temp6 = lhtflyear[15:3]
        temp7 = lhtflyear[18:3]
        temp8 = lhtflyear[21:3]
        temp9 = lhtflyear[24:3]
        tempa = lhtflyear[27:3]
        tempb = lhtflyear[30:3]
        tempc= lhtflyear[33:3]
    else:
        temp = lhtflyear[(i*36):(i*36+3)]
        temp2 = lhtflyear[(3+ i*36):(3+ i*36+3)]
        temp3 = lhtflyear[(6+ i*36):(6+ i*36+3)]
        temp4 = lhtflyear[(9+ i*36):(9+ i*36+3)]
        temp5 = lhtflyear[(12+ i*36):(12+ i*36+3)]
        temp6 = lhtflyear[(15+ i*36):(15+ i*36+3)]
        temp7 = lhtflyear[(18+ i*36):(18+ i*36+3)]
        temp8 = lhtflyear[(21+ i*36):(21+ i*36+3)]
        temp9 = lhtflyear[(24+ i*36):(24+ i*36+3)]
        tempa = lhtflyear[(27+ i*36):(27+ i*36+3)]
        tempb = lhtflyear[(30+ i*36):(30+ i*36+3)]
        tempc = lhtflyear[(33+ i*36):(33+ i*36+3)]

    Jan = np.append(Jan,temp,axis=0)
    Feb = np.append(Feb,temp2,axis=0)
    Mer = np.append(Mer,temp3,axis=0)
    Apr = np.append(Apr,temp4,axis=0)
    May = np.append(May,temp5,axis=0)
    Jun = np.append(Jun,temp6,axis=0)
    Jul = np.append(Jul,temp7,axis=0)
    Aug = np.append(Aug,temp8,axis=0)
    Sep = np.append(Sep,temp9,axis=0)
    Oct = np.append(Oct,tempa,axis=0)
    Nov = np.append(Nov,tempb,axis=0)
    Dec = np.append(Dec,tempc,axis=0)


# In[13]:


from scipy import interpolate

y = np.arange(210)

interpolate.interp2d(Jan[:,0,0])


# In[30]:


from matplotlib.pyplot import figure
fig = figure(figsize=(20,10))
plot_prate = lhtflyear[3,:,:]
plt.imshow(plot_prate)
tep = monname[0]


# In[45]:


plt.imshow(climate[7,:])


# In[9]:


for x in month:
    # 範囲指定用の文字列構成
    start1 = str(year) + '-' + str(x) + '-1'
    end1 = str(year) + '-' + str(x) + '-10'
    start2 = str(year) + '-' + str(x) + '-11'
    end2 = str(year) + '-' + str(x) + '-20'
    start3 = str(year) + '-' + str(x) + '-21'
    
    if x == 12:
        nextmonth = 1
        lastday = str(year+1) + '-' + str(nextmonth) + '-1'
    else:
        nextmonth = x + 1
        lastday = str(year) + '-' + str(nextmonth) + '-1'

    end3 = datetime.datetime.strptime(lastday,'%Y-%m-%d') - datetime.timedelta(days = 1)
    end3 = end3.strftime('%Y-%m-%d')
    # print(end3)
    data_mean_10dy1 = data.loc[start1:end1].mean(dim='time')
    data_mean_10dy2 = data.loc[start2:end2].mean(dim='time')
    data_mean_10dy3 = data.loc[start3:end3].mean(dim='time')

    # 月毎に10日区切りで3つ作ったら保存用の配列に代入
    hoge = np.vstack((np.vstack((data_mean_10dy1,data_mean_10dy2)),data_mean_10dy3)) # これは1ヶ月分
    print(end3)

#print(data_mean_10dy1.shape)
#print(data_mean_10dy2)
#print(data_mean_10dy3)


# In[5]:


# datatimeindexを使って月とか週、曜日でも平均取れる
key = data['time'].dt.month
data.groupby(key).mean(dim='time')


# In[6]:


clevels = np.arange(-20,50,5) # 塗りつぶし用

# 描画
fig = plt.figure(figsize=(8,5))

# PlateCaree: 正距円筒図法, central_longitude: 図の中心の経度
ax = fig.add_subplot(111,projection=ccrs.PlateCarree(central_longitude=180))

# (2) 図を描く
data_mean_10dy.plot.contourf(ax=ax,transform=ccrs.PlateCarree(),levels=clevels,cmap='ocean',cbar_kwargs={'label':'Downward Longwave Radiation Flux at surface (W/m$^{2}$)'})
# transform = にはデータ自身の座標系を指定する; 大抵の場合ccrs.PlateCarree()としておけば問題ない

# (3) 目盛り
xticks = np.arange(13)*30
yticks = -90 + np.arange(7)*30
ax.set_xticks(xticks,crs=ccrs.PlateCarree())
ax.set_yticks(yticks,crs=ccrs.PlateCarree())
##　地図投影用の書式を設定
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)


# (4) 描画領域の設定
ax.set_extent([60,180,0,210],crs=ccrs.PlateCarree())


# (5) オプション
ax.coastlines() #海岸線
ax.gridlines(draw_labels=False) #罫線：ラベルはすでに上で描いたので"False"

plt.show()


# In[7]:


ncfile = '01_ESTOC_ForcingData/NCEP_NCAR_Forcing/NCEP/dlwrf.sfc.gauss.1949.nc'
nc2 = xr.open_dataset(ncfile)
print(nc2)

