#!/usr/bin/env python
# coding: utf-8

# In[1]:


import xarray as xr
import inspect
import pandas as pd
import numpy as np
import datetime
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.util as cutil
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import os

month = [1,2,3,4,5,6,7,8,9,10,11,12]
prateyear = xr.DataArray()
#month10dy = xr.DataArray(np.zeros((3,94,192))) # ループの中でやらないとどんどん増える


# prate.sfc.gauss.XXXX.nc読み込み、１０日毎に単純平均を取ったらバイナリファイルprate10dy.datに保存
# 他にdlwrf,lhtfl,shtfl,ulwrf,uflx,uswrf,vflxがある
# lhtflは次のFreshWaterFlux作成でも使う

##ファイルがあるときは一旦削除する
if os.path.isfile('prate10dy.dat'):
    os.remove('prate10dy.dat')

with open('prate10dy.dat',mode='a') as f:
    for year in range(1948,2022): # 1948-2021
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


# In[2]:


# 要素削除とガベージコレクター
import gc

del data_mean_10dy1,data_mean_10dy3,data_mean_10dy2
del month10dy,pratenc,prateyear,temp

gc.collect()


# In[3]:


##ファイルがあるときは一旦削除する
if os.path.isfile('lhtfl10dy.dat'):
    os.remove('lhtfl10dy.dat')


# In[4]:



# 上と同様にlhtfl.sfc.gauss.XXXX.ncを読み込んでバイナリファイルlhtfl10dy.datに保存
lhtflyear = xr.DataArray()

## ファイル存在チェック入れる
with open('lhtfl10dy.dat',mode='a') as f:
    for year in range(1948,2022): # 1948-2021
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


# In[25]:


##fresh = (prate - lhtfl/2.5e6)*0.1
#plt.imshow((prate[2663,:,:]-lhtfl[2663,:,:]/2.5e6)*0.1 ,vmin=-1*10**-6,vmax=1*10**-6)
#plt.savefig('miss.png')


# In[5]:


# gabage collcter
del data_mean_10dy1,data_mean_10dy3,data_mean_10dy2
del lhtflyear
del month10dy,temp
gc.collect()


# In[6]:


##ファイルがあるときは一旦削除する
if os.path.isfile('dswrf10dy.dat'):
    os.remove('dswrf10dy.dat')


# In[7]:


# dswrf.sfc.gauss.XXXX.ncを読み込んで１０日平均を取ってバイナリファイルに一度保存
dswrfyear = dlwrfyear = uswrfyear = ulwrfyear = shtflyear = uflxyear = vflxyear = xr.DataArray()
with open('dswrf10dy.dat',mode='a') as f:
    for year in range(1948,2022): # 1948-2021
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


# In[8]:


# gabage collecter
del data_mean_10dy1,data_mean_10dy2,data_mean_10dy3
del dswrfyear
del month10dy,temp
gc.collect()


# In[9]:


##ファイルがあるときは一旦削除する
if os.path.isfile('dlwrf10dy.dat'):
    os.remove('dlwrf10dy.dat')


# In[10]:


# dlwrf.sfc.gauss.XXXX.ncファイルを読んで１０日平均を取ってバイナリファイルに保存
with open('dlwrf10dy.dat',mode='a') as f:
    for year in range(1948,2022): # 1948-2021
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


# In[11]:


# gabage collecter
del data_mean_10dy1,data_mean_10dy3,data_mean_10dy2
del dlwrfyear
del month10dy,temp
gc.collect()


# In[12]:


##ファイルがあるときは一旦削除する
if os.path.isfile('ulwrf10dy.dat'):
    os.remove('ulwrf10dy.dat')


# In[13]:


# ulwrf.sfc.gauss.XXXX.ncファイルを読んで１０日平均を取ってバイナリファイルに保存
with open('ulwrf10dy.dat',mode='a') as f:
    for year in range(1948,2022): # 1948-2021
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


# In[14]:


# gabage collecter
del data_mean_10dy1,data_mean_10dy3,data_mean_10dy2
del ulwrfyear
del month10dy,temp
gc.collect()


# In[15]:


##ファイルがあるときは一旦削除する
if os.path.isfile('uswrf10dy.dat'):
    os.remove('uswrf10dy.dat')


# In[16]:


# uswrf.sfc.gauss.XXXX.ncフィアルを読んで１０日平均を取ってバイナリファイルに保存
with open('uswrf10dy.dat',mode='a') as f:
    for year in range(1948,2022): # 1948-2021
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


# In[17]:


# gabage collecter
del data_mean_10dy1,data_mean_10dy3,data_mean_10dy2
del uswrfyear
del month10dy,temp
gc.collect()


# In[18]:


##ファイルがあるときは一旦削除する
if os.path.isfile('shtfl10dy.dat'):
    os.remove('shtfl10dy.dat')


# In[19]:


# shtfl.sfc.gauss.XXXX.ncファイルを読んで１０日平均を取ってバイナリファイルに保存
with open('shtfl10dy.dat',mode='a') as f:
    for year in range(1948,2022): # 1948-2021
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


# In[20]:


# gabage collecter
del data_mean_10dy1,data_mean_10dy3,data_mean_10dy2
del shtflyear
del month10dy,temp
gc.collect()


# In[21]:


##ファイルがあるときは一旦削除する
if os.path.isfile('uflx10dy.dat'):
    os.remove('uflx10dy.dat')


# In[22]:


# uflx.sfc.gauss.XXXX.ncファイルを読んで１０日平均を取ってバイナリファイルに保存
with open('uflx10dy.dat',mode='a') as f:
    for year in range(1948,2022): # 1948-2021
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


# In[23]:


# gabage collecter
del data_mean_10dy1,data_mean_10dy3,data_mean_10dy2
del uflxyear
del month10dy,temp
gc.collect()


# In[24]:


##ファイルがあるときは一旦削除する
if os.path.isfile('vflx10dy.dat'):
    os.remove('vflx10dy.dat')


# In[25]:


# vflx.sfc.gauss.XXXX.ncファイルを読んで１０日平均を取ってバイナリファイルに保存
with open('vflx10dy.dat',mode='a') as f:
    for year in range(1948,2022): # 1948-2021
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


# In[26]:


# gabage collecter
del data_mean_10dy1,data_mean_10dy3,data_mean_10dy2
del vflxyear
del month10dy,temp
gc.collect()


# In[27]:


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




