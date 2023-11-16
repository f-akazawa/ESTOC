#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import xarray as xr
import numpy as np
from scipy import interpolate
import pandas as pd
import datetime
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
#import cartopy.crs as ccrs
import ctypes as c
import struct
import os
import glob
import gc


# In[2]:


## landfile 読み込み
## 0が陸、1が海
## readsize(64800) からshapeは180*360と判断
land = np.fromfile('downloads/lstags.onedeg.dat',dtype='int32').reshape(180,360)

## ほしいデータは天地逆

## フリップするとわかりやすい図になる
##land = np.flipud(land)

## ヘッダのバイト数と読み込むバイト数を指定
# oisst読み込み用
bytes = 8*4
rect = 360*180


# In[33]:


# ファイルがあったら一旦削除する
# 全部消してるのでデバッグ時は注意！！
templist = glob.glob('work/*.npy')
for fname in templist:
    if os.path.isfile(fname):
        os.remove(fname)



# In[34]:


## ersst.v5.yyyymm.ncファイル読み込み
## 1981以降は上のセル、OISSTのweeklyを使う
## ESTOCに期間を合わせるため１９４８年１月からのデータを利用する
year = 1948
month = 1
# 1９４８ ～1981
# 1981/11からOISSTデータがある
while year <= 1981:
    month = 1
    while month <=12:
        if month < 10:
            ym = str(year)+'0'+str(month)
        else:
            ym = str(year)+str(month)
            
        filepath = 'downloads/ersst.v5.'+ym+'.nc'
        #print(filepath)
        
        ersst = xr.open_dataset(filepath)
        sst = ersst['sst']
        fname = 'work/ersst'+str(ym)
        ## SSTを確認用にバイナリで一旦保存
        np.save(fname,sst[0,0,:,:])
        
        
        month +=1
    year +=1


# In[3]:


## 内挿するための座標値確認のためにNCファイルからLat,Lonを拾う
ersst_param = xr.open_dataset('downloads/ersst.v5.198101.nc')

orig_lat = np.array(ersst_param['lat'])
#orig_lat = np.append(np.array(ersst_param['lat']),90)
orig_lon = np.append(np.array(ersst_param['lon']),360) # 0～358なので後ろに０番目を足しておく

# 内挿するための座標値、これもNCファイルからの読み取りで決定(oisstのサイズ)
xi,yi = np.mgrid[0.5:360:1,-89.5:90:1]


# In[46]:


# ersstの１０日平均をスタックする配列を初期化
ersst_10dy_stack = np.zeros((89,180,1)) # ()は２つ必要、多次元配列はタプルを指定するため

year = 1948
month = 1

# 内挿するポイントの数
num_interp_points = 30

# 内挿結果を格納する新しい2次元データの形状を計算する
interp_shape = (180,360, num_interp_points)

# 内挿結果を格納する新しい2次元データを作成する
interpolated_data = np.zeros((89,180,num_interp_points))

while year <=1981:
    month = 1
    while month <=12:
        if month <10:
            ym = str(year)+'0'+str(month)
        else:
            ym = str(year)+str(month)
            
        fname = './work/ersst'+ym
        if month == 12:
            bname = './work/ersst'+str(year+1)+'01'
        else:
            bname= './work/ersst'+str(int(ym)+1)
        month +=1
        #前の月（Forward）と後ろの月（Back)のファイル名がこのループの中でできている
        forward = np.load(fname+'.npy')
        back = np.load(bname+'.npy')
        
        # ERSSTデータにはNanがあるので0で埋める（8/7）
        ## ERSSTは温度データで０が含まれるので陸地Nanは-1.0e33で埋める
        # -1.0*10**33 と書いたが、指数表記も行けるので素直に-1.0e33としても良かったかも？（２０２３・９・１２）
        np.nan_to_num(forward,nan=-1.0e33,copy=False)
        np.nan_to_num(back,nan=-1.0e33,copy=False)
        
       
        fmonth = forward
        bmonth = back
        #fmonthとbmonthの間を30日に増やして10日平均を3つ作る

        # 内挿を実行する
        for i in range(fmonth.shape[0]):
            for j in range(fmonth.shape[1]):
                interpolated_data[i, j, :] = np.linspace(fmonth[i, j], bmonth[i, j], num_interp_points)

                
        # 10daily作成
        dy1 = interpolated_data[:,:,0:10].mean(axis=2)
        dy2 = interpolated_data[:,:,10:20].mean(axis=2)
        dy3 = interpolated_data[:,:,20:30].mean(axis=2)

        dymean = np.stack([dy1,dy2,dy3],2)
        
        # １０dailyを一旦ファイルにバイナリ保存
        #np.save('./work/10dymean'+ym, dymean)
        
        # どんどんスタックしていく
        ersst_10dy_stack = np.append(ersst_10dy_stack,dymean,axis=2)
        
        # 198111まで利用するのでそこに到達したらBreak
        if fname == './work/ersst198111':
            break;
            
    year += 1

print('end')


# In[47]:


# ersst_stack[:,:,0]は初期化で作ったので削除する
# 180,360,4602(1854/1~1981/11)
ersst_10dy_stack = ersst_10dy_stack[:,:,1:]

# ForcingDataと並びが違うので並べ替え
# [時間、縦Lat,横Lon]
ersst_10dy_stack = ersst_10dy_stack.transpose(2,0,1)



# In[48]:


# ERSSTの経度データは０度から２度刻み、３５８度までなので後ろに０番目のデータを追加して３６０度の値とする（他も同様、画面端処理のため）
addzero = ersst_10dy_stack[:,:,0].reshape(ersst_10dy_stack.shape[0],ersst_10dy_stack.shape[1],1) # 0番目の列を取り出して2次元＞3次元に変換
ersst181 = np.append(ersst_10dy_stack,addzero,axis=2) # axis=0奥行き方向、1行方向、2列方向


# In[49]:


# デバック用にファイル保存
if os.path.isfile('./work/ersst181.npy'):
    os.remove('./work/ersst181.npy')
    
np.save('./work/ersst181',ersst181)


# In[ ]:





# In[56]:


num_nan = np.count_nonzero(ersst181[0,:,:]  -1.0e33)
print(num_nan)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


## 穴埋めデバッグの場合は以下からスタート


# In[4]:


ersst181 = np.load('./work/ersst181.npy')


# In[4]:


# ERSSTのlandmaskを作る
iland = np.load('work/ersst194801.npy')

ilandzero = iland[:,0]

iland = np.hstack((iland,ilandzero.reshape(-1,1))) # 画面端処理のため181番目に0番目を入れる


# In[5]:


# ERSST の海陸データilandは陸地がNan、海が数値（温度）
iland= (np.isnan(iland).copy())

# この時点でilandは陸True,海Falseになっている


# In[6]:


# forcingとデータ形式を合わせるbool -> 0/1
# 陸地＝０，海＝１

rev_iland = np.where(iland == True,0,1)


# In[9]:


# ERSSTのランドマスクをESTOCランドマスクに合わせる内挿処理
ersstmask = interpolate.interp2d(orig_lon,orig_lat,rev_iland,kind='linear')(xi[:,0],yi[0,:])


# In[44]:


## ERSSTのサイズをESTOCに合わせる内挿処理
# 89*180サイズなので180＊360に内挿
temp = []
for i in range(ersst181.shape[0]):
    slicedays = ersst181[i,:,:]
    
    zi = interpolate.interp2d(orig_lon,orig_lat,slicedays,kind='linear')(xi[:,0],yi[0,:])
    
    interp_ersst = np.append(temp,zi).reshape([i+1,180,360])
    
    temp = interp_ersst
    
    if i % 1000 == 0:
        print(i)

print('end')


# In[45]:


if os.path.isfile('work/interp_ersst.npy'):
    os.remove('work/interp_ersst.npy')

np.save('work/interp_ersst',interp_ersst)


# In[10]:


# debug file read
interp_ersst = np.load('./work/interp_ersst.npy')


# In[11]:


## 画面端の処理
# 画面端の処理をするために横に1列増やす、左右必要
addzero = interp_ersst[:,:,0].reshape(interp_ersst.shape[0],interp_ersst.shape[1],1) # 0番目の列の値を抜き出して3次元に直す
add359 = interp_ersst[:,:,359].reshape(interp_ersst.shape[0],interp_ersst.shape[1],1) # 360番目の値を抜き出して3次元に直す

ersst362 = np.append(add359,(np.append(interp_ersst,addzero,axis=2)),axis=2)


# In[ ]:





# In[ ]:





# In[ ]:





# In[12]:


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


# In[13]:


estocweight = np.where(estoclandmask == True,0,1)


# In[14]:


## mask の中身はBool
mask = ((estocweight == 1) & (ersstmask < 1-10**-10)) 
landindex =  np.where(mask == 1)
######  ersstmask , estocmask　で＆を取ったmaskとlandindex、ここまではFluxと見た目ほぼ同じ８/２８


# In[15]:


ersst362_old = ersst362.copy()


# In[16]:


####
## landindexの中身０から(mask==0として確認　5/30)
####
for i in enumerate(landindex[0]): # i = loop index
    lat = (landindex[0][i[0]])
    lon = (landindex[1][i[0]])
    estocweight[lat,lon] = 0 # 6/26 =0に戻す
    ## ここのESTOCWeightを確認する6/26
    ## この時点ではmask == estocwaightになっている。
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
estocweight_old = estocweight.copy() # 計算用に必要なので作っておく 
estocweight_orig = estocweight.copy() # 同上


# In[20]:


## 畳み込み計算部

for index in range(ersst362.shape[0]): # ersst362.shape[0] = 1221
#for index in range(10): # for debug

    estocweight = estocweight_orig.copy()
    
    for counter in range(1000):# 1000回繰り返す（又は閾値を超えたらループ終了
        resmax = -10000000000
        ersst362_old[index,:,:] = ersst362[index,:,:] ## コレも必要
        
        estocweight_old[:,:] = estocweight[:,:] # フラグ１の場所を更新
        
        ## estocweight0,360 に更新（右と左両方追加、323)
        left = estocweight[:,1]
        right = estocweight[:,360]
        estocweight_old[:,0] = right # 0の前に360の値を更新
        estocweight_old[:,361] = left #360の先に1の値を更新
        ## 同じことをersst362でもやる
        ersst362_old[index,:,0] = ersst362[index,:,360]
        ersst362_old[index,:,361] = ersst362[index,:,1]
        
        ## ここがズレてるので注意
        for i in enumerate(landindex[0]): # i = loop index
            lat = (landindex[0][i[0]])
            lon = (landindex[1][i[0]])+1 # 6/9
            ## ループの1回目は０であるはず、ESTOCWEIGHT lat,lon
            # 1箇所だけlon=360がある
            
            ## latも値を確認0、179があるとだめ
            calcflag = estocweight_old[lat-1,lon] + estocweight_old[lat+1,lon] + estocweight_old[lat,lon-1] + estocweight_old[lat,lon+1] 
            #if lon != 360:
            calcdata = (ersst362_old[index,lat-1,lon]*estocweight_old[lat-1,lon] + ersst362_old[index,lat+1,lon]*estocweight_old[lat+1,lon]\
                            +ersst362_old[index,lat,lon-1]*estocweight_old[lat,lon-1] + ersst362_old[index,lat,lon+1]*estocweight_old[lat,lon+1])
                
            if calcflag > 0:
                calcweight = calcdata/(calcflag)
                res = abs(ersst362_old[index,lat,lon]-calcweight)
                ersst362[index,lat,lon] = calcweight
                estocweight[lat,lon] = 1
                resmax = max(resmax,res)
           
        if  0.0000001 > resmax:
            #print('index=',index , 'counter=',counter,'resmax=',resmax)
            break


## ファイル書き出しを別セルにする（遅いので）
ersst362[:,:,1:361].tofile('ERSST_only.dat')
print('end')


# In[18]:


#### ERSST ここまでで完成
## テスト表示用に読み込み
ersstonly = ersst362[:,:,1:361]


# In[ ]:


## 以下はデバック用コード
##############################################


