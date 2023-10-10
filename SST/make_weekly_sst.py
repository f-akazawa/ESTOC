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


# In[8]:


## 内挿するための座標値確認のためにNCファイルからLat,Lonを拾う
ersst_param = xr.open_dataset('downloads/ersst.v5.198101.nc')

orig_lat = np.array(ersst_param['lat'])
#orig_lat = np.append(np.array(ersst_param['lat']),90)
orig_lon = np.append(np.array(ersst_param['lon']),360) # 0～358なので後ろに０番目を足しておく

# 内挿するための座標値、これもNCファイルからの読み取りで決定(oisstのサイズ)
xi,yi = np.mgrid[0.5:360:1,-89.5:90:1]


# In[36]:


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
        np.nan_to_num(forward,nan=-1.0*10**33,copy=False)
        np.nan_to_num(back,nan=-1.0*10**33,copy=False)
        
       
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


# In[5]:


# ersst_stack[:,:,0]は初期化で作ったので削除する
# 180,360,4602(1854/1~1981/11)
ersst_10dy_stack = ersst_10dy_stack[:,:,1:]

# ForcingDataと並びが違うので並べ替え
# [時間、縦Lat,横Lon]
ersst_10dy_stack = ersst_10dy_stack.transpose(2,0,1)



# In[38]:


# ERSSTの経度データは０度から２度刻み、３５８度までなので後ろに０番目のデータを追加して３６０度の値とする（他も同様、画面端処理のため）
addzero = ersst_10dy_stack[:,:,0].reshape(ersst_10dy_stack.shape[0],ersst_10dy_stack.shape[1],1) # 0番目の列を取り出して2次元＞3次元に変換
ersst181 = np.append(ersst_10dy_stack,addzero,axis=2) # axis=0奥行き方向、1行方向、2列方向


# In[39]:


# デバック用にファイル保存
if os.path.isfile('./work/ersst181.npy'):
    os.remove('./work/ersst181.npy')
    
np.save('./work/ersst181',ersst181)


# In[ ]:





# In[ ]:


## 穴埋めデバッグの場合は以下からスタート


# In[3]:


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


# In[ ]:





# In[17]:


plt.imshow(np.flipud(estocweight_old),vmin=-10**-6 ,vmax=10**-6)


# In[18]:


plt.imshow(np.flipud(mask))


# In[21]:


plt.imshow(estocweight_orig)


# In[22]:


plt.imshow(ersst362[1,:,:])


# In[27]:


print(np.amax(ersst362))


# In[28]:


print(np.amin(ersst362))


# In[ ]:





# In[ ]:





# In[ ]:





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
            print('index=',index , 'counter=',counter,'resmax=',resmax)
            break


## ファイル書き出しを別セルにする（遅いので）
ersst362[:,:,1:361].tofile('ERSST_only.dat')
print('end')


# In[18]:


#### ERSST ここまでで完成
#### OISSTでもファイルを読んでESTOCでマスクして埋め込み計算をする
ersstonly = ersst362[:,:,1:361]


# In[19]:


plt.imshow(ersstonly[0,:,:])


# In[25]:


plt.imshow(ersstonly[1,:,:])


# In[61]:


plt.imshow(mask)


# In[21]:


plt.imshow(estocweight)


# In[22]:


plt.imshow(estocweight_orig)


# In[ ]:





# In[ ]:





# In[26]:


## OISST V2のみ対応　出港前暫定版






dat = xr.open_dataset('./downloads/sst.oisst.mon.mean.1982.nc')
sst = dat['sst']

## sst['time','lat','lon'] time = 499


# In[23]:


## oisst 10dy stack
oisst_10dy_stack = np.zeros((180,360,1))


# In[24]:


# 内装するポイント
num_interp_points = 30
# 内挿結果を格納する新しい配列
interpolated_data = np.zeros((180,360,num_interp_points))


# In[27]:


## sst[0:493,:,:] time 0（１９８２．１） から493（２０２３・２）まで使って10日平均作る
## 出港前暫定版なのでベタ打ち

for time in range(0,494):
    if time == 0:
        #最初の1981年12月はデータがないので特別
        forward = sst[time,:,:]
        back = sst[time,:,:]
    else:
        forward = sst[time-1,:,:]
        back = sst[time,:,:]
        
    # Naｎを別の値で埋める
    # ERSSTと同じ値で陸地を埋めている、複素数表記が可能だったので-1.0e33でも良いはず
    np.nan_to_num(forward,nan=-1.0*10**33,copy=False)
    np.nan_to_num(back,nan=-1.0*10**33,copy=False)
    
    fmonth = forward
    bmonth = back
    
    #  先月、今月の間を内挿して30日作る＞その後10日平均を計算する
    for i in range(fmonth.shape[0]):
        for j in range(fmonth.shape[1]):
            interpolated_data[i,j,:] = np.linspace(fmonth[i,j],bmonth[i,j],num_interp_points)
            
    # 10daily作成
    dy1 = interpolated_data[:,:,0:10].mean(axis=2)
    dy2 = interpolated_data[:,:,10:20].mean(axis=2)
    dy3 = interpolated_data[:,:,20:30].mean(axis=2)
    
    dymean = np.stack([dy1,dy2,dy3],2)
    
    # 更にスタック
    oisst_10dy_stack = np.append(oisst_10dy_stack,dymean,axis=2)


# In[28]:


np.save('oisst_10dy_stack',oisst_10dy_stack)


# In[30]:


## ERSSTと配列の順番が違うので並び替える
# ERSSTとは並びが違うので並べ替える
oisst_10dy_stack = oisst_10dy_stack.transpose(2,0,1)


# In[ ]:





# In[ ]:





# In[105]:


fmonth.shape[1]


# In[ ]:





# In[ ]:





# In[144]:


# fortran用のデータファイルは各データセクションの前に必ずヘッダがつく
#
#　OISST V2用
#
# OISST V2.1ではフォーマットが変わったので使えない
#
read_bytes = 0 # カウンターの初期化

# スタックしていく配列を初期化
oisst_stack = np.zeros((180,360,1))


## oisst1981 ～oisst2020
year = 1981
while year <= 2020:
    filepath = 'downloads/oisst.'+ str(year)
    
    # 1ループ毎のテンポラリ,カッコの数足りないとエラーになる
    looptemp = xr.DataArray(np.zeros((1,180,360))) 

    with open(filepath,'rb')as f:
    
        while True:
            np.fromfile(f,dtype='>i',count=1) # header 
            rec1 = np.fromfile(f,dtype='>i',count=8) # rec1
            # rec1のsizeが0だったらデータはもう無いのでループを抜ける
            if rec1.size == 0:
                break;

            np.fromfile(f,dtype='>i',count=1) # header
            rec2 = np.fromfile(f,dtype='>f',count=rect).reshape(180,360)
            np.fromfile(f,dtype='>i',count=1) # header
            np.fromfile(f,dtype='>f',count=rect)# rec3
            np.fromfile(f,dtype='>i',count=1) # header
            np.fromfile(f,dtype='>i1',count=rect)# rec4
        
            ## データの区切りにも何やら入っているので読み飛ばし
            np.fromfile(f,dtype='>i',count=4)
        
            read_bytes += 1+8+1+rect+1+rect+1+rect+4
        
            # 3次元配列に２次元の要素をタプルとしてV方向に結合
            looptemp = np.vstack((looptemp,[rec2]))
            
            # rec2を3次元にしてスタック（ersstとshapeを合わせるため）
            rec2 = np.expand_dims(rec2,axis=2)
            oisst_stack = np.append(oisst_stack,rec2,axis=2)
        
    # 1列目は０なので省いて保存する
    fname = './work/oisst'+str(year)

    # 確認するために一旦numpyバイナリでファイルに保存
    
    #np.save(fname,looptemp[1:,:,:])
    
    year +=1
print('end')


# In[145]:


# 0番目削除
oisst_stack = oisst_stack[:,:,1:]

#　ersst と軸はズレているので注意（縦、横、時間）
#　最終的に書き出すときにERSSTと合わせている（時間、縦、横）

if os.path.isfile('./work/oisst_stack.npy'):
    os.remove('./work/oisst_stack.npy')
    
np.save('./work/oisst_stack',oisst_stack)


# In[ ]:


# debug用
oisst_stack = np.load('./work/oisst_stack.npy')


# In[146]:


# オリジナルソースはWeekly > daily変換した後、10日平均をつくている

# 内挿するポイントの数()
num_interp_points = 7
# 内挿結果を格納する新しい2次元データの形状を計算する
interp_shape = (180,360, num_interp_points)
# 内挿結果を格納する新しい2次元データを作成する
interpolated_data = np.zeros(interp_shape)
# dailyに戻したoisstをスタックする配列初期化
oisst_daily_stack = np.zeros((180,360,1))


for index in range(oisst_stack.shape[2]):
                             
    if index == 2024:
        break;
    else:
        fweek = oisst_stack[:,:,index]
        bweek = oisst_stack[:,:,index+1]
    
    for i in range(fweek.shape[0]):
        for j in range(fweek.shape[1]):
            interpolated_data[i, j, :] = np.linspace(fweek[i, j], bweek[i, j], num_interp_points)
    
    oisst_daily_stack = np.append(oisst_daily_stack,interpolated_data,axis=2)

print('end')


# In[147]:


# 0番目を削除
oisst_daily_stack = oisst_daily_stack[:,:,1:]

if os.path.isfile('./work/oisst_daily_stack.npy'):
    os.remove('./work/oisst_daily_stack.npy')

np.save('./work/oisst_daily_stack',oisst_daily_stack)


# In[ ]:


#debug 用
oisst_daily_stack = np.load('./work/oisst_daily_stack.npy')


# In[148]:


# 10日平均を作ってスタック
oisst_10dy_stack = np.zeros((180,360,1))

for index in range(0,oisst_daily_stack.shape[2],10):
    if index >= 0:
        d1 = oisst_daily_stack[:,:,index:index+10].mean(axis=2)
    elif (index +10 > oisst_daily_stack.shape[2]):# 10日足りないところの処理どうする？
        d1 = oisst_daily_stack[:,:,index:index+8]
    
    # 3次元にしてからスタック
    d1 = np.expand_dims(d1,axis=2)
    oisst_10dy_stack = np.append(oisst_10dy_stack,d1,axis=2)
    


# In[149]:


# 0番目を削除してファイル保存
oisst_10dy_stack = oisst_10dy_stack[:,:,1:]

# ERSSTとは並びが違うので並べ替える
oisst_10dy_stack = oisst_10dy_stack.transpose(2,0,1)

if os.path.isfile('./work/oisst_10dy_stack.npy'):
    os.remove('./work/oisst_10dy_stack.npy')

np.save('./work/oisst_10dy_stack',oisst_10dy_stack)


# In[ ]:





# In[ ]:





# In[ ]:


# debug 用
oisst_10dy_stack = np.load('./work/oisst_10dy_stack.npy')


# In[31]:


## 画面端の処理
# 画面端の処理をするために横に1列増やす、左右必要
addzero = oisst_10dy_stack[:,:,0].reshape(oisst_10dy_stack.shape[0],180,1) # 0番目の列の値を抜き出して3次元に直す
add359 = oisst_10dy_stack[:,:,359].reshape(oisst_10dy_stack.shape[0],180,1) # 360番目の値を抜き出して3次元に直す

oisst362 = np.append(add359,(np.append(oisst_10dy_stack,addzero,axis=2)),axis=2)


# In[32]:


if os.path.isfile('./work/oisst362.npy'):
    os.remove('./work/oisst362.npy')

np.save('./work/oisst362',oisst362)


# In[ ]:





# In[ ]:





# In[ ]:





# In[33]:


## OISST ランドマスク作成
## 1が海、0が陸
oisst_landmask = np.fromfile('downloads/lstags.onedeg.dat',dtype='>i',count=rect).reshape(180,360)

# ERSSTと変数名を合わせるだけのコピー
oisstmask = oisst_landmask


# In[34]:


## ESTOCmask再作成
#　ERSSTで使って色々変化しているのでファイルから再作成する
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

estocweight = np.where(estoclandmask == True,0,1)


# In[35]:


## oisstmask 180*360なのでESTOCのサイズに合っているため内装処理はいらない
## mask の中身はBool
mask = ((estocweight == 1) & (oisstmask < 1-10**-10)) 
landindex =  np.where(mask == 1)
######  ersstmask , estocmask　で＆を取ったmaskとlandindex、ここまではFluxと見た目ほぼ同じ８/２８


# In[36]:


oisst362_old = oisst362.copy()


# In[37]:


## ERSSTでも使っているので再作成する

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


# In[38]:


## 畳み込み計算部

for index in range(oisst362.shape[0]): # oisst362.shape[0] = 1417

    estocweight = estocweight_orig.copy()

    if index % 500 == 0:
        print(index)
    
    for counter in range(1000):# 1000回繰り返す（又は閾値を超えたらループ終了
        resmax = -10000000000
        oisst362_old[index,:,:] = oisst362[index,:,:] ## コレも必要
        
        estocweight_old[:,:] = estocweight[:,:] # フラグ１の場所を更新
        
        ## estocweight0,360 に更新（右と左両方追加、3/23)
        left = estocweight[:,1]
        right = estocweight[:,360]
        estocweight_old[:,0] = right # 0の前に360の値を更新
        estocweight_old[:,361] = left #360の先に1の値を更新
        ## 同じことをFresh362でもやる
        oisst362_old[index,:,0] = oisst362[index,:,360]
        oisst362_old[index,:,361] = oisst362[index,:,1]
        
        ## ここがズレてる
        for i in enumerate(landindex[0]): # i = loop index
            lat = (landindex[0][i[0]])
            lon = (landindex[1][i[0]])+1 # 6/9
            ## ループの1回目は０であるはず、ESTOCWEIGHT lat,lon
            # 1箇所だけlon=360がある
            
            ## latも値を確認0、179があるとだめ
            calcflag = estocweight_old[lat-1,lon] + estocweight_old[lat+1,lon] + estocweight_old[lat,lon-1] + estocweight_old[lat,lon+1] 
            #if lon != 360:
            calcdata = (oisst362_old[index,lat-1,lon]*estocweight_old[lat-1,lon] + oisst362_old[index,lat+1,lon]*estocweight_old[lat+1,lon]\
                            +oisst362_old[index,lat,lon-1]*estocweight_old[lat,lon-1] + oisst362_old[index,lat,lon+1]*estocweight_old[lat,lon+1])
                
            if calcflag > 0:
                calcweight = calcdata/(calcflag)
                res = abs(oisst362_old[index,lat,lon]-calcweight)
                oisst362[index,lat,lon] = calcweight
                estocweight[lat,lon] = 1
                resmax = max(resmax,res)
                
        if 0.0000001 > resmax:
            break

## ファイル書き出しを別セルにする（遅いので）
oisst362[:,:,1:361].tofile('OISST_only.dat')
print('end')


# In[39]:


## ERSST_only.dat + OISST_only.dat で完成のハズ・・・
oisstonly = oisst362[:,:,1:361]


# In[21]:


ersstonly = np.fromfile('./ERSST_only.dat').reshape(1221,180,360)
oisstonly = np.fromfile('./OISST_only.dat').reshape(1483,180,360)


# In[40]:


merge = np.concatenate([ersstonly,oisstonly])


# In[43]:


#　確認用リトルエンディアンファイル
merge.tofile('SST_full.dat')


# In[42]:


merge = np.fromfile('SST_full.dat')


# In[ ]:





# In[ ]:





# In[44]:


# バイトオーダーをリトルからビックにスワップする
big_endian = merge.byteswap()


# In[45]:


##　提出ファイルはビックエンディアンで書き出す
with open('SST_big.bin' , 'wb') as f:
    big_endian.tofile(f)
    
    


# In[ ]:





# In[ ]:





# In[65]:


with open('SST_big.bin','rb') as readfile:
    data = np.fromfile(readfile, dtype='>f8' , count = 64800)


# In[66]:


hoge = data.reshape(180,360)


# In[67]:


plt.imshow(hoge)


# In[97]:


np.place(estocmask , estocmask == 100 , 0)


# In[98]:


pltestocmask = estocmask + pltimg


# In[99]:


np.max(pltestocmask)


# In[103]:


from matplotlib.colors import ListedColormap
# 特定の値を白色にするための閾値を設定
threshold_value = 100

# カラーマップを取得（例としてViridisを使用）
cmap = plt.cm.viridis

# カラーマップをコピーして新しいカラーマップを作成
new_cmap_colors = cmap(np.linspace(0, 1, cmap.N))
new_cmap_colors[int(threshold_value * cmap.N):] = [1, 1, 1, 1]  # 白色にする
new_cmap = ListedColormap(new_cmap_colors)

# プロット
plt.imshow(pltestocmask, cmap=new_cmap)
plt.colorbar()

plt.show()


# In[105]:


pltimg.tofile('ersst02.dat')


# In[106]:


(ersst_landmask)


# In[57]:


np.max(ersstmask)


# In[58]:


## ersst_bin_landmaskを内挿してESTOCのサイズに合わせる
ersstmask = interpolate.interp2d(orig_lon,orig_lat,ersst_landmask,kind='linear')(xi[:,0],yi[0,:])


# In[59]:


## 穴埋め計算用のersstweightを作る
## ESTOCが１，ERSSTが１以外の所をマスクで作る
ersstweight = np.where(estoclandmask == True,0,1)
mask = ((estocmask == 1) & (ersstmask == -100) )

landindex = np.where(mask == 1)

for i in enumerate(landindex[0]):
    lat = (landindex[0][i[0]])
    lon = (landindex[1][i[0]])
    ersstweight[lat,lon] = 0


# In[14]:


## ersstweight の左右を増やす
##0の左に３５９の値　３６０の右に元の０の値それぞれ値を入れておく
left = ersstweight[:,0]
right = ersstweight[:,359]

ersstweight = np.hstack([(right.reshape(-1,1)),ersstweight]) # 0の前に360の値を足す
ersstweight = np.hstack([ersstweight,(left.reshape(-1,1))]) #360の先に0の値を足す

# oisstweightはこの時点で180 * 362
# ループ計算用に最初の状態をコピーしておく
ersstweight_orig = ersstweight.copy()
ersstweight_old = ersstweight.copy()
ersst362_old = ersst362.copy()


# In[32]:


np.savetxt('ersstmask.csv',ersstmask,delimiter=',')
np.savetxt('estocmask.csv',estocmask,delimiter=',') # 0が陸、数値が海


# In[61]:


print(np.min(ersstmask))


# In[60]:


plt.imshow(np.flipud(ersstmask))


# In[17]:


plt.imshow(np.flipud(mask))


# In[18]:


plt.imshow(np.flipud(estocmask))


# In[29]:


# landindexは陸地(landmask==True)の座標がタプルのndarray(配列）で返る
# landindex[0] = latitude
# landindex[1] = longitude


#for index in range(ersst362.shape[0]): # 4602
for index in range(3): # 4602
    
    print('index=',index)
        
    ersstweight = ersstweight_orig
    
    for counter in range(1000):# 1000回繰り返す（又は閾値を超えたらループ終了
        resmax = -10000000000
        ersst362_old[index,:,:] = ersst362[index,:,:] ## コレも必要
        
        ersstweight_old[:,:] = ersstweight[:,:] # フラグ１の場所を更新
        
        ## estocweight0,360 に更新（右と左両方追加、3/23)
        ersstweight_old[:,0] = ersstweight[:,360] # 0の前に360の値を更新
        ersstweight_old[:,361] = ersstweight[:,1] #360の先に1の値を更新
        ## 同じことをFresh362でもやる
        ersst362_old[index,:,0] = ersst362[index,:,360]
        ersst362_old[index,:,361] = ersst362[index,:,1]
        
        for i in enumerate(landindex[0]): # i = loop index
            lat = (landindex[0][i[0]])
            lon = (landindex[1][i[0]])+1 # 6/9
            ## ループの1回目は０であるはず、ESTOCWEIGHT lat,lon
            # 1箇所だけlon=360がある
            
            ## latも値を確認0、179があるとだめ
            calcflag = ersstweight_old[lat-1,lon] + ersstweight_old[lat+1,lon] + ersstweight_old[lat,lon-1] + ersstweight_old[lat,lon+1] 
            #if lon != 360:
            calcdata = (ersst362_old[index,lat-1,lon]*ersstweight_old[lat-1,lon] + ersst362_old[index,lat+1,lon]*ersstweight_old[lat+1,lon]\
                            +ersst362_old[index,lat,lon-1]*ersstweight_old[lat,lon-1] + ersst362_old[index,lat,lon+1]*ersstweight_old[lat,lon+1])
                
            if calcflag > 0:
                calcweight = calcdata/(calcflag)
                res = abs(ersst362_old[index,lat,lon]-calcweight)
                ersst362[index,lat,lon] = calcweight
                ersstweight[lat,lon] = 1
                resmax = max(resmax,res)
        

        if 0.5 > resmax: # 閾値は要確認
            print('counter=',counter)
            break

## ファイル書き出し
np.save('./work/ersst_calc_review',ersst362[:,:,1:361])
print('end')


# In[23]:


ersst362 = np.load('./work/ersst_calc_review.npy')
hoge = ersst362[0,:,:]


# In[24]:


hoge.tofile('ersst3.dat')


# In[27]:


## 陸地をNanに戻してプロット
## debugのために行う
## 提出するときにはNanをーのデカい値を入れておくこと
hoge[hoge== 0] = np.nan


# In[28]:


plt.imshow(np.flipud(hoge))
plt.savefig('nanplot.png')


# In[26]:


hoge.tofile('ersst3.dat')


# In[24]:





# In[25]:





# In[3]:





# In[ ]:





# In[33]:





# In[18]:





# In[19]:





# In[20]:





# In[21]:


## 穴埋め計算用のoisstweightを作る
## ESTOCが１，OISSTERSSTが１以外の所をマスクで作る
oisstweight = np.where(estocmask == True,0,1)
mask = (estocmask == 1) & (oisstmask < 1-10**-10)

landindex = np.where(mask == 1)

for i in enumerate(landindex[0]):
    lat = (landindex[0][i[0]])
    lon = (landindex[1][i[0]])
    oisstweight[lat,lon] = 0


# In[22]:





# In[23]:


## oisstweight の左右を増やす
##0の左に３５９の値　３６０の右に元の０の値それぞれ値を入れておく
left = oisstweight[:,0]
right = oisstweight[:,359]

oisstweight = np.hstack([(right.reshape(-1,1)),oisstweight]) # 0の前に360の値を足す
oisstweight = np.hstack([oisstweight,(left.reshape(-1,1))]) #360の先に0の値を足す

# oisstweightはこの時点で180 * 362
# ループ計算用に最初の状態をコピーしておく
oisstweight_orig = oisstweight.copy()
oisstweight_old = oisstweight.copy()
oisst362_old = oisst362.copy()


# In[24]:


# landindexは陸地(landmask==True)の座標がタプルのndarray(配列）で返る
# landindex[0] = latitude
# landindex[1] = longitude


#for index in range(oisst_10dy_stack.shape[0]): # 1417
for index in range(3): # 1417
    
    print('index=',index)
    
    oisstweight = oisstweight_orig
    
    for counter in range(1000):# 1000回繰り返す（又は閾値を超えたらループ終了
        resmax = -10000000000
        oisst362_old[index,:,:] = oisst362[index,:,:] ## コレも必要
        
        oisstweight_old[:,:] = oisstweight[:,:] # フラグ１の場所を更新
        
        ## estocweight0,360 に更新（右と左両方追加、3/23)
        oisstweight_old[:,0] = oisstweight[:,360] # 0の前に360の値を更新
        oisstweight_old[:,361] = oisstweight[:,1] #360の先に1の値を更新
        ## 同じことをFresh362でもやる
        oisst362_old[index,:,0] = oisst362[index,:,360]
        oisst362_old[index,:,361] = oisst362[index,:,1]
        
        for i in enumerate(landindex[0]): # i = loop index
            lat = (landindex[0][i[0]])
            lon = (landindex[1][i[0]])+1 # 6/9
            ## ループの1回目は０であるはず、ESTOCWEIGHT lat,lon
            # 1箇所だけlon=360がある
            
            ## latも値を確認0、179があるとだめ
            calcflag = oisstweight_old[lat-1,lon] + oisstweight_old[lat+1,lon] + oisstweight_old[lat,lon-1] + oisstweight_old[lat,lon+1] 
            #if lon != 360:
            calcdata = (oisst362_old[index,lat-1,lon]*oisstweight_old[lat-1,lon] + oisst362_old[index,lat+1,lon]*oisstweight_old[lat+1,lon]\
                            +oisst362_old[index,lat,lon-1]*oisstweight_old[lat,lon-1] + oisst362_old[index,lat,lon+1]*oisstweight_old[lat,lon+1])
                
            if calcflag > 0:
                calcweight = calcdata/(calcflag)
                res = abs(oisst362_old[index,lat,lon]-calcweight)
                oisst362[index,lat,lon] = calcweight
                oisstweight[lat,lon] = 1
                resmax = max(resmax,res)
        

        if 0.5 > resmax: # 閾値は要確認
            print('counter=',counter)
            break

## ファイル書き出し
np.save('./work/oisst_calc_review',oisst362[:,:,1:361])
print('end')


# In[25]:


oisst362 = np.load('./work/oisst_calc_review.npy')
hoge = oisst362[0,:,:]


# In[26]:


hoge.tofile('oisst2.dat')


# In[27]:


plt.imshow(hoge)


# In[20]:


del oisst_daily_stack oisst_d10dy_stack　oisst_stack
gc.collect()
# ersst_stack.npy と oisst_10dy_stack.npyを読んでくっつけたら完成


# In[38]:


ersst = np.load('./work/ersst_calc.npy')
oisst = np.load('./work/oisst_10dy_stack.npy')

result = np.dstack([ersst,oisst])


# In[39]:


# ファイル保存

result.tofile('./work/sst1854-2020.dat')


# In[26]:


## 以下デバッグセル

# 10日平均ができている
#ersst_stack[180,360,4602]

# 週平均なので直さないとだめ
#oisst_stack[180,360,2025]

oisst_stack.shape[2]


# In[28]:


xr.open_dataset('downloads/ersst.v5.185401.nc')


# In[24]:


np.save('./work/ersst_stack.npy',ersst_stack)


# In[25]:


np.save('./work/oisst_10dy_stack.npy',oisst_10dy_stack)


# In[29]:


test02 = np.load('./work/ersst185401.npy')


# In[46]:


indexlist = np.where(np.isnan(test02))


# In[74]:


landmask


# In[47]:


plt.imshow(test02)


# In[59]:


ersst_orig.shape


# In[57]:


d1 = np.load('./work/10dymean185401.npy')
d2 = np.load('./work/10dymean185402.npy')

ersst_stack = np.append(d1,d2,axis=2)


# In[6]:


xi,yi = np.mgrid[0.5:360:1,-89.5:90:1]
forward = np.load('./work/sst185401.npy')

# 欠損値がNaNになっているので地上は－100を入れる
# copy=Falseで元の配列でNaNを置換している（デフォルトではコピーが生成される）
np.nan_to_num(forward,nan=-100,copy=False)

fmonth = interpolate.interp2d(orig_lon,orig_lat,forward,kind='linear')(xi[:,0],yi[0,:])


# In[7]:


plt.imshow(fmonth)


# In[8]:


# 翌月も読み込んで内挿に使用する
back = np.load('./work/sst185402.npy')
np.nan_to_num(back,nan=-100,copy=False)

bmonth = interpolate.interp2d(orig_lon,orig_lat,back,kind='linear')(xi[:,0],yi[0,:])


# In[9]:


plt.imshow(bmonth)


# In[30]:


#fmonthとbmonthの間を30日に増やして10日平均を3つ作る

# 内挿するポイントの数
num_interp_points = 50


# 内挿結果を格納する新しい2次元データの形状を計算する
interp_shape = (fmonth.shape[0], fmonth.shape[1], num_interp_points)

# 内挿結果を格納する新しい2次元データを作成する
interpolated_data = np.zeros(interp_shape)

# 内挿を実行する
for i in range(fmonth.shape[0]):
    for j in range(fmonth.shape[1]):
        interpolated_data[i, j, :] = np.linspace(fmonth[i, j], bmonth[i, j], num_interp_points)

# 結果の表示
#print(interpolated_data)


# In[31]:


# 10daily作成

dy1 = interpolated_data[:,:,0:10].mean(axis=2)
dy2 = interpolated_data[:,:,10:20].mean(axis=2)
dy3 = interpolated_data[:,:,20:30].mean(axis=2)

dymean = np.stack([dy1,dy2,dy3],2)


# In[29]:


plt.imshow(interpolated_data[:,:,30])


# In[9]:


# 先頭は初期化の０なので削除する
oisst_stack = oisst_stack[:,:,1:]


# In[ ]:





# In[10]:


## oisstはWeeklyデータなので１０日平均を作る
data = oisst_stack[:,:,0]

# 移動平均の窓サイズ（１０日平均なので１０）
window_size = 10

# 行方向の移動平均を計算
row_moving_averages = np.apply_along_axis(lambda m: np.convolve(m, np.ones(window_size), 'valid') / window_size, axis=1, arr=data)

# 列方向の移動平均を計算
column_moving_averages = np.apply_along_axis(lambda m: np.convolve(m, np.ones(window_size), 'valid') / window_size, axis=0, arr=data)


# In[ ]:





# In[20]:


# 以下デバッグセル


d1 = rec2
d2 = rec2

hoge = np.stack([d1,d2],2)
d3 = np.expand_dims(d1,axis=2)
hua = np.append(hoge,d3,axis=2)


# In[34]:


## OISSTファイルは180*360
## ersstファイルは89*180 > 2x2度グリッドのmonthlyデータを1X1度の10dailyに補間する
## 外挿プログラムは所在不明




# In[ ]:





# In[ ]:





# In[ ]:


### 以下デバッグ、テスト用セル


# In[3]:


with open('downloads/oisst.2020','rb') as f:

    np.fromfile(f,dtype='>i',count=1) # header 
    rec1 = np.fromfile(f,dtype='>i',count=8) # rec1
    np.fromfile(f,dtype='>i',count=1) # header
    rec2 = np.fromfile(f,dtype='>f',count=rect).reshape(180,360)
    np.fromfile(f,dtype='>i',count=1) # header
    rec3 = np.fromfile(f,dtype='>f',count=rect).reshape(180,360)# rec3
    np.fromfile(f,dtype='>i',count=1) # header
    rec4 = np.fromfile(f,dtype='>i1',count=rect).reshape(180,360)# rec4
    
    ## データの区切りにも何やら入っているので読み飛ばし
    np.fromfile(f,dtype='>i',count=4)
    
    # 2回目
    np.fromfile(f,dtype='>i',count=1) # header 
    rec5 = np.fromfile(f,dtype='>i',count=8) # rec1
    np.fromfile(f,dtype='>i',count=1) # header
    rec6 = np.fromfile(f,dtype='>f',count=rect).reshape(180,360)
    np.fromfile(f,dtype='>i',count=1) # header
    np.fromfile(f,dtype='>f',count=rect)# rec3
    np.fromfile(f,dtype='>i',count=1) # header
    np.fromfile(f,dtype='>i1',count=rect)# rec4
    
    ## データの区切りにも何やら入っているので読み飛ばし
    np.fromfile(f,dtype='>i',count=4)
    
    # 3回目
    np.fromfile(f,dtype='>i',count=1) # header 
    hoge = np.fromfile(f,dtype='>i',count=8) # rec1
    np.fromfile(f,dtype='>i',count=1) # header
    rec10 = np.fromfile(f,dtype='>f',count=rect).reshape(180,360)
    np.fromfile(f,dtype='>i',count=1) # header
    np.fromfile(f,dtype='>f',count=rect)# rec3
    np.fromfile(f,dtype='>i',count=1) # header
    np.fromfile(f,dtype='>i1',count=rect)# rec4

print(hoge)
plt.imshow(rec10)


# In[3]:


fig = figure(figsize=(20,10))
plt.imshow(land)
plt.xlabel('longitude')
plt.ylabel('latitude')


# In[23]:


ersst = xr.open_dataset("downloads/ersst.v5.185401.nc")
sst = ersst['sst']

plt.imshow(sst[0,0,:,:])


# In[24]:


ssta = ersst['ssta']
plt.imshow(ssta[0,0,:,:])


# In[22]:


ersst


# In[13]:


## データファイルはビッグエンディアンなので読み取り方注意
## 各データセクションの前に必ずヘッダが入る　1*4byte

head = np.fromfile('downloads/oisst.2019',dtype='>i',offset=0,count=1) # rec1についているヘッダ 1*4byte
rec1 = np.fromfile('downloads/oisst.2019',dtype='>f',offset=1*4,count=8)
rec2 = np.fromfile('downloads/oisst.2019',dtype='>f',offset=1*4+8*4+1*4,count=rect).reshape(180,360)
rec3 = np.fromfile('downloads/oisst.2019',dtype='>f',offset=bytes+rect,count=rect).reshape(180,360)
rec4 = np.fromfile('downloads/oisst.2019',dtype='>i',offset=bytes+rect+rect,count=rect).reshape(180,360)

plt.imshow(rec4,vmin=0,vmax=30)
print(rec1)


# In[10]:


#head = np.fromfile('downloads/oisst.2020',dtype='>i',count=1) # rec1についているヘッダ 1*4byte
rec1 = np.fromfile('downloads/oisst.2020',dtype='>i',count=8)
rec2 = np.fromfile('downloads/oisst.2020',dtype='>f',count=rect).reshape(180,360)
rec3 = np.fromfile('downloads/oisst.2020',dtype='>f',count=rect).reshape(180,360)
rec4 = np.fromfile('downloads/oisst.2020',dtype='>i',count=rect).reshape(180,360)

plt.imshow(rec4,vmin=0,vmax=30)


# In[10]:


class OisstYearlyFormat(c.Structure):
    _fields_ = [
        ("head1", c.c_int), # rec1のヘッダ
        ("rec1", c.c_int*8), # date and version number 8 4byte interger words
        ("head2", c.c_int), # rec2のヘッダ
        ("rec2", c.c_float*360*180), # 4byte real words 360*180
        ("head3", c.c_int), # rec3のヘッダ
        ("rec3", c.c_float*360*180), # 4byte real words 360*180
        ("head4", c.c_int), # rec4のヘッダ
        ("rec4", c.c_short*360*180) # 1byte integer words
    ]


# In[12]:


f = FortranFile('downloads/oisst.2020','r')
print(f.read_ints(np.int32))


# In[102]:





# In[71]:


iland = np.zeros((89,180))

for i in enumerate(indexlist[0]):
    lat = (indexlist[0][i[0]])
    lon = (indexlist[1][i[0]])
    iland[lat,lon] = 1

# iland 陸が1、海が0

# iland==1 の場所はERSSTのSSTでも0にする
# ilandで陸地だったらersstの同位置を０にする\n",
# bool indexで配列(landmask)が返る\n",
landmask = (iland >= 1)
# True の部分（地上）を０にする\n",
index = 0
while index < 4602:
    ersst_stack[index,:,:][landmask == True] = 0
    index += 1


# In[81]:


ncepmask = interpolate.interp2d(orig_lon,orig_lat,rev_iland,kind='linear')(xi[:,0],yi[0,:])


# In[85]:


test1 = np.random.rand(89,181)


# In[93]:


testmask = interpolate.interp2d(orig_lon,orig_lat,test1,kind='linear')


# In[95]:


xnew = np.linspace(0.5,360,1)
ynew = np.linspace(-89.5,90,1)
newmask = testmask(xi[:,0],yi[0,:])


# In[15]:


plt.imshow(estocweight)


# In[18]:


plt.imshow(mask)


# In[ ]:


iland = np.zeros((89,180))

for i in enumerate(indexlist[0]):
    lat = (indexlist[0][i[0]])
    lon = (indexlist[1][i[0]])
    iland[lat,lon] = 1

# iland 陸が1、海が0

# iland==1 の場所はERSSTのSSTでも-1.0e33にする
# ilandで陸地だったらersstの同位置を-1.0e33にする\n",
# bool indexで配列(landmask)が返る\n",
landmask = (iland >= 1)
# True の部分（地上）を０にする\n",
index = 0
while index < 4602:
    ersst_stack[index,:,:][landmask == True] = −１．０＊＊３３
    index += 1


# In[ ]:





# In[46]:


hoge = np.load('./work/ersst185401.npy')


# In[47]:


hoa = np.nan_to_num(hoge,nan=-1.0*10**-33,copy=False)


# In[48]:


hoa


# In[50]:


print(-1.0*10**-33)


# In[ ]:




