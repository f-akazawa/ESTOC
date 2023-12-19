#!/usr/bin/env python
# coding: utf-8

## 穴埋め計算まで終わったデータにESTOCの陸＝0の場所に-1.0e33を入れる
#＃　これで提出データになる。
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

#estoclandmask は陸地True,海FalseのBool配列なので海1、陸0に直す
## estocweighも０番と
estocmask = np.where(landmask == True,0,1)
#######


# In[ ]:


fw = np.load('fresh_anaume.npy')
gh = np.load('heat_anaume.npy')
snr = np.load('snr_anaume.npy')


# In[ ]:


fw[:,estocmask == 0] = -1.0e33
gh[:,estocmask == 0] = -1.0e33
snr[:,estocmask == 0] = -1.0e33


# In[ ]:


temp = fw.byteswap()

if os.path.isfile('fwflux_big.bin'):
    os.remove('fwflux_big.bin')
    
with open('fwflux_big.bin','wb') as f:
    temp.tofile(f)


# In[ ]:


temp = gh.byteswap()

if os.path.isfile('heat_big.bin'):
    os.remove('heat_big.bin')
    
with open('heat_big.bin','wb') as f:
    temp.tofile(f)


# In[ ]:


temp = snr.byteswap()

if os.path.isfile('snr_big.bin'):
    os.remove('snr_big.bin')
    
with open('snr_big.bin','wb') as f:
    temp.tofile(f)


# In[ ]:





# In[ ]:


## 以下デバッグセル


# In[4]:
