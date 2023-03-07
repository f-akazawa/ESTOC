# ESTOC Data Converter

"ESTOC Data Converter" is Python3 version of NCEP forcing data creation program written in Fortran.

# Requirement

* python3.x
* scipy
* xarray
```
NETCDF files
dswrf.sfc.gauss.yyyy.nc
dlwrf.sfc.gauss.yyyy.nc
lhtfl.sfc.gauss.yyyy.nc
shtfl.sfc.gauss.yyyy.nc
uswrf.sfc.gauss.yyyy.nc
ulwrf.sfc.gauss.yyyy.nc
uflx.sfc.gauss.yyyy.nc
vflx.sfc.gauss.yyyy.nc
prate.sfc.gauss.yyyy.nc

Binary files
landft06.big
prev_climate7902nc2.big
```
# Usage

```bash
python netcdf2big.py
python dat2interpol2d.py
```

# Note

NCファイルはカレントディレクトリに置く必要があります

# Author

f-akazawa

* Fumihiko Akazawa
* JAMSTEC RIGC
* akazawa@jamstec.go.jp

# License

"ESTOC Data Converter" is under [MIT license](https://en.wikipedia.org/wiki/MIT_License).

