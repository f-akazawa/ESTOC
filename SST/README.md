# ESTOC Data Converter

"ESTOC Data Converter" is Python3 version of ERSST and OISST data creation program written in Fortran.

# Requirement

* python3.x
* scipy
* xarray
```
	ERSST https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/netcdf/
OISST V2.1 https://downloads.psl.noaa.gov/Datasets/noaa.oisst.v2.highres/
```
# Usage

```bash
python make_weekly_sst.py
python post_make_OISST.py
```

# Note
実行順
1) make_weekly_sst.py
2) post_make_OISST.py

make_weekly_sstを先に動かしてERSSTの10日平均ファイルを作ります。
そこで出来たファイルを利用して次のpost_make_OISSTではOISSTの10日平均作成と
ERSSTファイルとのマージを行います。

# Author

f-akazawa

* Fumihiko Akazawa
* JAMSTEC RIGC
* akazawa@jamstec.go.jp

# License

"ESTOC Data Converter" is under [MIT license](https://en.wikipedia.org/wiki/MIT_License).

