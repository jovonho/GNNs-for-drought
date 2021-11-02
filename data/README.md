# Data

The `interim` folder, with the associated processed data. All these interim datasets were produced [by the `ml_drought` project](https://github.com/esowc/ml_drought/tree/master/src/preprocess). Each subfolder in `interim` contains a specific data product:

- `reanalysis-era5-single-levels-monthly-means_preprocessed`: Data from the ERA5 climate data store. Specifically, the dataset in this folder contains the following variables:
    - `p84.162`: [Vertical integral of divergence of moisture flux](https://apps.ecmwf.int/codes/grib/param-db?id=162084): How is moisture changing at this pixel>
    - `pev`: [Potential Evaporation](https://apps.ecmwf.int/codes/grib/param-db?id=228251): How easy would it be for water to evaporate given these atmospheric conditions?
    - `sp`: [Surface Pressure](https://apps.ecmwf.int/codes/grib/param-db?id=134): What is the weight of all the air in the column of air above this pixel?
    - `t2m`: [2 metre temperature](https://confluence.ecmwf.int/display/CKB/ERA5%3A+2+metre+temperature)

- `chirps_preprocessed`: Precipitation data collected from the [Climate Hazards Group InfraRed Precipitation with Station Data](https://www.chc.ucsb.edu/data/chirps) dataset.

- `VCI_preprocessed`: VCI data collected by the [NOAA](https://www.star.nesdis.noaa.gov/smcd/emb/vci/VH/vh_validation_AUSwheat.php).

In addition, the `static` folder contains data which is static in time (as opposed to the above datasets, which are all time-varying):
- `srtm_preprocessed`: Topography data from the [shuttle radar topography mission (SRTM) digital elevation model (DEM)](https://www2.jpl.nasa.gov/srtm/)
- `esa_cci_landcover_preprocessed`: Land cover classifications by the [ESA climate change initiative](http://www.esa-landcover-cci.org/).


## Detailed description
```
chirps_preprocessed

Dimensions:  (time: 464, lon: 35, lat: 45)
Coordinates:
  * time     (time) datetime64[ns] 1981-01-31 1981-02-28 ... 2019-08-31
  * lon      (lon) float32 33.75 34.0 34.25 34.5 34.75 ... 41.5 41.75 42.0 42.25
  * lat      (lat) float32 6.0 5.75 5.5 5.25 5.0 ... -4.0 -4.25 -4.5 -4.75 -5.0
Data variables:
    precip   (time, lat, lon) float64 ...

reanalysis-era5-single-levels-monthly-means_preprocessed

Dimensions:  (time: 486, lon: 35, lat: 45)
Coordinates:
  * time     (time) datetime64[ns] 1979-01-31 1979-02-28 ... 2019-06-30
  * lon      (lon) float32 33.75 34.0 34.25 34.5 34.75 ... 41.5 41.75 42.0 42.25
  * lat      (lat) float32 6.0 5.75 5.5 5.25 5.0 ... -4.0 -4.25 -4.5 -4.75 -5.0
Data variables:
    p84.162  (time, lat, lon) float32 ...
    pev      (time, lat, lon) float32 ...
    sp       (time, lat, lon) float32 ...
    t2m      (time, lat, lon) float32 ...

VCI_preprocessed (Target)

Dimensions:  (time: 454, lon: 35, lat: 45)
Coordinates:
  * time     (time) datetime64[ns] 1981-08-31 1981-09-30 ... 2019-05-31
  * lon      (lon) float32 33.75 34.0 34.25 34.5 34.75 ... 41.5 41.75 42.0 42.25
  * lat      (lat) float32 6.0 5.75 5.5 5.25 5.0 ... -4.0 -4.25 -4.5 -4.75 -5.0
Data variables:
    VCI      (time, lat, lon) float64 ...


static\srtm_preprocessed

<xarray.Dataset>
Dimensions:     (lon: 35, lat: 45)
Coordinates:
  * lon         (lon) float32 33.75 34.0 34.25 34.5 ... 41.5 41.75 42.0 42.25
  * lat         (lat) float32 6.0 5.75 5.5 5.25 5.0 ... -4.25 -4.5 -4.75 -5.0
Data variables:
    topography  (lat, lon) float32 ...
Attributes:
    CDI:                 Climate Data Interface version 1.9.7.1 (http://mpime...
    history:             Mon Sep 30 08:36:48 2019: cdo remapbil,/Users/gabrie...
    Conventions:         CF-1.5
    GDAL_AREA_OR_POINT:  Area
    GDAL:                GDAL 2.3.3, released 2018/12/14
    CDO:                 Climate Data Operators version 1.9.7.1 (http://mpime...

```