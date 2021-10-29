# Data

The `interim` folder, with the associated processed data. All these interim datasets were produced [by the `ml_drought` project](https://github.com/esowc/ml_drought/tree/master/src/preprocess). Each subfolder in `interim` contains a specific data product:

- `reanalysis-era5-single-levels-monthly-means_preprocessed`: Data from the ERA5 climate data store. Specifically, the dataset in this folder contains the following variables:
    - 	`p84.162`: [Vertical integral of divergence of moisture flux](https://apps.ecmwf.int/codes/grib/param-db?id=162084): How is moisture changing at this pixel>
    - `pev`: [Potential Evaporation](https://apps.ecmwf.int/codes/grib/param-db?id=228251): How easy would it be for water to evaporate given these atmospheric conditions?
    - `sp`: [Surface Pressure](https://apps.ecmwf.int/codes/grib/param-db?id=134): What is the weight of all the air in the column of air above this pixel?
    - `t2m`: [2 metre temperature](https://confluence.ecmwf.int/display/CKB/ERA5%3A+2+metre+temperature)

- `chirps_preprocessed`: Precipitation data collected from the [Climate Hazards Group InfraRed Precipitation with Station Data](https://www.chc.ucsb.edu/data/chirps) dataset.

- `VCI_preprocessed`: VCI data collected by the [NOAA](https://www.star.nesdis.noaa.gov/smcd/emb/vci/VH/vh_validation_AUSwheat.php).

In addition, the `static` folder contains data which is static in time (as opposed to the above datasets, which are all time-varying):
- `srtm_preprocessed`: Topography data from the [shuttle radar topography mission (SRTM) digital elevation model (DEM)](https://www2.jpl.nasa.gov/srtm/)
- `esa_cci_landcover_preprocessed`: Land cover classifications by the [ESA climate change initiative](http://www.esa-landcover-cci.org/).
