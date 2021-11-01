import cdsapi
import pathlib
import numpy as np


def load_cds_data():
    """
    Test to load data from the CDS API.
    You need to create an account and create a .cdsapirc file with your API key
    in your home directory.
    Instructions here: https://cds.climate.copernicus.eu/api-how-to
    """

    pathlib.Path("./data").mkdir(parents=True, exist_ok=True)

    c = cdsapi.Client()

    c.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": [
                "2m_temperature",
                "mean_total_precipitation_rate",
                "soil_type",
                "total_precipitation",
            ],
            "year": "2018",
            "month": ["01"],
            "day": ["01", "02"],
            "time": [
                "00:00",
                "01:00",
                "02:00",
                "03:00",
                "04:00",
                "05:00",
                "06:00",
                "07:00",
                "08:00",
                "09:00",
                "10:00",
                "11:00",
                "12:00",
                "13:00",
                "14:00",
                "15:00",
                "16:00",
                "17:00",
                "18:00",
                "19:00",
                "20:00",
                "21:00",
                "22:00",
                "23:00",
            ],
            "format": "grib",
        },
        "./data/Jan2018_test.grib",
    )


if __name__ == "__main__":
    load_cds_data()
