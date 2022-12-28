import json
import datetime as dt
import pandas as pd
import shapely

from main import delineate_wma
from lib.utils import delineate_pysheds

fdir_path = './hyd_na_dir_30s.tif'

with open('analysis/outlets.json') as f:
    outlets = json.load(f)

df = pd.DataFrame(columns=['method', 'watershed', 'lat', 'lon', 'length', 'area', 'time'])

i = 0
for j, outlet in enumerate(outlets['features']):
    lon, lat = outlet['geometry']['coordinates']

    for method in ['wma', 'pysheds']:

        if j == 0:
            if method == 'pysheds':
                delineate_pysheds(lat, lon, fdir_path, remove_sinks=True)
            else:
                delineate_wma(lat, lon, fdir_path)  # warmup

        start_time = dt.datetime.now()
        if method == 'pysheds':
            catchment = delineate_pysheds(lat, lon, fdir_path, remove_sinks=True)
        else:
            catchment = delineate_wma(lat, lon, fdir_path)
        elapsed_time = dt.datetime.now() - start_time

        try:
            shape = shapely.from_geojson(json.dumps(catchment))
        except:
            continue
        df.loc[i] = [method, j, lat, lon, shape.length, shape.area, elapsed_time.total_seconds()]
        print(list(df.loc[i]))
        i += 1

df.to_csv('./analysis/results.csv', index=False)
