from main import delineate

if __name__ == '__main__':
    lon, lat = -114.955326, 31.921271  # Colorado River
    fdir_path = 'hyd_na_dir_30s.tif'
    delineate(lat, lon, fdir_path, save_tif=True, save_geojson=True)
