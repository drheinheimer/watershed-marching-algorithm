import json
from pysheds.grid import Grid
import numpy as np


def shapes_to_geojson(shapes, remove_sinks=False, stringify=False):
    features = []

    for geometry, value in shapes:
        if remove_sinks:
            geometry.update(
                coordinates=[geometry['coordinates'][0]]
            )
        feature = {
            "type": "Feature",
            "geometry": geometry,
            "properties": {
                "value": value
            }
        }
        features.append(feature)

    geojson = {
        'type': 'FeatureCollection',
        'features': features
    }

    if stringify:
        geojson = json.dumps(geojson)

    return geojson


def delineate_pysheds(lat, lon, fdir_path, remove_sinks=False):
    grid = Grid.from_raster(fdir_path)
    fdir = grid.read_raster(fdir_path)

    catchment = grid.catchment(lon, lat, fdir, snap='center')
    grid.clip_to(catchment)
    catch_view = grid.view(catchment, dtype=np.uint8)
    shapes = grid.polygonize(catch_view)
    catchment_geojson = shapes_to_geojson(shapes, remove_sinks=remove_sinks, stringify=False)
    return catchment_geojson
