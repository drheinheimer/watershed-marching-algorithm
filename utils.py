import json


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
