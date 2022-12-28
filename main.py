import os
import json
import datetime as dt

import numpy as np
from pysheds.grid import Grid
from numba import jit, njit
import shapely

ridge_value = 5

_offsets = [
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
    (-1, 0),
    (-1, 1),
]
offsets = np.array(_offsets)
offsets_doubled = np.array(_offsets * 2)

ul = (0, 0)
ur = (0, 1)
lr = (1, 1)
ll = (1, 0)
_corners = [ul, ur, lr, ll]
corners_doubled = np.array(_corners * 2)


@jit
def get_next_rc_coords(row, col, last_dir, next_dir):
    pivot_dir = next_dir - last_dir + 4
    if pivot_dir >= 8:
        pivot_dir -= 8
    elif pivot_dir < 0:
        pivot_dir += 8
    start_idx = int(last_dir / 2)
    if last_dir % 2:
        n_corners = np.ceil(pivot_dir / 2)
    else:
        n_corners = np.floor(pivot_dir / 2)
    corners = corners_doubled[start_idx:start_idx + n_corners]
    coord = np.array([row, col])
    return [coord + c for c in corners]


@jit  # jit is critical here
def check_flows_to_outlet(tgrid, fdir, r, c, outlet, d=0, max_depth=15000):
    if d > max_depth:
        raise Exception("Max depth exceeded!")
    if (r, c) == outlet:
        return True

    if tgrid[r, c] == 0:
        return False

    dir_val = fdir[r, c]

    if dir_val == 0:
        return False

    row_offset, col_offset = offsets[int(np.log2(dir_val))]
    new_r, new_c = r + row_offset, c + col_offset

    try:
        tgrid[new_r, new_c]
    except:
        return False

    # recursively check if new r, c flows to outlet
    flows_to_outlet = check_flows_to_outlet(tgrid, fdir, new_r, new_c, outlet, d=d + 1)
    if tgrid[new_r, new_c] != ridge_value:
        tgrid[new_r, new_c] = int(flows_to_outlet)
    return flows_to_outlet


@jit
def get_next_ridge_point(tgrid, fdir, starting_row, starting_col, outlet, starting_dir=0):
    last_flows_to_outlet = True

    # get the search space based on the starting direction
    search_space = offsets_doubled[starting_dir: starting_dir + 8 + 1]

    for i, (_r, _c) in enumerate(search_space):
        next_r, next_c = starting_row + _r, starting_col + _c

        flows_to_outlet = check_flows_to_outlet(tgrid, fdir, next_r, next_c, outlet)

        if flows_to_outlet and not last_flows_to_outlet:

            ridge_dir = starting_dir + i
            if ridge_dir >= 8:
                ridge_dir -= 8
            return (next_r, next_c), ridge_dir
        else:
            last_flows_to_outlet = flows_to_outlet


@jit
def delineate_coords(tgrid, fdir, row, col):
    rc_coords = []

    cnt = 0

    # The next starting dir is a critical part of the algorithm, since you need to start searching in the right
    # direction. If you search in the wrong direction, you may get stuck in a corner.
    starting_dir = 0
    first_ridge_dir = None
    last_ridge_dir = None

    outlet = (row, col)
    starting_row, starting_col = row, col
    tgrid[starting_row, starting_col] = ridge_value

    while True:  # and cnt <= 150000:

        next_ridge_point, next_ridge_dir = \
            get_next_ridge_point(tgrid, fdir, starting_row, starting_col, outlet, starting_dir=starting_dir)
        if not next_ridge_point:
            raise Exception("No next ridge point found")

        next_row, next_col = next_ridge_point
        tgrid[next_row, next_col] = ridge_value
        if cnt:
            next_rc_coords = get_next_rc_coords(starting_row, starting_col, last_ridge_dir, next_ridge_dir)
            for next_rc_coord in next_rc_coords:
                rc_coords.append(next_rc_coord)

        else:
            first_ridge_dir = next_ridge_dir

        if next_ridge_point == outlet:

            # close the polygon
            next_rc_coords = get_next_rc_coords(next_row, next_col, next_ridge_dir, first_ridge_dir)
            for next_rc_coord in next_rc_coords:
                rc_coords.append(next_rc_coord)

            # that's it, we're done!
            break

        starting_dir = next_ridge_dir + 5 if next_ridge_dir <= 2 else next_ridge_dir - 3
        starting_row, starting_col = next_row, next_col
        last_ridge_dir = next_ridge_dir
        # cnt += 1


def delineate_wma(lat, lon, fdir_path, save_tif=False, save_geojson=False):
    # Numba slows this down, so no need to jit this

    grid = Grid.from_raster(fdir_path)
    fdir = grid.read_raster(fdir_path)
    tgrid = fdir * 0 + 255
    col, row = np.floor(~fdir.affine * (lon, lat)).astype(int)
    rc_coords = delineate_coords(tgrid, fdir, row, col)
    # coords = [tgrid.affine * [c, r] for r, c in rc_coords]

    # catchment_geojson = {
    #     'type': 'FeatureCollection',
    #     'features': [
    #         {
    #             'type': 'Feature',
    #             'properties': {},
    #             'geometry': {
    #                 'type': 'Polygon',
    #                 'coordinates': [coords]
    #             }
    #         }
    #     ]
    # }

    # if save_tif:
    #     grid.clip_to(tgrid)
    #     clipped_tgrid_view = grid.view(tgrid)
    #     print('Writing tgrid to file')
    #     fname = 'diagnostics.tif'
    #     if os.path.exists(fname):
    #         os.remove(fname)
    #     grid.to_raster(clipped_tgrid_view, fname)
    #
    # if save_geojson:
    #     with open('catchment.json', 'w') as f:
    #         f.write(json.dumps(catchment_geojson, indent=2))

    # return catchment_geojson


if __name__ == '__main__':
    lon, lat = -90.01207, 29.946071
    fdir_path = './hyd_na_dir_30s.tif'

    starting_time = dt.datetime.now()
    delineate_wma(lat, lon, fdir_path)
    elapsed_time = dt.datetime.now() - starting_time
    print(f'elapsed time: {elapsed_time.total_seconds()}')
