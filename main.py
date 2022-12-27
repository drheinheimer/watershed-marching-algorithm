import os
import json
import datetime as dt
import numpy as np
from pysheds.grid import Grid
from numba import jit

ridge_value = np.intc(5)

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
    orientation = last_dir % 2
    pivot_dir = next_dir - last_dir + 4
    if pivot_dir >= 8:
        pivot_dir -= 8
    elif pivot_dir < 0:
        pivot_dir += 8
    if orientation == 0:
        start_idx = int(last_dir / 2)
        n_corners = np.floor(pivot_dir / 2)
    else:
        start_idx = int(last_dir / 2)
        n_corners = np.ceil(pivot_dir / 2)
    end_idx = int(start_idx + n_corners)
    corners = corners_doubled[start_idx:end_idx]
    return [[row + c[0], col + c[1]] for c in corners]


@jit  # jit is critical here
def compute_flows_to_outlet(tgrid, fdir, r, c, outlet, d=0, max_depth=15000):
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
    flows_to_outlet = compute_flows_to_outlet(tgrid, fdir, new_r, new_c, outlet, d=d + 1)
    if tgrid[new_r, new_c] != ridge_value:
        tgrid[new_r, new_c] = 0 if not flows_to_outlet else 1
    return flows_to_outlet


@jit
def get_next_ridge_point(tgrid, fdir, starting_row, starting_col, outlet, starting_dir=0):
    last_flows_to_outlet = True

    # get the search space based on the starting direction
    search_space = offsets_doubled[starting_dir: starting_dir + 8 + 1]

    for i, (_r, _c) in enumerate(search_space):
        next_r, next_c = starting_row + _r, starting_col + _c

        flows_to_outlet = compute_flows_to_outlet(tgrid, fdir, next_r, next_c, outlet)

        if flows_to_outlet and not last_flows_to_outlet:

            ridge_dir = starting_dir + i
            if ridge_dir >= 8:
                ridge_dir -= 8
            return (next_r, next_c), ridge_dir
        else:
            last_flows_to_outlet = flows_to_outlet


def delineate(lat, lon, fdir_path):
    # Note: numba slows this down, so no need to jit this

    grid = Grid.from_raster(fdir_path)
    fdir = grid.read_raster(fdir_path)
    tgrid = fdir * 0 + 255
    col, row = np.floor(~fdir.affine * (lon, lat)).astype(int)

    print('Starting delineation...')
    start_time = dt.datetime.now()

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

    while True and cnt <= 150000:

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
        cnt += 1

    coords = [tgrid.affine * [c, r] for r, c in rc_coords]

    catchment_geojson = {
        'type': 'FeatureCollection',
        'features': [
            {
                'type': 'Feature',
                'properties': {},
                'geometry': {
                    'type': 'Polygon',
                    'coordinates': [coords]
                }
            }
        ]
    }

    print(f'Elapsed time: {dt.datetime.now() - start_time}')

    try:
        print('Writing tgrid to file')
        fname = 'cgrid.tif'
        if os.path.exists(fname):
            os.remove(fname)
        grid.to_raster(tgrid, fname)
    except:
        pass

    # fig, ax = plt.subplots(figsize=(12, 9))
    # fig.patch.set_alpha(0)
    # plt.imshow(catchment, cmap='Greys_r', zorder=1)
    # plt.title('Catchment', size=14)
    # plt.tight_layout()
    # plt.show()

    with open('out.json', 'w') as f:
        f.write(json.dumps(catchment_geojson, indent=2))
    #
    # print('finished!')


if __name__ == '__main__':
    lon, lat = -114.955326, 31.921271  # colorado
    # lon, lat = -75.512871, 39.703829 delaware
    fdir_path = 'hyd_na_dir_30s.tif'

    # WMA...
    delineate(lat, lon, fdir_path)
