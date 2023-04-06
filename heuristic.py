import math
import time
import pyradiance as pr
import frads as fr
from frads.geom import Vector


def blinds_profile_angle(sundir, window_normal):
    """Blinds profile angle."""
    vangle = sundir.angle_from(window_normal)
    if vangle > math.pi/2:
        return -1
    else:
        # rotate window normal clockwise 90°
        windowx = window_normal.rotate_3d(Vector(0, 0, 1), -math.pi/2)
        windowx = windowx.scale(windowx * sundir)
        profile_angle = (sundir - windowx).angle_from(window_normal)
    return profile_angle


def main():

    mtx = fr.load_matrix("./ext_sensors_r4sky.mtx")

    with open("Resources/Delft-2022.wea", 'r') as f:
        meta, weas = fr.parse_wea(f.read())

    ed_threshold = 16000
    orientations = {'s': Vector(0, -1, 0), 'e': Vector(1, 0, 0), 'w': Vector(-1, 0, 0), 'n': Vector(0, 1, 0)}

    res = []

    for wea in weas:
        if wea.time.hour >=8 and wea.time.hour <= 17:
            stime = time.perf_counter()
            _controls = {}
            _sky_mtx = fr.load_matrix(fr.genskymtx([wea], meta, mfactor=4, rotate=-22))
            sky = fr.gen_perez_sky(wea.time, meta.latitude, meta.longitude, meta.timezone, dirnorm=wea.dni, diffhor=wea.dhi, rotate=-22)
            sky_prims = pr.parse_primitive(sky.decode())
            sun_dir = Vector(*sky_prims[1].fargs[:3])
            res2 = fr.multiply_rgb(mtx, _sky_mtx, weights=[47.4, 119.9, 11.6]).flatten()
            irrad = {k: v for k, v in zip(orientations, res2)}
            for o in orientations:
                if irrad[o] > ed_threshold:
                    # check cutoff angle
                    pangle = blinds_profile_angle(sun_dir, orientations[o])
                    if pangle >= 0:
                        cangle = math.ceil((90 - 2 * math.degrees(pangle)) / 10) + 1
                    else:
                        cangle = 0
                    _controls[o] = cangle
                else:
                    _controls[o] = 0
            etime = time.perf_counter()
            print(f"Time: {wea.time}, {irrad} {_controls} {etime-stime}")
            res.append(f"{wea.time}, {','.join(map(str, _controls.values()))}")

    with open('rb.csv', 'w') as f:
        f.write('\n'.join(res))


if __name__ == '__main__':
    main()
