import math
import time
import pyradiance as pr
import frads as fr
from frads.geom import Vector


def blinds_profile_angle(altitude, azimuth, sundir, window_normal):
    """Blinds profile angle."""
    vangle = sundir.angle_from(window_normal)
    # print(f"{vangle=}")
    if vangle > math.pi/2:
        return -1
    else:
        # rotate window normal clockwise 90°
        windowx = window_normal.rotate_3d(Vector(0, 0, 1), -math.pi/2)
        windowx = windowx.scale(windowx * sundir)
        profile_angle = (sundir - windowx).angle_from(window_normal)
        # window surface azimuth
        # print(f"{math.degrees(altitude)=} {math.degrees(azimuth)=} {math.degrees(profile_angle)=}")
        # round up to nearest 10 degrees
    return profile_angle


sensor_path = 'ext_sensors.txt'
octree = 'scene.oct'
mtx_path = "./ext_sensors_r4sky.mtx"
mtx = fr.load_matrix(mtx_path)

with open(sensor_path, 'rb') as f:
    sensor = f.read()

with open("Resources/Delft-2022.wea", 'r') as f:
    meta, weas = fr.parse_wea(f.read())

ed_threshold = 16000
orientations = {'s': Vector(0, -1, 0), 'e': Vector(1, 0, 0), 'w': Vector(-1, 0, 0), 'n': Vector(0, 1, 0)}
controls = []

res = []

for wea in weas:
    if wea.time.hour >=8 and wea.time.hour <= 17:
        stime = time.perf_counter()
        _controls = {}
        _sky_mtx = fr.load_matrix(fr.genskymtx([wea], meta, mfactor=4, rotate=-22))
        sky = fr.gen_perez_sky(wea.time, meta.latitude, meta.longitude, meta.timezone, dirnorm=wea.dni, diffhor=wea.dhi, rotate=-22)
        # sky = fr.gen_perez_sky(wea.time, meta.latitude, meta.longitude, meta.timezone, dirnorm=wea.dni, diffhor=wea.dhi )
        sky_prims = pr.parse_primitive(sky.decode())
        sun_dir = Vector(*sky_prims[1].fargs[:3])
        altitude = math.asin(sun_dir.z)
        azimuth = math.atan2(sun_dir.y, sun_dir.x)
        # with open("wea.oct", 'wb') as f:
        #     f.write(pr.oconv(stdin=sky, octree=octree))
        # res = pr.rtrace(sensor, 'wea.oct', params=['-ab', '3', '-ad', '65536', '-lw', '1e-5', '-aa', '0'], header=False, irradiance=True)
        # irrad = {k: sum(float(v)*c for v,c in zip(l.split('\t'), [47.7, 119.9, 11.6])) for k, l in zip(orientations, res.decode().splitlines())}
        res2 = fr.multiply_rgb(mtx, _sky_mtx, weights=[47.4, 119.9, 11.6]).flatten()
        irrad = {k: v for k, v in zip(orientations, res2)}
        for o in orientations:
            if irrad[o] > ed_threshold:
                # check cutoff angle
                pangle = blinds_profile_angle(altitude, azimuth, sun_dir, orientations[o])
                if pangle >= 0:
                    # pangle = math.ceil(math.degrees(pangle) / 10) * 10
                    cangle = math.ceil((90 - 2 * math.degrees(pangle)) / 10) + 1
                    # cangle = math.cos(pangle)/(1-math.sqrt(1-math.cos(pangle)**2)) - pangle
                    # cangle = max(0, min(80, math.ceil(math.degrees(cangle) / 10) * 10))
                else:
                    cangle = 0
                _controls[o] = cangle
            else:
                _controls[o] = 0
        etime = time.perf_counter()
        controls.append(_controls)
        print(f"Time: {wea.time}, {irrad} {_controls} {etime-stime}")
        res.append(f"{wea.time}, {','.join(map(str, _controls.values()))}")

with open('rulebased.csv', 'w') as f:
    f.write('\n'.join(res))
