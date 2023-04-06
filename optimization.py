from frads import methods, parsers
from pathlib import Path
import frads as fr
import multiprocessing as mp
import numpy as np
import pandas as pd
import pyradiance as pr
import rbfopt
import subprocess as sp


# ## Read the configuration file
config = parsers.parse_mrad_config(Path("default.cfg"))

views = [config['RaySender'][f'view{i}'].split() for i in range(1, 9)]

# ## Assemble the model and generate the matrices
# Note: the "no_multiply" option is set to True in the configuration file
with methods.assemble_model(config) as model:
    mtx_paths = methods.three_phase(model, config)

# ## Load view and daylight matrices into memory
matrices = {'vmx': {}, 'dmx': {}}
for i in range(1, 49):
    matrices['vmx'][i] = fr.load_matrix(mtx_paths.pvmx[f'gridglazing{i}'])
    matrices['dmx'][i] = fr.load_matrix(mtx_paths.dmx[f'glazing{i}'])
matrices['vmx']['door'] = fr.load_matrix(mtx_paths.pvmx['gridCCC-Doors-Glass'])
matrices['vmx']['skylight'] = fr.load_matrix(mtx_paths.pvmx['gridCCC-Skylights-Glass'])
matrices['dmx']['door'] = fr.load_matrix(mtx_paths.dmx['CCC-Doors-Glass'])
matrices['dmx']['skylight'] = fr.load_matrix(mtx_paths.dmx['CCC-Skylights-Glass'])

# ## Load blinds BSDF into memory
bsdf = [fr.load_matrix(f"Resources/glazings/0blinds.xml")]
for i in range(1, 10):
    bsdf.append(fr.load_matrix(f"Resources/BSDF/blinds{(i-1)*10:02d}.xml"))

# Room octree
room_ot = "scene.oct"

# Blinds geometry file paths
sblinds = [None, *(f"Objects/sblinds{d*10:02d}.rad" for d in range(9))]
eblinds = [None, *(f"Objects/eblinds{d*10:02d}.rad" for d in range(9))]
wblinds = [None, *(f"Objects/wblinds{d*10:02d}.rad" for d in range(9))]
nblinds = [None, *(f"Objects/nblinds{d*10:02d}.rad" for d in range(9))]

# ## Loop through the weather file and implement control
epw_path = "Resources/Delft-2022.wea"
with open(epw_path) as f:
    meta, wea = parsers.parse_wea(f.read())


def evalglare(img, ev):
    proc = sp.run([Path(pr.__file__).parent/'bin'/'evalglare', '-i', ev], input=img, stderr=sp.PIPE, stdout=sp.PIPE)
    return float(proc.stdout.decode().split(':')[1].split()[0])


# Optimization using rbfopt (pyomo + bonmin for mixed integer non-linear programming)
def main(w):
    # print(w)
    # Defining the objective function for optimization
    def objfunc(x):
        # print(x)
        window_wpis = np.zeros(294)
        south_idx = range(1, 10)
        west_idx = range(10, 25)
        north_idx = range(25, 34)
        east_idx = range(34, 49)
        for si in south_idx:
            window_wpis += fr.multiply_rgb(matrices['vmx'][si], bsdf[int(x[0])], matrices['dmx'][si], _sky_mtx, weights=[47.4, 119.9, 11.6]).flatten()
        for ei in east_idx:
            window_wpis += fr.multiply_rgb(matrices['vmx'][ei], bsdf[int(x[1])], matrices['dmx'][ei], _sky_mtx, weights=[47.4, 119.9, 11.6]).flatten()
        for wi in west_idx:
            window_wpis += fr.multiply_rgb(matrices['vmx'][wi], bsdf[int(x[2])], matrices['dmx'][wi], _sky_mtx, weights=[47.4, 119.9, 11.6]).flatten()
        for ni in north_idx:
            window_wpis += fr.multiply_rgb(matrices['vmx'][ni], bsdf[int(x[3])], matrices['dmx'][ni], _sky_mtx, weights=[47.4, 119.9, 11.6]).flatten()
        window_wpis += fr.multiply_rgb(matrices['vmx']['door'], bsdf[0], matrices['dmx']['door'], _sky_mtx, weights=[47.4, 119.9, 11.6]).flatten()
        window_wpis += fr.multiply_rgb(matrices['vmx']['skylight'], bsdf[0], matrices['dmx']['skylight'], _sky_mtx, weights=[47.4, 119.9, 11.6]).flatten()
        avg_wpi = window_wpis[:-8].mean()
        evs = window_wpis[-8:]
        blinds = [sblinds[int(x[0])], eblinds[int(x[1])], wblinds[int(x[2])], nblinds[int(x[3])]]
        blinds = [b for b in blinds if b is not None]
        tmpoct = f'temp{w.time.strftime("%m%d_%H%M")}.oct'
        with open(tmpoct, 'wb') as f:
            f.write(pr.oconv(_sky_path, *blinds, octree=room_ot))
        dgps = []
        # for v, ev in zip(views, evs):
        for i, v  in enumerate(views):
            img = pr.rpict(v, tmpoct, xres=800, yres=800, params=['-w', '-ab', '0', '-ps', '1'])
            imgname = f"{'_'.join(map(str, x))}_view{i}.hdr"
            with open(f"ctrltestimg/{imgname}.hdr", 'wb') as f:
                f.write(img)
            dgps.append(evalglare(img, str(evs[i])))
        dgp = max(dgps)
        if dgp < 0.38:
            pen = 1
        else:
            pen = 0.0001
        # with open("ctrltest.csv", "a") as wtr:
        #     wtr.write(",".join(map(str, x)))
        #     wtr.write(","+str(avg_wpi)+",")
        #     wtr.write(",".join(map(str, evs)))
        #     wtr.write(",")
        #     wtr.write(",".join(map(str, dgps))+"\n")
        return avg_wpi * pen * -1
    _sky_mtx = fr.load_matrix(fr.genskymtx([w], meta, mfactor=4, rotate=-22))
    _sky_path = f"Sources/perez/{w.time.strftime('%m%d_%H%M')}.rad"
    bb = rbfopt.RbfoptUserBlackBox(4, np.array([0] * 4), np.array([9] * 4), np.array(['I'] * 4), objfunc)
    settings = rbfopt.RbfoptSettings(max_evaluations = 10) 
    alg = rbfopt.RbfoptAlgorithm(settings, bb)
    objval, x, itercount, evalcount, fast_evalcount = alg.optimize()
    return (w, x, objval)

if __name__ == '__main__':
    import time
    wea = [w for w in wea if w.time.hour >= 8 and w.time.hour <= 17]
    pool = mp.Pool(mp.cpu_count())
    result = pool.map(main, wea)
    df = pd.DataFrame([r[1] for r in result], columns=['south', 'east', 'west', 'north'])
    df.set_index((r[0].time for r in result), inplace=True)
    df.to_csv('optctrl.csv')
    # stime = time.perf_counter()
    # main(wea[0])
    # print(time.perf_counter() - stime)
