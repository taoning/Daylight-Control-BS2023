import multiprocessing as mp
from pathlib import Path
import frads as fr
from frads import methods, parsers
import subprocess as sp
import numpy as np
import pandas as pd
import sys


# Blinds geometry file paths
sblinds = [None, *(f"Objects/sblinds{d*10:02d}.rad" for d in range(9))]
eblinds = [None, *(f"Objects/eblinds{d*10:02d}.rad" for d in range(9))]
wblinds = [None, *(f"Objects/wblinds{d*10:02d}.rad" for d in range(9))]
nblinds = [None, *(f"Objects/nblinds{d*10:02d}.rad" for d in range(9))]

skydir = Path('Sources') / 'perez'

config = parsers.parse_mrad_config(Path("default.cfg"))

views = [config['RaySender'][f'view{i}'].split() for i in range(1, 9)]

with methods.assemble_model(config) as model:
    mtx_paths = methods.three_phase(model, config)

matrices = {'vmx': {}, 'dmx': {}}
for i in range(1, 49):
    matrices['vmx'][i] = fr.load_matrix(mtx_paths.pvmx[f'gridglazing{i}'])
    matrices['dmx'][i] = fr.load_matrix(mtx_paths.dmx[f'glazing{i}'])
    matrices['vmx']['door'] = fr.load_matrix(mtx_paths.pvmx['gridCCC-Doors-Glass'])
    matrices['vmx']['skylight'] = fr.load_matrix(mtx_paths.pvmx['gridCCC-Skylights-Glass'])
    matrices['dmx']['door'] = fr.load_matrix(mtx_paths.dmx['CCC-Doors-Glass'])
    matrices['dmx']['skylight'] = fr.load_matrix(mtx_paths.dmx['CCC-Skylights-Glass'])

bsdf = [fr.load_matrix(f"Resources/glazings/0blinds.xml")]
for i in range(1, 10):
    bsdf.append(fr.load_matrix(f"Resources/BSDF/blinds{(i-1)*10:02d}.xml"))

room_ot = "scene.oct"

epw_path = "Resources/Delft-2022.wea"
with open(epw_path) as f:
    meta, weas = fr.parse_wea(f.read())

weas = [w for w in weas if w.time.hour >= 8 and w.time.hour <= 17]

def evalctrl(wea_ctrl):
    # ctrl = [0, 1, 0, 7]
    # get illums
    name = wea_ctrl[0]
    wea = wea_ctrl[1]
    dt_stamp = wea.time.strftime("%m%d_%H%M")
    ctrl = wea_ctrl[2]
    wpi = []
    sky_mtx = fr.load_matrix(fr.genskymtx([wea], meta, mfactor=4, rotate=-22))
    sky_path = f"Sources/perez/{dt_stamp}.rad"
    window_wpis = np.zeros(294)
    south_idx = range(1, 10)
    east_idx = range(34, 49)
    west_idx = range(10, 25)
    north_idx = range(25, 34)
    for si in south_idx:
        window_wpis += fr.multiply_rgb(
            matrices['vmx'][si], 
            bsdf[int(ctrl[0])], 
            matrices['dmx'][si], 
            sky_mtx, 
            weights=[47.4, 119.9, 11.6]).flatten()
    for ei in east_idx:
        window_wpis += fr.multiply_rgb(
            matrices['vmx'][ei], 
            bsdf[int(ctrl[1])], 
            matrices['dmx'][ei], 
            sky_mtx, 
            weights=[47.4, 119.9, 11.6]).flatten()
    for wi in west_idx:
        window_wpis += fr.multiply_rgb(
            matrices['vmx'][wi], 
            bsdf[int(ctrl[2])], 
            matrices['dmx'][wi], 
            sky_mtx, 
            weights=[47.4, 119.9, 11.6]).flatten()
    for ni in north_idx:
        window_wpis += fr.multiply_rgb(
            matrices['vmx'][ni], 
            bsdf[int(ctrl[3])], 
            matrices['dmx'][ni], 
            sky_mtx, weights=[47.4, 119.9, 11.6]).flatten()
    window_wpis += fr.multiply_rgb(
        matrices['vmx']['door'], 
        bsdf[0], 
        matrices['dmx']['door'], 
        sky_mtx, 
        weights=[47.4, 119.9, 11.6]).flatten()
    window_wpis += fr.multiply_rgb(
        matrices['vmx']['skylight'], 
        bsdf[0], 
        matrices['dmx']['skylight'], 
        sky_mtx, 
        weights=[47.4, 119.9, 11.6]).flatten()
    wpi.append(window_wpis[:-8].mean())
    # render
    blinds = [sblinds[int(ctrl[0])], eblinds[int(ctrl[1])], wblinds[int(ctrl[2])], nblinds[int(ctrl[3])]]
    blinds = [b for b in blinds if b is not None]
    # Write a .rif file and run rad
    rif = ["oconv= -i scene.oct"]
    rif.append(f"scene= {str(sky_path)} {' '.join(blinds)}")
    rif.append("ZONE=i 3.1 17 -24.48 -1.57 0 6.92")
    rif.append("QUALITY=M")
    rif.append("VARIABILITY=H")
    rif.append("DETAIL=H")
    rif.append("RESOLUTION=800")
    rif.append(f"PICTURE={name}hdr/{name}_{dt_stamp}")
    for vi, _ in enumerate(views):
        rif.append(f"view=view{vi+1} -vf Views/view{vi+1}.vf")
    # rif.append("INDIRECT=5")
    # rif.append(f"AMBFILE={name}_{dt_stamp}"
    # print("\n".join(rif))
    rif_path = Path(f"{name}hdr/{name}_{dt_stamp}.rif")
    with open(rif_path, "w") as f:
        f.write("\n".join(rif))
    sp.run(['rad', str(rif_path)])
    # Run evalglare
    dgps = []
    for vi, _ in enumerate(views):
        img = f"{name}hdr/{name}_{dt_stamp}_view{vi+1}.hdr"
        proc = sp.check_output(['evalglare', '-i', str(window_wpis[-8+vi]), img])
        dgps.append(proc.decode().split(':')[1].split()[0])
    return wpi, window_wpis[-8:], dgps


if __name__ == "__main__":
    ctrl_path = Path(sys.argv[1])
    name = ctrl_path.stem
    ctrldf = pd.read_csv(ctrl_path, index_col=0, parse_dates=True)
    wea_ctrls = []
    for idx, wea in enumerate(weas):
        ctrl = ctrldf.iloc[idx, :].tolist()
        wea_ctrls.append([name, wea, ctrl])
    pool = mp.Pool(mp.cpu_count())
    result = pool.map(evalctrl, wea_ctrls)
    df = pd.DataFrame((r[0] for r in result), columns=['avg_wpi'])
    df[[f'ev{i}' for i in range(1, 9)]] = [r[1] for r in result]
    df[[f'dgp{i}' for i in range(1, 9)]] = [r[2] for r in result]
    df.set_index((wea.time for wea in weas), inplace=True)
    df.to_csv(f"{name}_illum_dgp.csv")
