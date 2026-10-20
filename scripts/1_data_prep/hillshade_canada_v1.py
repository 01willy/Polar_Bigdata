"""Canada (NW) 250 m DEM mosaic and hillshade for the map figures (2026-10-20), same recipe as scripts/4_visualization/paper_v3/hillshade_maps_v1.py
(Copernicus DEM GLO-30 tiles of the grid -> 250 m grid in NAD83 / Canada Atlas Lambert (EPSG:3978) by average resampling; ESRI hillshade,
azimuth 315°, altitude 45°, vertical exaggeration 2). Writes data/processed/map_canada/hillshade_v1/{dem_250m.tif, shade_250m.tif, hillshade_v1_meta.json}.
Run: nice -n 10 python3 scripts/1_data_prep/hillshade_canada_v1.py
"""
import importlib.util, json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "4_visualization" / "paper_v3")); sys.path.insert(0, str(ROOT / "src"))
spec = importlib.util.spec_from_file_location("hillshade_maps_v1", ROOT / "scripts/4_visualization/paper_v3/hillshade_maps_v1.py")
HS = importlib.util.module_from_spec(spec); sys.modules["hillshade_maps_v1"] = HS; spec.loader.exec_module(HS)
import maps_alt_v3 as MV  # noqa: E402
from rasterio.crs import CRS  # noqa: E402

PROC = ROOT / "data" / "processed"
HS.TILES["canada"] = PROC / "map_canada" / "canada_grid_dem_tiles_v1.csv"
HS.HS_DIR["canada"] = PROC / "map_canada" / "hillshade_v1"
_dst = HS.dst_crs
HS.dst_crs = lambda key, proj: CRS.from_epsg(3978) if key == "canada" else _dst(key, proj)
t0 = time.time()
proj = MV.projection("canada"); reg = MV.REG["canada"]
d = dict(key="canada", proj=proj, box=reg["box"], pad_m=reg["pad_m"])
d["ext"] = MV.box_extent(proj, d["box"], d["pad_m"])
out = HS.HS_DIR["canada"]; out.mkdir(parents=True, exist_ok=True)
info = dict(dem=HS.build_dem("canada", d, out), crs="EPSG:3978 (NAD83 / Canada Atlas Lambert)", box=reg["box"], pad_m=reg["pad_m"], extent=list(map(float, d["ext"])),
            recipe="scripts/4_visualization/paper_v3/hillshade_maps_v1.py build_dem (Resampling.average, hillshade az 315, alt 45, ve 2)", elapsed_s=round(time.time() - t0, 1))
(out / "hillshade_v1_meta.json").write_text(json.dumps(info, ensure_ascii=False, indent=1, default=str))
print(json.dumps({k: v for k, v in info.items() if k != "dem"}, default=str)); print(info["dem"])
