"""ABoVE ReSALT (InSAR 기반 30m ALT + 계절침하 + 불확실성) 전체 다운로드 — NASA Earthdata."""
import earthaccess, os, glob
earthaccess.login(strategy="netrc")
res = earthaccess.search_data(doi="10.3334/ORNLDAAC/2004")
print(f"{len(res)} granule 다운로드 시작 → data/raw/resalt")
files = earthaccess.download(res, "data/raw/resalt")
tot = sum(os.path.getsize(f) for f in glob.glob("data/raw/resalt/*.nc4"))
print(f"완료: {len(glob.glob('data/raw/resalt/*.nc4'))}개, {tot/1e9:.2f}GB")
