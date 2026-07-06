"""C(고정밀 국소 데모 + 정확도-범위 트레이드오프): SOTA 조건 재현.
평탄 툰드라(P-band 물리 통함)에 PolSAR+InSAR+공변량 총동원 → 고정밀. 범위 넓히며 RMSE 상승 정량화.
 scope: 평탄툰드라(slope<2,elev<150) → 완만 → PolSAR유효전체 → (참조)전역.
 각 scope: GBM 14공변량 / GBM 18(+PolSAR+InSAR) / Diffusion 18 / 앙상블 / PolSAR단독. 공간블록 CV.
산출: data/processed/curated_scope_results.csv, figures/06_deep_learning/accuracy_vs_scope.png
"""
import numpy as np
import pandas as pd
import sys
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingRegressor
import torch, torch.nn as nn
sys.path.insert(0, "src")
from polar.plotstyle import use_polar, FROZEN, THAWED
from polar.outputs import figpath
plt = use_polar()

torch.manual_seed(0); np.random.seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
CLIP = (np.log1p(1), np.log1p(600))
BASE = ["dem_elev", "dem_slope", "dem_aspect_sin", "dem_aspect_cos", "dem_tpi", "dem_rough",
        "e5_maat", "e5_tdd", "e5_fdd", "e5_sqrt_tdd", "e5_twarm", "e5_tcold", "e5_stl1", "e5_swe"]
PHYS = ["polsar_alt", "polsar_std", "insar_sub", "insar_alt"]

p = pd.read_csv("data/processed/dl_dataset_polsar.csv")
i = pd.read_csv("data/processed/dl_dataset_insar.csv")[["insar_alt", "insar_sub", "insar_dist"]]
d = pd.concat([p, i], axis=1)
for c in PHYS:
    d[c] = d[c].fillna(d[c].median())


class CondNet(nn.Module):
    def __init__(self, dd):
        super().__init__()
        self.xemb = nn.Sequential(nn.Linear(dd, 128), nn.SiLU(), nn.Linear(128, 128))
        self.net = nn.Sequential(nn.Linear(130, 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(),
                                 nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, 1))
    def forward(self, yt, t, x):
        return self.net(torch.cat([yt.unsqueeze(-1), t.unsqueeze(-1), self.xemb(x)], -1)).squeeze(-1)


def diff_fp(Xtr, ytr, Xte, ymu, ysd, epochs=60, T=100, S=48):
    betas = torch.linspace(1e-4, 0.02, T, device=dev); acp = torch.cumprod(1 - betas, 0)
    net = CondNet(Xtr.shape[1]).to(dev); opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    Xt = torch.tensor(Xtr); yz = torch.tensor((ytr - ymu) / ysd)
    for ep in range(epochs):
        net.train(); idx = torch.randperm(len(Xt))
        for k in range(0, len(Xt), 8192):
            b = idx[k:k + 8192]; xb = Xt[b].to(dev); y0 = yz[b].to(dev)
            ti = torch.randint(0, T, (len(b),), device=dev); a = acp[ti]
            eps = torch.randn_like(y0); ytt = torch.sqrt(a) * y0 + torch.sqrt(1 - a) * eps
            opt.zero_grad(); ((net(ytt, ti.float() / T, xb) - eps) ** 2).mean().backward(); opt.step()
    net.eval(); out = []
    with torch.no_grad():
        for k in range(0, len(Xte), 8192):
            xb = torch.tensor(Xte[k:k + 8192]).to(dev)
            xe = xb.unsqueeze(0).expand(S, -1, -1).reshape(S * len(xb), -1)
            y = torch.randn(S * len(xb), device=dev)
            for ti in reversed(range(T)):
                a = acp[ti]; b_ = betas[ti]; tt = torch.full((S * len(xb),), ti / T, device=dev)
                eps = net(y, tt, xe); mean = (y - b_ / torch.sqrt(1 - a) * eps) / torch.sqrt(1 - b_)
                y = mean + (torch.sqrt(b_) * torch.randn_like(y) if ti > 0 else 0)
            out.append((y.reshape(S, len(xb)).cpu().numpy() * ysd + ymu).mean(0))
    return np.concatenate(out)


def spatial_cv(sub, feats, phys=False):
    blk = (np.floor(sub.lat / 0.5).astype(int).astype(str) + "_" + np.floor(sub.lon / 0.5).astype(int).astype(str))
    nb = blk.nunique()
    if nb < 3:
        return np.nan, np.nan, np.nan
    k = min(6, nb); alt = sub.alt_cm.values.astype(np.float32); ylog = np.log1p(alt)
    X = sub[feats].values.astype(np.float32)
    eg, ed, en = [], [], []
    for trn, te in GroupKFold(k).split(X, groups=blk):
        mu = X[trn].mean(0); sd = X[trn].std(0) + 1e-6
        Ztr, Zte = (X[trn] - mu) / sd, (X[te] - mu) / sd
        ymu, ysd = ylog[trn].mean(), ylog[trn].std() + 1e-6
        g = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=0).fit(Ztr, ylog[trn])
        pg = np.expm1(np.clip(g.predict(Zte), *CLIP)); eg.append(pg - alt[te])
        if phys:
            pdd = np.expm1(np.clip(diff_fp(Ztr, ylog[trn], Zte, ymu, ysd), *CLIP))
            ed.append(pdd - alt[te]); en.append(0.5 * pg + 0.5 * pdd - alt[te])
    R = lambda e: float(np.sqrt(np.mean(np.concatenate(e) ** 2))) if e else np.nan
    return R(eg), R(ed), R(en)


SCOPES = [
    ("평탄툰드라\n(slope<2,elev<150)", (d.dem_slope < 2) & (d.dem_elev < 150) & (d.polsar_valid == 1)),
    ("완만\n(slope<5,elev<400)", (d.dem_slope < 5) & (d.dem_elev < 400) & (d.polsar_valid == 1)),
    ("PolSAR유효\n전체", d.polsar_valid == 1),
    ("전역\n(다양지형)", pd.Series(True, index=d.index)),
]
rows = []
for name, m in SCOPES:
    sub = d[m].reset_index(drop=True)
    g14, _, _ = spatial_cv(sub, BASE, phys=False)
    g18, dif, ens = spatial_cv(sub, BASE + PHYS, phys=True)
    ps_alone = float(np.sqrt(np.mean((sub.polsar_alt.values - sub.alt_cm.values) ** 2)))
    pr = np.corrcoef(sub.polsar_alt, sub.alt_cm)[0, 1]
    rows.append(dict(scope=name.replace("\n", " "), n=len(sub), alt_sd=round(sub.alt_cm.std(), 1),
                     polsar_r=round(pr, 2), gbm14=round(g14, 2), gbm18=round(g18, 2),
                     diff18=round(dif, 2), ens18=round(ens, 2), polsar_alone=round(ps_alone, 1)))
    print(f"[{name.split(chr(10))[0]}] n={len(sub):6d} SD={sub.alt_cm.std():.0f} PolSAR_r={pr:+.2f} | "
          f"GBM14 {g14:.1f} GBM18 {g18:.1f} Diff18 {dif:.1f} ENS18 {ens:.1f}")

out = pd.DataFrame(rows)
out.to_csv("data/processed/curated_scope_results.csv", index=False)
print("\n=== 정확도-범위 트레이드오프 (공간블록 CV, cm) ===")
print(out.to_string(index=False))

# 시각화: scope별 RMSE(GBM14 vs +물리18) + ALT SD
names = [r["scope"].split(" ")[0] for r in rows]
x = np.arange(len(rows)); w = 0.34
fig, ax = plt.subplots(figsize=(11, 6))
ax.bar(x - w / 2, [r["gbm14"] for r in rows], w, color="#9aa7b4", edgecolor="#333", label="공변량 14 (기존)")
ax.bar(x + w / 2, [r["ens18"] for r in rows], w, color=FROZEN, edgecolor="#333", label="+PolSAR·InSAR 18 (앙상블)")
for xi, r in zip(x, rows):
    ax.text(xi - w / 2, r["gbm14"] + 0.2, f"{r['gbm14']:.1f}", ha="center", fontsize=8.5)
    ax.text(xi + w / 2, r["ens18"] + 0.2, f"{r['ens18']:.1f}", ha="center", fontsize=8.5, color=FROZEN)
    ax.text(xi, -1.4, f"n={r['n']//1000}k\nSD{r['alt_sd']:.0f} r{r['polsar_r']:.2f}", ha="center", fontsize=7.5)
ax.axhline(11.5, color=THAWED, ls="--", lw=1, alpha=0.7); ax.text(3.3, 11.7, "SOTA ~12cm", color=THAWED, fontsize=8, ha="right")
ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylabel("공간블록 CV RMSE (cm)")
ax.set_ylim(-2.5, max(r["gbm14"] for r in rows) + 3)
ax.set_title("정확도-범위 트레이드오프 — 좁고 평탄할수록 물리관측(PolSAR)이 통해 정확\n"
             "(SOTA 12cm는 좁은·평탄·직접관측 조건의 값. 범위 넓히면 floor 17cm로 수렴)", weight="bold")
ax.legend(); ax.grid(axis="y", alpha=0.3)
fig.tight_layout(); fig.savefig(figpath("deep_learning", "accuracy_vs_scope")); plt.close(fig)
print("saved", figpath("deep_learning", "accuracy_vs_scope"))
