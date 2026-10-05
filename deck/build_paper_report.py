# -*- coding: utf-8 -*-
"""build_paper_report.py: 논문 작성 전 최종 보고 덱(국문, 16:9, 문법 V) 빌드, 렌더, 점검.

입력
- deck/deck_spec_paper_report.json: 쪽별 원형, 제목, 부제, 결론 줄, 보이는 글, 노트(대본, 참고 줄)
- deck/assets/paper_report/MANIFEST.json, *.png: 덱 전용 차트(경로 B, 배치 크기 300 dpi)
- deck/assets/paper_report/tables/*.json: 헤어라인 표(낱말 단위 줄 나눔 포함)
- outputs/figures/paper/v3_restructure/Fig*.png: 논문 그림 v3(600 dpi). 스펙이 패널을 지정한 쪽은 크롭
출력
- deck/render/permafrost_paper_report.pptx, .pdf, deck/render/paper_report_pages/page-XX.png
- deck/assets/paper_report/crops/*.png, S13–S15 단계 차트
실행
  cd deck && OMP_NUM_THREADS=2 nice -n 10 python3 build_paper_report.py [--no-pdf] [--audit-only]
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN  # noqa: E402

import final_lib as fl  # noqa: E402
import paper_report_lib as L  # noqa: E402
from paper_report_lib import ACC, F_M, F_S, F_X, GRAY, GRAY2, INK, run, text  # noqa: E402

ROOT = L.ROOT
DECK = L.DECK
RENDER = DECK / "render"
PAGES = RENDER / "paper_report_pages"
PPTX = RENDER / "permafrost_paper_report.pptx"
PDF = RENDER / "permafrost_paper_report.pdf"
SCRATCH = Path(os.environ.get("PR_TMP", tempfile.gettempdir()))   # LibreOffice 임시 작업 폴더

SPEC_D = json.load(open(DECK / "deck_spec_paper_report.json", encoding="utf-8"))
SPEC = {s["id"]: s for s in SPEC_D["slides"] + SPEC_D["appendix"]}
MAN = json.load(open(L.ASSET / "MANIFEST.json", encoding="utf-8"))
TAB = {sid: json.load(open(ROOT / v["file"], encoding="utf-8")) for sid, v in MAN["tables"].items()}

STATUS = {}          # 쪽별 상태(그림 대기, 크롭 사용) → 보고
CROPS_MADE = {}


def panel(name):
    p = L.PANEL_DIR / name
    assert p.exists(), p
    return p


def crop(name):
    if name not in CROPS_MADE:
        CROPS_MADE[name] = L.make_crop(name)
    return CROPS_MADE[name]


def chart(sl, sid):
    c = MAN["charts"][sid]
    g = c["geometry_in"]
    w, h = c["size_in"]
    L.place(sl, ROOT / c["file"], g["x"], g["y"], w, h)
    return g["y"] + h


def fill_row_h(t, y, support=False, bottom=6.35):
    g = t["geometry_in"]
    hh = g.get("header_h") or t["style"]["header_h"]
    n = len(t["wrapped_rows"])
    avail = bottom - (0.85 if support else 0.0) - y - hh
    return max(g["row_h"], min(1.0, round(avail / n, 3)))


SPEC_TABLES = {"S34", "AP10"}      # 2026-10-05 행을 바꾼 표는 스펙에서 다시 줄을 나눈다


def spec_table(sid, rows=None):
    vt = SPEC[sid]["visible_text"]["table"]
    return L.table_def(vt["columns"], rows if rows is not None else vt["rows"], vt["geometry_in"])


def table_slide(sl, sid, y=None, support_w=12.0):
    t = spec_table(sid) if sid in SPEC_TABLES else TAB[sid]
    y = t["geometry_in"]["y"] if y is None else y
    sup = bool(t.get("support_line"))
    rh = fill_row_h(t, y, support=sup)
    yb = L.hairline_table(sl, t, y=y, row_h=rh)
    if sup:
        s = t["support_line"]
        body = s["text"]
        if sid == "S31":                       # 영문 예문을 구 경계에서 두 줄로(한 낱말 고아 줄 방지)
            body = body.replace(" RMSE for ", " RMSE\nfor ", 1)
        L.support_line(sl, t["geometry_in"]["x"], yb + 0.22, support_w, body, lead=s.get("lead") or None)
    return yb


def gap(sl, sid, x, y, w, h, lines, what):
    L.gap_marker(sl, x, y, w, h, lines)
    STATUS.setdefault(sid, []).append("그림 대기: " + what)


# ================================================================ 쪽별 본문
def b_S01(sl, s):
    """표지: 글 열 C1–C3(5.85 in), 오른쪽 알래스카 물리 잔차 결합 ALT 지도 5.50 × 5.00 in(C4 에서 시작, 세로 가운데)."""
    vt = s["visible_text"]
    text(sl, L.ML, 1.90, 5.85, 1.40, [run(s["title"], 32, ACC, F_X)], anchor=MSO_ANCHOR.BOTTOM, line_spacing=1.05)
    text(sl, L.ML, 3.40, 5.85, 0.80, [run(s["subtitle"], 18, GRAY2, F_M)], line_spacing=1.15)
    text(sl, L.ML, 4.45, 5.85, 0.36, [run(vt["presenter"], 16, INK, F_S)])
    text(sl, L.ML, 4.85, 5.85, 0.30, [run(vt["date_event"], 14, GRAY, F_M)])
    g = s["evidence"]["geometry_in"]
    L.place(sl, panel("Alaska_ALT_map_v3_c_slide.png"), g["x"], g["y"], g["w"], g["h"])
    STATUS.setdefault("S01", []).append("알래스카 ALT 지도 슬라이드판(5.50 × 5.00 in), 로고 없음(소속 미정 기본안)")


def b_S02(sl, s):
    """연구 의의 표(12.00 폭, 보조 지도 없음). 표 묶음(표 + 약어 줄)을 본문 영역 세로 가운데에 둔다."""
    vt = s["visible_text"]
    g = vt["table"]["geometry_in"]
    t = L.table_def(vt["table"]["columns"], vt["table"]["rows"], g)
    yb = L.hairline_table(sl, t, y=g["y"], row_h=g["row_h"])
    lead, body = vt["support_line"].split("  ", 1)
    L.support_line(sl, L.ML, yb + 0.22, 12.0, body, lead=lead)


def b_S03(sl, s):
    chart(sl, "S03")


def b_S04(sl, s):
    table_slide(sl, "S04")


def b_S05(sl, s):
    L.place(sl, panel("Fig1_a_slide.png"), L.ML, L.Y0, h=5.20)
    L.place(sl, panel("Fig1_b_slide.png"), L.COL[3], L.Y0, h=5.20)
    STATUS.setdefault("S05", []).append("Fig 1a·1b 슬라이드판")


def pair(sl, heads, imgs, y_head=L.Y0, h=4.75):
    for x, t, p in zip((L.COL[0], L.COL[3]), heads, imgs):
        L.panel_head(sl, x, y_head, 5.85, t)
        L.place(sl, p, x, y_head + 0.40, h=h)


def b_S07(sl, s):
    pair(sl, s["visible_text"]["panel_heads"], [crop("f1c"), crop("f1d")])
    STATUS.setdefault("S07", []).append("Fig 1c·1d 크롭")


def b_S08(sl, s):
    """검증 사다리: 왼쪽 Fig 1e 슬라이드판 7.50 × 4.60 in, 오른쪽 C5–C6 굵은 리드 줄 3줄."""
    yc = L.Y0 + (L.H - 4.60) / 2                     # 본문 영역 세로 가운데
    L.place(sl, panel("Fig1_e_slide.png"), L.ML, yc, 7.50, 4.60)
    items = [tuple(x.split("  ", 1)) for x in s["visible_text"]["lead_lines"]]
    L.lead_lines(sl, L.COL[4], yc + 0.15, 3.80, items, gap_pt=14, line_spacing=1.08)
    STATUS.setdefault("S08", []).append("Fig 1e 슬라이드판(3지역, 5단계)과 근거 줄 3줄")


def b_S06(sl, s):
    chart(sl, "S06")
    L.hairline_table(sl, TAB["S06"])


def b_S09(sl, s):
    chart(sl, "S09")
    text(sl, L.ML, 6.10, 12.0, 0.40, [run(s["visible_text"]["symbol_key_line"], 18, INK, F_M)])


def b_S10(sl, s):
    table_slide(sl, "S10")


FIG6 = L.V3 / "Fig6.png"


def b_S11(sl, s):
    """두 지도 슬라이드판(각 4.26 × 3.91 in)과 공유 컬러바(9.95 × 0.79 in, C1 에서 막대 가운데가 두 지도 사이)."""
    heads = s["visible_text"]["panel_heads"]
    cb = panel("Fig6_cbar_slide.png")
    cw, ch = L.img_size(cb)[0] / 300.0, L.img_size(cb)[1] / 300.0
    yc = L.Y0 + L.H - ch
    pair(sl, heads, [panel("Fig6_a_slide.png"), panel("Fig6_b_slide.png")], h=yc - 0.10 - (L.Y0 + 0.40))
    L.place(sl, cb, L.COL[0], yc, cw, ch)
    STATUS.setdefault("S11", []).append("Fig 6a·6b 지도와 공유 컬러바 슬라이드판")


def b_S12(sl, s):
    L.place(sl, crop("f3b"), L.ML, L.Y0, h=5.20)
    STATUS.setdefault("S12", []).append("Fig 3b 크롭")


STAGES = {}


def b_stage(sl, s, k):
    if not STAGES:
        for i, p in enumerate(L.make_label_curve_stages(), 1):
            STAGES[i] = p
    L.place(sl, STAGES[k], L.ML, L.Y0, 12.0, 5.2)
    STATUS.setdefault(s["id"], []).append("Fig 2a·b 원천 값으로 단계 차트 새로 그림")


def b_S13(sl, s):
    b_stage(sl, s, 1)


def b_S14(sl, s):
    b_stage(sl, s, 2)


def b_S15(sl, s):
    b_stage(sl, s, 3)


def b_S16(sl, s):
    """왼쪽 Fig 4b(편향 대 보정 이득 전체), 오른쪽 Fig 4d(재보정 몫, 재보정 너머 몫). 대상 지도 자리는 제거(2026-10-05)."""
    L.place(sl, crop("f4b"), L.COL[0], L.Y0, h=5.20)
    p = crop("f4d")
    w, h = L.fit(p, 5.85, 5.20)
    L.place(sl, p, L.COL[3], L.Y0 + (L.H - h) / 2, w, h)
    STATUS.setdefault("S16", []).append("Fig 4b·4d 크롭(Fig 4 갱신판, 패널 d 재보정 몫 분리)")


def b_S17(sl, s):
    L.place(sl, crop("f3c"), L.COL[1], L.Y0, h=5.20)
    STATUS.setdefault("S17", []).append("Fig 3c 크롭(가로축 이름 붙임)")


def b_S18(sl, s):
    """Fig 4c 포레스트 하나(A2, C2–C5 가운데). 대상 지도 자리는 제거(2026-10-05)."""
    p = crop("f4c")
    w, h = L.fit(p, 7.90, 5.20)
    L.place(sl, p, L.COL[1], L.Y0 + (L.H - h) / 2, w, h)
    STATUS.setdefault("S18", []).append("Fig 4c 크롭")


def b_S19(sl, s):
    ak, ca = crop("f7a_ak"), crop("f7a_ca")
    iw_ak, ih_ak = L.img_size(ak)
    sc = 5.85 / iw_ak                                   # 두 패널 같은 배율(같은 y 척도)
    h = ih_ak * sc
    y_head = L.Y0 + (L.H - (0.40 + h + 0.10 + 0.36)) / 2
    heads = s["visible_text"]["panel_heads"]
    for x, t, p in zip((L.COL[0], L.COL[3]), heads, (ak, ca)):
        iw, ih = L.img_size(p)
        L.panel_head(sl, x, y_head, 5.85, t)
        L.place(sl, p, x, y_head + 0.40, iw * sc, ih * sc)
    text(sl, L.ML, y_head + 0.40 + h + 0.10, 12.0, 0.36, [run(s["visible_text"]["x_axis"], 18, INK, F_M)],
         align=PP_ALIGN.CENTER)
    STATUS.setdefault("S19", []).append("Fig 7a 알래스카·캐나다 패널 크롭(본문판 Δ, 절대 RMSE 짝 그림 대기)")


def b_S20(sl, s):
    chart(sl, "S20")


def b_S21(sl, s):
    w, h = L.place(sl, panel("Fig7_b_slide.png"), L.ML, L.Y0, w=5.40)
    eb = s["visible_text"]["explanation_block_v9"]["text"].replace(", ", ",\n", 1)   # 구 경계에서 두 줄로
    L.break_runs(text(sl, L.ML, L.Y0 + h + 0.12, 5.85, 0.78, [run(eb, 18, INK, F_M)], line_spacing=1.1,
                      space_after=0))
    L.place(sl, panel("Fig7_c_slide.png"), L.COL[3], L.Y0 + 0.25, h=4.80)
    STATUS.setdefault("S21", []).append("Fig 7b·7c 슬라이드판")


def b_S22(sl, s):
    pair(sl, s["visible_text"]["panel_heads"], [crop("f5a"), crop("f5c")])
    STATUS.setdefault("S22", []).append("Fig 5a·5c 크롭")


def b_S23(sl, s):
    p = crop("f5e")
    x, w = L.COL[1], 7.90
    L.place(sl, p, x, L.Y0 + 0.42, w=w)
    for frac, t in zip((0.228, 0.524, 0.845), s["visible_text"]["panel_heads"]):
        cx = x + frac * w
        text(sl, cx - 0.5, L.Y0, 1.0, 0.34, [run(t, 18, INK, F_S)], align=PP_ALIGN.CENTER, wrap=False)
    STATUS.setdefault("S23", []).append("Fig 5e 크롭(패널 머리 국문)")


def b_S24(sl, s):
    L.place(sl, panel("Fig1_a_slide.png"), L.ML, L.Y0, h=5.20)
    chart(sl, "S24")
    STATUS.setdefault("S24", []).append("지도는 Fig 1a 슬라이드판")


def b_S25(sl, s):
    chart(sl, "S25")


def b_S26(sl, s):
    chart(sl, "S26")


def b_S27(sl, s):
    cols = s["visible_text"]["columns"]
    c = MAN["charts"]["S27"]
    L.place(sl, ROOT / c["file"], L.COL[0], L.Y0, *c["size_in"])
    p2 = panel("Fig7_b_small_slide.png")
    w2, h2 = L.fit(p2, 3.80, 3.40)
    L.place(sl, p2, L.COL[2], L.Y0 + (3.40 - h2) / 2, w2, h2)
    p3 = panel("Fig5_c_slide.png")
    w3, h3 = L.fit(p3, 3.80, 3.40)
    L.place(sl, p3, L.COL[4], L.Y0 + (3.40 - h3) / 2, w3, h3)
    for x, col in zip((L.COL[0], L.COL[2], L.COL[4]), cols):
        text(sl, x, L.Y0 + 3.55, 3.80, 0.80, [run(col["claim"], 18, INK, F_S)], line_spacing=1.1, space_after=0)
        text(sl, x, L.Y0 + 4.40, 3.80, 0.25, [run(col["evidence_tag"], 12, GRAY, F_M)])
    for xv in (4.617, 8.717):
        fl.vline(sl, xv, L.Y0, 4.65, color=fl.HAIR, lw=0.75)
    STATUS.setdefault("S27", []).append("열 2 Fig 7b, 열 3 Fig 5c 슬라이드판")


def b_S28(sl, s):
    """전망: 표시 해상도 그림(12.00 폭)과 그 아래 굵은 리드 줄 3줄(한 줄씩)."""
    p = ROOT / s["evidence"]["ref"]
    w, h = L.place(sl, p, L.ML, L.Y0 - 0.08, w=12.0)
    items = [tuple(x.split("  ", 1)) for x in s["visible_text"]["lead_lines"]]
    L.lead_lines(sl, L.ML, L.Y0 - 0.08 + h + 0.10, 12.0, items, gap_pt=3, line_spacing=1.0)
    STATUS.setdefault("S28", []).append("표시 해상도 그림(28쪽과 같음)과 리드 줄 3줄")


def b_S29(sl, s):
    chart(sl, "S29")
    L.hairline_table(sl, TAB["S29"])


def b_S30(sl, s):
    table_slide(sl, "S30")


def b_S31(sl, s):
    table_slide(sl, "S31")


def b_S32(sl, s):
    chart(sl, "S32")


def b_S33(sl, s):
    table_slide(sl, "S33")


def b_S34(sl, s):
    table_slide(sl, "S34")


def b_map(sl, s):
    """8부 지도 슬라이드판 PNG: 경로만 참조(크롭·복사 없음). 배치 크기 12.00 × 5.20 in 를 넘지 않게 맞춘다."""
    p = ROOT / s["evidence"]["ref"]
    w, h = L.fit(p, 12.0, 5.2)
    x = L.ML if w > 11.9 else L.ML + (12.0 - w) / 2
    y = L.Y0 + (L.H - h) / 2                      # 낮은 PNG 는 본문 영역 세로 가운데
    L.place(sl, p, x, y, w, h)
    STATUS.setdefault(s["id"], []).append(f"지도 PNG 참조 {s['evidence']['ref'].split('/')[-1]}(다시 생성 예정)")


def result_rows(s):
    """등록 판정 칸이 채워진 행만 그린다(임시 값 금지). 의미 칸은 비어 있을 수 있다(실행 중 등)."""
    return [r["cells"] for r in s["visible_text"]["table"]["rows_all"] if r["cells"][2]]


def b_SX(sl, s):
    """결과 요약 표: 행 높이는 칸의 줄 수로 정하고(줄 0.30 in), 남는 높이는 행에 고르게 나눈다(표 아래끝 6.45 in 이하)."""
    vt = s["visible_text"]["table"]
    rows = result_rows(s)
    g = vt["geometry_in"]
    t = L.table_def(vt["columns"], rows, g)
    rhs = L.row_heights(t)
    spare = 6.45 - g["y"] - t["style"]["header_h"] - sum(rhs)
    if spare < 0:
        raise SystemExit(f"{s['id']} 표가 본문 영역을 넘음: {spare:.2f} in")
    rhs = [h + min(spare / len(rhs), 0.25) for h in rhs]
    L.hairline_table(sl, t, row_h=rhs)
    STATUS.setdefault(s["id"], []).append(f"결과 행 {len(rows)}개, 행 높이 {[round(h, 2) for h in rhs]}")


def b_SW1(sl, s):
    """워크플로 순차 적용 결과: 왼쪽 Fig 7e 슬라이드판 6.00 × 4.00 in(세로 가운데), 오른쪽 C4–C6 굵은 리드 줄 3줄."""
    g = s["evidence"]["geometry_in"]
    yc = L.Y0 + (L.H - 4.0) / 2
    L.place(sl, panel("Fig7_e_slide.png"), g["chart"]["x"], yc, 6.0, 4.0)
    items = [tuple(x.split("  ", 1)) for x in s["visible_text"]["lead_lines"]]
    L.lead_lines(sl, g["lead_lines"]["x"], yc + 0.75, g["lead_lines"]["w"], items, gap_pt=14, line_spacing=1.08)
    STATUS.setdefault("SW1", []).append("Fig 7e 슬라이드판과 근거 줄 3줄")


def b_AP1(sl, s):
    if FIG6.exists() and "f6c" in L.FIG6_CROPS:
        p = crop("f6c")
        w, h = L.fit(p, 7.90, 5.20)
        L.place(sl, p, L.COL[1], L.Y0, w, h)          # C2 에서 시작
        STATUS.setdefault("AP1", []).append("Fig 6c 크롭(기존 ALT 지도 묶음 포함)")
        return
    gap(sl, "AP1", L.ML, L.Y0, 12.0, 5.2, ["그림 대기", "Fig 6c 학습기별 전이 오차 포레스트"], "Fig 6c 학습기 포레스트")


def b_AP2(sl, s):
    table_slide(sl, "AP2")


def b_AP3(sl, s):
    chart(sl, "AP3")


def b_AP4(sl, s):
    table_slide(sl, "AP4")


def b_AP5(sl, s):
    chart(sl, "AP5")


def b_AP6(sl, s):
    table_slide(sl, "AP6")


def b_AP7(sl, s):
    table_slide(sl, "AP7")


def b_AP8(sl, s):
    chart(sl, "AP8")


def b_AP9(sl, s):
    table_slide(sl, "AP9")


def b_AP10(sl, s):
    table_slide(sl, "AP10")


# ================================================================ 참고문헌(저자 알파벳순, 번호 없음, 2단, 12 pt)
REFS = [
    "Aalto, J., Karjalainen, O., Hjort, J. & Luoto, M. Statistical forecasting of current and future circum-Arctic ground temperatures and active layer thickness. Geophys. Res. Lett. 45, 4889–4898 (2018). https://doi.org/10.1029/2018GL078007",
    "Feng, D., Beck, H., Lawson, K. & Shen, C. The suitability of differentiable, physics-informed machine learning hydrologic models for ungauged regions and climate change impact assessment. Hydrol. Earth Syst. Sci. 27, 2357–2373 (2023). https://doi.org/10.5194/hess-27-2357-2023",
    "Gautam, S. et al. Machine learning and process-based modeling of spatiotemporal changes in active layer thickness across Alaska. Sci. Rep. 15, 42420 (2025). https://doi.org/10.1038/s41598-025-26586-w",
    "Hollmann, N. et al. Accurate predictions on small data with a tabular foundation model. Nature 637, 319–326 (2025). https://doi.org/10.1038/s41586-024-08328-6",
    "Jia, X. et al. Physics-guided machine learning for scientific discovery: an application in simulating lake temperature profiles. ACM/IMS Trans. Data Sci. 2(3), 20 (2021). https://doi.org/10.1145/3447814",
    "Karjalainen, O. et al. Circumpolar permafrost maps and geohazard indices for near-future infrastructure risk assessments. Sci. Data 6, 190037 (2019). https://doi.org/10.1038/sdata.2019.37",
    "Linnenbrink, J., Milà, C., Ludwig, M. & Meyer, H. kNNDM CV: k-fold nearest-neighbour distance matching cross-validation for map accuracy estimation. Geosci. Model Dev. 17, 5897–5912 (2024). https://doi.org/10.5194/gmd-17-5897-2024",
    "Meyer, H. & Pebesma, E. Machine learning-based global maps of ecological variables and the challenge of assessing them. Nat. Commun. 13, 2208 (2022). https://doi.org/10.1038/s41467-022-29838-9",
    "Moore, M. A. et al. ABoVE: Soil Moisture and Active Layer Thickness in Alaska, USA and Canada, 2005–2024, Version 2. ORNL DAAC (2025). https://doi.org/10.3334/ORNLDAAC/2369",
    "Muñoz-Sabater, J. et al. ERA5-Land: a state-of-the-art global reanalysis dataset for land applications. Earth Syst. Sci. Data 13, 4349–4383 (2021). https://doi.org/10.5194/essd-13-4349-2021",
    "Nelson, F. E. et al. Estimating active-layer thickness over a large region: Kuparuk River basin, Alaska, U.S.A. Arct. Alp. Res. 29, 367–378 (1997). https://doi.org/10.1080/00040851.1997.12003258",
    "O'Malley, D. et al. In-context learning enables continental-scale subsurface temperature prediction from sparse local observations. arXiv:2605.16665 (2026).",
    "Pilyugina, P. et al. A physics-informed machine learning framework for permafrost stability assessment. IEEE Access 13, 96423–96433 (2025). https://doi.org/10.1109/ACCESS.2025.3573072",
    "Ploton, P. et al. Spatial validation reveals poor predictive performance of large-scale ecological mapping models. Nat. Commun. 11, 4540 (2020). https://doi.org/10.1038/s41467-020-18321-y",
    "Poggio, L. et al. SoilGrids 2.0: producing soil information for the globe with quantified spatial uncertainty. SOIL 7, 217–240 (2021). https://doi.org/10.5194/soil-7-217-2021",
    "Qu, J., Holzmüller, D., Varoquaux, G. & Le Morvan, M. TabICLv2: a better, faster, scalable, and open tabular foundation model. arXiv:2602.11139 (2026).",
    "Ran, Y. et al. New high-resolution estimates of the permafrost thermal state and hydrothermal conditions over the Northern Hemisphere. Earth Syst. Sci. Data 14, 865–884 (2022). https://doi.org/10.5194/essd-14-865-2022",
    "Ran, Y. et al. Permafrost degradation increases risk and large future costs of infrastructure on the Third Pole. Commun. Earth Environ. 3, 238 (2022). https://doi.org/10.1038/s43247-022-00568-6",
    "Read, J. S. et al. Process-guided deep learning predictions of lake water temperature. Water Resour. Res. 55, 9173–9190 (2019). https://doi.org/10.1029/2019WR024922",
    "Stefan, J. Über die Theorie der Eisbildung, insbesondere über die Eisbildung im Polarmeere. Ann. Phys. 278, 269–286 (1891).",
    "Streletskiy, D. A., CALM, GTN-P, Wieczorek, M., Heim, B. & Bartsch, A. GTN-P CALM: 35 years of Active Layer Thickness (ALT) across latitudinal and elevational gradients in the Northern Hemisphere. PANGAEA (2025). https://doi.org/10.1594/PANGAEA.972777",
    "Veremeeva, A. et al. ALLena: Thaw depth measurements of the active layer in the Lena River Delta region from 1998 to 2022, Northeastern Siberia. PANGAEA (2025). https://doi.org/10.1594/PANGAEA.973813",
    "Wadoux, A. M. J.-C., Heuvelink, G. B. M., de Bruin, S. & Brus, D. J. Spatial cross-validation is not the right way to evaluate map accuracy. Ecol. Model. 457, 109692 (2021). https://doi.org/10.1016/j.ecolmodel.2021.109692",
    "Wang, G. et al. Simulation of active layer thickness based on multi-source remote sensing data and integrated machine learning models: a case study of the Qinghai-Tibet Plateau. Remote Sens. 17, 2006 (2025). https://doi.org/10.3390/rs17122006",
    "Westermann, S. et al. ESA Permafrost Climate Change Initiative (Permafrost_cci): Permafrost active layer thickness for the Northern Hemisphere, v4.0. NERC EDS CEDA (2024). https://doi.org/10.5285/d34330ce3f604e368c06d76de1987ce5",
    "Willard, J. D. et al. Predicting water temperature dynamics of unmonitored lakes with meta-transfer learning. Water Resour. Res. 57, e2021WR029579 (2021). https://doi.org/10.1029/2021WR029579",
    "Zhang, C. et al. Combining a climate-permafrost model with fine resolution remote sensor products to quantify active-layer thickness at local scales. Environ. Res. Lett. 19, 044030 (2024). https://doi.org/10.1088/1748-9326/ad31dc",
]
REF_NOTES = ("[확인 필요] Stefan 1891 서지는 [verify]. Qu et al. 2026 의 ICML 2026 표기는 arXiv 쪽 기재만 있어 학회 표기를 "
             "넣지 않았다. Streletskiy et al. 2025 저자 목록, Moore et al. 2025 연도(DataCite 2026) 확인 필요. 서지 전문은 "
             "docs/MANUSCRIPT_DRAFT_METHODS_INTRO_2026-09-30.md 참고문헌 표와 scirep_format_and_drafts.md 4.10 에서 옮김.")
REF_PT, REF_LS, REF_SP = 12, 1.0, 3
LINE_IN = REF_PT * 1.22 * REF_LS / 72.0


def ref_lines(s, w_in=5.85 - 0.25):
    words, lines, cur = s.split(" "), 0, ""
    for wd in words:
        cand = wd if not cur else cur + " " + wd
        if L.text_w_in(cand, REF_PT) <= w_in * 0.97 or not cur:
            cur = cand
        else:
            lines += 1
            cur = wd
    return lines + (1 if cur else 0)


def ref_columns(cap=5.45):
    cols, cur, used = [], [], 0.0
    for r in REFS:
        hgt = ref_lines(r) * LINE_IN + REF_SP / 72.0
        if cur and used + hgt > cap:
            cols.append(cur)
            cur, used = [], 0.0
        cur.append(r)
        used += hgt
    cols.append(cur)
    return cols


def ref_column(sl, x, y, refs):
    from pptx.util import Inches
    tb = text(sl, x, y, 5.85, 5.5, [[run(r, REF_PT, INK, F_M)] for r in refs], line_spacing=REF_LS,
              space_after=REF_SP)
    for p in tb.text_frame.paragraphs:
        pPr = p._p.get_or_add_pPr()
        pPr.set("marL", str(int(Inches(0.25))))
        pPr.set("indent", str(-int(Inches(0.25))))


# ================================================================ 조립(스펙의 쪽 순서와 쪽 번호)
def deck_order():
    keys = []
    for x in SPEC_D["slides"]:
        if x["id"] == "S35":
            keys += ["S35a", "S35b"]
        else:
            keys.append(x["id"])
    return keys + [x["id"] for x in SPEC_D["appendix"]]


ORDER = deck_order()


def build():
    prs = fl.new_deck()
    rcols = ref_columns()
    if len(rcols) > 4:
        raise SystemExit(f"참고문헌이 2쪽(4단)을 넘음: {len(rcols)}단")
    ref_pages = [int(v) for v in str(SPEC["S35"]["page"]).replace("–", "-").split("-")]
    ref_pages = list(range(ref_pages[0], ref_pages[-1] + 1))
    for key in ORDER:
        sid = key.split("+")[0]
        sid = "S35" if sid.startswith("S35") else sid
        s = SPEC[sid]
        sl = fl.blank(prs)
        if sid == "S01":
            b_S01(sl, s)
        elif sid == "S35":
            k = 0 if key == "S35a" else 1
            L.header(sl, s["title"], None, ref_pages[k])
            for x, refs in zip((L.COL[0], L.COL[3]), rcols[2 * k: 2 * k + 2]):
                ref_column(sl, x, L.Y0, refs)
            L.notes(sl, None, s["notes"].get("reference_line"), [REF_NOTES] if k == 0 else None)
            continue
        else:
            L.header(sl, s["title"], s.get("subtitle"), s["page"])
            if sid.startswith("SM"):
                b_map(sl, s)
            elif sid.startswith("SX"):
                b_SX(sl, s)
            else:
                globals()["b_" + sid](sl, s)
            if s.get("takeaway"):
                L.takeaway(sl, s["takeaway"])
        n = s["notes"]
        extra = [f"[제작 상태] {x}" for x in STATUS.get(sid, [])]
        L.notes(sl, n.get("script"), n.get("reference_line"), extra or None)
    RENDER.mkdir(parents=True, exist_ok=True)
    prs.save(PPTX)
    print(f"[pptx] {PPTX.relative_to(ROOT)}  {len(prs.slides)} 쪽")
    return prs


def render():
    L.export_pdf(PPTX, PDF, SCRATCH / "lo")
    PAGES.mkdir(parents=True, exist_ok=True)
    for old in PAGES.glob("page-*.png"):
        old.unlink()
    subprocess.run(["nice", "-n", "10", "pdftoppm", "-r", "60", "-png", str(PDF), str(PAGES / "page")], check=True)
    print(f"[pdf] {PDF.relative_to(ROOT)}, 쪽 PNG {len(list(PAGES.glob('page-*.png')))}개")


def audit():
    rep = L.pptx_audit(str(PPTX))
    ext = dict(L.deck_extra_audit(str(PPTX)))
    hits = 0
    for i, v in rep:
        key = ORDER[i - 1]
        e = ext[i]
        msgs = []
        if v["fill_shapes"]:
            msgs.append(f"fill {v['fill_shapes']}")
        if v["off_font"]:
            msgs.append(f"font {v['off_font']}")
        if v["off_size"]:
            msgs.append(f"size {v['off_size']}")
        if v["accent_fail"]:
            msgs.append(f"accent {v['accent_outside_title']}")
        if v["big_num"]:
            msgs.append(f"bignum {v['big_num']}")
        if v["off_grid"]:
            msgs.append(f"grid {v['off_grid']}")
        if v["para_end_da"]:
            msgs.append(f"da {v['para_end_da']}")
        if v["pic_scale_bad"]:
            msgs.append(f"pic_scale {v['pic_scale_bad']}")
        if e["table_off"]:
            msgs.append(f"table_font {e['table_off']}")
        if v["accent_outside_title"] + e["table_accent"] > 1:
            msgs.append(f"accent_total {v['accent_outside_title'] + e['table_accent']}")
        if e["boxed_text"]:
            msgs.append(f"boxed_text {e['boxed_text']}")
        codes = [c for c in e["codes"] if not key.startswith("S35")]
        if codes:
            msgs.append(f"codes {sorted(set(codes))}")
        if e["dash"]:
            msgs.append(f"dash {e['dash']}")
        if e["bullets"] > 4:
            msgs.append(f"bullets {e['bullets']}")
        if e["connectors_bad"]:
            msgs.append(f"arrows {e['connectors_bad']}")
        if msgs:
            hits += 1
            print(f"  {i:2d} {key:5s} " + "; ".join(msgs))
    print(f"[audit] 지적 있는 쪽 {hits}/{len(rep)}")
    return rep, ext


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-pdf", action="store_true")
    ap.add_argument("--audit-only", action="store_true")
    a = ap.parse_args()
    if not a.audit_only:
        build()
        for k, v in STATUS.items():
            print(f"  [status] {k}: {' / '.join(v)}")
    audit()
    if not a.no_pdf and not a.audit_only:
        render()
