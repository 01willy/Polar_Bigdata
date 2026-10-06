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
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from PIL import Image  # noqa: E402
from pptx.util import Inches  # noqa: E402
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


SPEC_TABLES = {"S04", "S34", "AP10"}      # 2026-10-05 행을 바꾼 표는 스펙에서 다시 줄을 나눈다


def spec_table(sid, rows=None):
    vt = SPEC[sid]["visible_text"]["table"]
    return L.table_def(vt["columns"], rows if rows is not None else vt["rows"], vt["geometry_in"], emph=vt.get("emph"))


def table_from(vt, row_h=None):
    """스펙 표 정의(columns, rows, geometry_in, emph)로 표를 그린다. 아래끝 y 를 돌려준다."""
    g = vt["geometry_in"]
    t = L.table_def(vt["columns"], vt["rows"], g, emph=vt.get("emph"))
    rh = row_h if row_h is not None else max(g["row_h"], *L.row_heights(t))
    return t, rh


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
    """표지(조정 지시 2차 1, 2, 11, 12): 바탕 그림이 있으면 전면(13.333 × 7.5), 없으면 오른쪽 알래스카 지도 슬라이드판.
    제목 32 pt 두 줄, 부제 18 pt, 발표자 16 pt, 소속 14 pt 는 왼쪽(폭 7.3 in), 로고 3개는 왼쪽 아래(높이 0.45 in)."""
    vt = s["visible_text"]
    bg = ROOT / "deck/assets/paper_report/cover_bg_v3.png"
    if bg.exists():
        pic = sl.shapes.add_picture(str(bg), 0, 0, Inches(13.333), Inches(7.5))
        pic.shadow.inherit = False
        STATUS.setdefault("S01", []).append("표지 바탕 그림 cover_bg_v1.png(전면)")
    else:
        L.place(sl, panel("Alaska_ALT_map_v3_c_slide.png"), 7.3, 1.25, 5.5, 5.0)
        STATUS.setdefault("S01", []).append("바탕 그림 없음, 알래스카 ALT 지도 슬라이드판(오른쪽)")
    # 제목 두 줄(사용자 요청). 32 pt 두 줄은 8.8 in 이라 흰 영역(5.7 in)을 넘으므로 28 pt, 구 경계에서 직접 나눔(보고 항목)
    title = s["title"].replace(" 위한 ", " 위한\n", 1)
    tw = 7.9 if bg.exists() else 6.3
    L.break_runs(text(sl, L.ML, 0.95, tw, 1.30, [run(title, 28, ACC, F_X)], anchor=MSO_ANCHOR.TOP, line_spacing=1.1))
    text(sl, L.ML, 2.45, 5.0, 0.40, [run(s["subtitle"], 18, GRAY2, F_M)])
    text(sl, L.ML, 3.30, 5.0, 0.36, [run(vt["presenter"], 16, INK, F_S)])
    text(sl, L.ML, 3.70, 5.0, 0.30, [run(vt["affiliation"], 14, GRAY2, F_M)])
    L.logos_row(sl, x_left=L.ML, y_bottom=6.90, h=L.LOGO_H_COVER, gap=L.LOGO_GAP_COVER)


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
    p = method_fig("S04")
    if p is not None:
        fig_lines(sl, s, p, s["visible_text"]["lead_lines"], "결과 요약 그림")
        return
    table_slide(sl, "S04")
    STATUS.setdefault("S04", []).append("R1 없음, 요지 표")


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
    w, h = 7.0, 7.0 * 4.6 / 7.5                        # 7.00 × 4.29(글자 약 12 pt), 오른쪽 열을 4.7 in 로 넓힘
    yc = L.Y0 + (L.H - h) / 2
    L.place(sl, panel("Fig1_e_slide.png"), L.ML, yc, w, h)
    items = [tuple(x.split("  ", 1)) for x in s["visible_text"]["lead_lines"]]
    L.lead_lines(sl, L.ML + w + 0.30, yc + 0.10, 12.0 - w - 0.30, items, gap_pt=14, line_spacing=1.08)
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
    """원고 구성과 분량: 왼쪽 절별 단어 수 차트(7.90 × 4.75), 오른쪽 보조 표(스펙 support_table, 2026-10-05 원고 상태)."""
    chart(sl, "S29")
    st = s["visible_text"]["support_table"]
    t = L.table_def(st["columns"], st["rows"], st["geometry_in"], align=["l", "l"])
    L.hairline_table(sl, t, y=st["geometry_in"]["y"], row_h=st["geometry_in"]["row_h"])


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
    """지도 슬라이드판 PNG: 경로만 참조(크롭·복사 없음). 배치 크기 12.00 × 5.20 in 를 넘지 않게 맞춘다.
    알래스카 지도 쪽은 음영 지도(Alaska_ALT_hillshade_slide.png)가 있으면 그것을 쓴다."""
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


# ================================================================ 발표 구조 개편(v2.0) 새 쪽
def method_fig(sid):
    p = ROOT / SPEC[sid]["evidence"]["ref"]
    return p if p.exists() else None


def b_method(sl, s):
    """방법 도식(12.0 × 5.2 in 이하, 300 dpi): 배치 크기 그대로, 본문 영역 가운데. 근거 줄은 두지 않는다(그림이 본문 영역을 채움)."""
    p = ROOT / s["evidence"]["ref"]
    with Image.open(p) as im:
        dpi = (im.info.get("dpi") or (300, 300))[0]
        w, h = im.size[0] / dpi, im.size[1] / dpi
    if w > 12.0 or h > 5.2:
        w, h = L.fit(p, 12.0, 5.2)
    L.place(sl, p, L.ML + (12.0 - w) / 2, L.Y0 + (5.2 - h) / 2, w, h)
    STATUS.setdefault(s["id"], []).append(f"방법 도식 {p.name}({w:.2f} × {h:.2f} in)")


b_N03 = b_N05 = b_N06 = b_N09 = b_N10 = b_N12 = b_method


def b_N08(sl, s):
    alt = s["evidence"].get("ref_alt")
    if alt and (ROOT / alt).exists():
        s = dict(s, evidence=dict(s["evidence"], ref=alt))
    b_method(sl, s)


def fig_lines(sl, s, p, lines, label):
    """그림 + 굵은 리드 줄 3줄 이하. 그림 폭이 7.9 in 이하이면 왼쪽에 두고 줄은 오른쪽 열(C4–C6 또는 C5–C6),
    그보다 넓으면 12.0 in 폭으로 줄이고 아래 높이가 남을 때만 줄을 둔다(없으면 노트에만)."""
    with Image.open(p) as im:
        dpi = (im.info.get("dpi") or (300, 300))[0]
        w, h = im.size[0] / dpi, im.size[1] / dpi
    items = [tuple(x.split("  ", 1)) for x in lines]
    if w <= 7.95:
        w, h = L.fit(p, min(w, 7.9), 5.2)
        L.place(sl, p, L.ML, L.Y0 + (L.H - h) / 2, w, h)
        xl = L.COL[3] if w <= 5.9 else L.COL[4]
        wl = L.COLR[5] - xl
        L.lead_lines(sl, xl, L.Y0 + 0.9 if w <= 5.9 else L.Y0 + 0.3, wl, items, gap_pt=14, line_spacing=1.08)
        STATUS.setdefault(s["id"], []).append(f"{label} {p.name}({w:.2f} × {h:.2f} in), 리드 줄 {len(items)}줄 오른쪽")
        return
    w, h = L.fit(p, 12.0, 5.2)
    n_lines = len(items)
    need = 0.25 + 0.36 * n_lines                   # 리드 줄 한 줄 0.36 in(18 pt, 줄 간격 1.0)
    if n_lines and h > 5.2 - need and s.get("shrink_for_lines", True):
        w, h = L.fit(p, 12.0, 5.2 - need)           # 전폭 그림을 줄여 리드 줄 자리를 만든다(글자 12 pt 이상 유지 여부는 보고)
    if n_lines and h <= 5.2 - need:
        L.place(sl, p, L.ML + (12.0 - w) / 2, L.Y0, w, h)
        L.lead_lines(sl, L.ML, L.Y0 + h + 0.22, 12.0, items, gap_pt=3, line_spacing=1.0)
        STATUS.setdefault(s["id"], []).append(f"{label} {p.name}(전폭, {w:.2f} × {h:.2f}), 리드 줄 {n_lines}줄 아래")
    else:
        L.place(sl, p, L.ML + (12.0 - w) / 2, L.Y0 + (5.2 - h) / 2, w, h)
        STATUS.setdefault(s["id"], []).append(f"{label} {p.name}(전폭, 높이 {h:.2f}), 리드 줄은 노트에만")


def b_N01(sl, s):
    p = method_fig("N01")
    if p is None:
        p = panel("Fig1_a_slide.png")
        STATUS.setdefault("N01", []).append("B1 없음, Fig 1a 슬라이드판")
        fig_lines(sl, s, p, s["visible_text"]["lead_lines"], "배경 그림")
        return
    if "_v1_" in p.name:                              # 이전 B1(세 열, 제품 불일치 포함): 전폭, 리드 줄은 노트에만
        fig_lines(sl, s, p, s["visible_text"]["lead_lines"], "배경 그림")
        return
    # B1 은 12.0 × 5.2 전폭 그림이고 오른쪽 아래(x 7.75–11.9, 아래에서 0.2–2.1 in)를 비워 두었다. 리드 줄은 그 자리에 둔다
    w, h = L.fit(p, 12.0, 5.2)
    L.place(sl, p, L.ML, L.Y0, w, h)
    items = [tuple(x.split("  ", 1)) for x in s["visible_text"]["lead_lines"]]
    L.lead_lines(sl, L.ML + 7.75, L.Y0 + h - 2.05, 4.15, items, gap_pt=4, line_spacing=1.0)
    STATUS.setdefault("N01", []).append(f"B1 전폭({w:.2f} × {h:.2f}) + 리드 줄 {len(items)}줄 오른쪽 아래 빈 자리")


def b_NB2(sl, s):
    fig_lines(sl, s, ROOT / s["evidence"]["ref"], s["visible_text"]["lead_lines"], "배경 그림")


def b_NR0(sl, s):
    fig_lines(sl, s, ROOT / s["evidence"]["ref"], s["visible_text"]["lead_lines"], "결과 요약 그림")


def fallback_table(sl, s):
    vt = s["fallback_table"]
    t, rh = table_from(vt)
    L.hairline_table(sl, t, y=vt["geometry_in"]["y"], row_h=rh)
    STATUS.setdefault(s["id"], []).append("그림 없음, 대체 표")


def b_NR2(sl, s):
    p = method_fig("NR2")
    if p is None:
        fb = ROOT / s["evidence"].get("fallback", "")
        if fb.exists():
            fig_lines(sl, s, fb, s["visible_text"]["lead_lines"], "결과 요약 그림(R0 대체)")
            return
        fallback_table(sl, s)
        return
    fig_lines(sl, s, p, s["visible_text"]["lead_lines"], "결과 요약 그림")


def b_NS1(sl, s):
    p = method_fig("NS1")
    if p is None:
        fallback_table(sl, s)
        return
    fig_lines(sl, s, p, s["visible_text"]["lead_lines"], "강점 그림")


def b_NTM(sl, s):
    """방법별 전이 지도: 그림 + 설명문(.md)의 RMSE 한 줄(있을 때만)."""
    p = ROOT / s["evidence"]["ref"]
    leg = ROOT / s["evidence"].get("legend", "")
    lines = []
    if leg.exists():
        txt = leg.read_text(encoding="utf-8")
        m = re.findall(r"RMSE[^.\n]*", txt)
        if m:
            lines = ["RMSE  " + m[0].strip()[:90]]
    if lines:
        fig_lines(sl, s, p, lines, "전이 지도")
    else:
        b_map(sl, s)


b_NOP = b_NTM
b_SM1H = b_map
b_NTM = b_NOP = b_map          # 설명문 파일 없음: 수치는 스펙 결론 줄·노트에 둔다(2026-10-05 조정 지시)


def b_NE1(sl, s):
    lines = list(s["visible_text"]["lead_lines"])[:3]
    if "[N]" in lines[-1]:
        vals = e1_values()
        if vals:
            lines[-1] = lines[-1].replace("사전 등록 가설 [N]", f"사전 등록 가설 {vals[0]}").replace("적합 최소 [N]건", f"적합 최소 {vals[1]}건")
        else:
            lines[-1] = lines[-1].split(", 사전 등록 가설")[0]      # 원천값이 없으면 가설·적합 수는 적지 않는다
            STATUS.setdefault("NE1", []).append("E1 원천값 없음, 가설·적합 수 생략")
    fig_lines(sl, s, ROOT / s["evidence"]["ref"], lines, "근거 범위 그림")


def e1_values():
    """E1 원천값(사전 등록 가설 수, 적합 건수). method/E1_*values*.json 또는 .csv 에서 찾는다."""
    for p in sorted((ROOT / "deck/assets/paper_report/method").glob("E1*value*")):
        try:
            if p.suffix == ".json":
                v = json.loads(p.read_text(encoding="utf-8"))
                flat = {}
                def walk(x, k=""):
                    if isinstance(x, dict):
                        for kk, vv in x.items():
                            walk(vv, kk)
                    elif isinstance(x, (int, float)):
                        flat[k] = x
                walk(v)
                hyp = next((flat[k] for k in flat if "hyp" in k.lower()), None)
                fit = next((flat[k] for k in flat if "fit" in k.lower()), None)
                if hyp and fit:
                    return int(hyp), int(fit)
        except Exception:
            continue
    return None


def b_NGL(sl, s):
    vt = s["visible_text"]["table"]
    g = vt["geometry_in"]
    t = L.table_def(vt["columns"], vt["rows"], g, emph=vt.get("emph"))
    if vt.get("compact"):                      # 용어 표(9행): 줄 수에 맞춘 낮은 행
        rh = [max(g["row_h"], 0.3 * max(len(c) for c in r) + 0.14) for r in t["wrapped_rows"]]
    else:
        rh = max(g["row_h"], *L.row_heights(t))
    L.hairline_table(sl, t, y=g["y"], row_h=rh, header_h=g.get("header_h"))


b_NCL = b_NLM = b_NGL
b_NST = b_method


def b_fitfull(sl, s):
    """결과 그림 한 장을 본문 영역(12.0 × 5.2 in)에 비율 유지로 최대한 크게(논문 판 그림 확대 포함)."""
    p = ROOT / s["evidence"]["ref"]
    w, h = L.fit(p, 12.0, 5.2)
    L.place(sl, p, L.ML + (12.0 - w) / 2, L.Y0 + (5.2 - h) / 2, w, h)
    STATUS.setdefault(s["id"], []).append(f"그림 {p.name}({w:.2f} × {h:.2f} in)")


b_NA3 = b_NUM = b_NG1 = b_NML = b_NP1 = b_NC1 = b_fitfull


def b_NLS(sl, s):
    """라벨 수에 따른 지도 변화: PPTX 에는 GIF(발표용, PowerPoint 가 재생), PDF 용 빌드에는 정지 5패널 PNG."""
    p = ROOT / s["evidence"]["ref"]
    gif = ROOT / s["evidence"].get("gif", "")
    use_gif = gif.exists() and not STATIC_MEDIA
    src = gif if use_gif else p
    w, h = L.fit(p, 12.0, 5.2)
    pic = sl.shapes.add_picture(str(src), Inches(L.ML + (12.0 - w) / 2), Inches(L.Y0 + (5.2 - h) / 2), Inches(w), Inches(h))
    pic.shadow.inherit = False
    STATUS.setdefault(s["id"], []).append(f"{'GIF' if use_gif else '정지 PNG'} {src.name}({w:.2f} × {h:.2f} in)")


b_NLSA = b_NLS
b_NL3 = b_SMH = b_SM1A = b_map
b_NM6B = b_method


def b_N02(sl, s):
    """기존 연구의 한계와 이 연구의 개선: M2_gap(12.0 × 3.4) 이 있으면 그림, 없으면 옛 2쪽 표(12.00 폭). 아래에 약어 줄."""
    p = method_fig("N02")
    sup = s["visible_text"].get("support_line", "")
    lead, body = sup.split("  ", 1) if "  " in sup else ("", sup)
    if p is not None:
        # M2_gap 은 12.0 × 5.2 in 전폭 그림이다. 약어 줄은 사양에 있을 때만, 결론 줄 위에 자리가 남을 때만 둔다
        w, h = L.place(sl, p, L.ML, L.Y0, h=5.20)
        if body and h <= 4.6:
            L.support_line(sl, L.ML, L.Y0 + h + 0.30, 12.0, body, lead=lead)
        STATUS.setdefault("N02", []).append(f"방법 도식 {p.name}({w:.2f} × {h:.2f} in)")
        return
    vt = SPEC["S02"]["visible_text"]
    g = vt["table"]["geometry_in"]
    t = L.table_def(vt["table"]["columns"], vt["table"]["rows"], g)
    yb = L.hairline_table(sl, t, y=g["y"], row_h=g["row_h"])
    L.support_line(sl, L.ML, yb + 0.22, 12.0, body, lead=lead)
    STATUS.setdefault("N02", []).append("M2_gap 없음, 옛 2쪽 표로 대신")


def b_N04(sl, s):
    t, rh = table_from(s["visible_text"]["table"])
    L.hairline_table(sl, t, row_h=rh)


def b_N07(sl, s):
    """Stefan 물리 모델과 계수 재보정: 개념 좌표도(12.00 × 4.15) + 재보정 수식 줄 + 기호 열쇠 줄."""
    p = ROOT / s["evidence"]["ref"]
    figs = ROOT / "deck" / "mk_paper_report_figs.py"
    if not p.exists() or p.stat().st_mtime < figs.stat().st_mtime:      # 그림 함수(색·글자)가 바뀌면 다시 그림
        subprocess.run([sys.executable, str(DECK / "mk_paper_report_n07.py")], check=True, capture_output=True,
                       env=dict(os.environ, OMP_NUM_THREADS="2"))
    w, h = L.place(sl, p, L.ML, L.Y0, 12.0, 4.15)
    y = L.Y0 + h + 0.12
    L.equation(sl, L.ML, y, 12.0, [("재보정 계수  ", "lead"), ("E", ""), ("n", "sub"), (" = (n E", ""), ("ls", "sub"),
                                   (" + κ E", ""), ("0", "sub"), (") / (n + κ),  κ = 10", "")])
    L.equation(sl, L.ML, y + 0.46, 12.0, [("E Stefan 계수, TDD 융해 도일, n 대상 라벨 수, E", ""), ("ls", "sub"),
                                          (" 대상 라벨 최소제곱 계수, E", ""), ("0", "sub"), (" 원천 계수", "")], size=16)
    STATUS.setdefault("N07", []).append("개념 좌표도 12.00 × 4.15(옛 9쪽 그림의 낮은 판)와 수식 줄")


def two_tables(sl, s):
    for vt in s["visible_text"]["tables"]:
        t, rh = table_from(vt)
        L.hairline_table(sl, t, x=vt["geometry_in"]["x"], y=vt["geometry_in"]["y"], row_h=rh)


b_N30 = b_N31 = two_tables


def b_NTY(sl, s, label):
    text(sl, L.ML, 2.75, 12.0, 0.70, [run(s["title"], 26, ACC, F_X)], align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    text(sl, L.ML, 3.55, 12.0, 0.50, [run(s["subtitle"], 18, GRAY, F_M)], align=PP_ALIGN.CENTER)
    L.logos_topright(sl)
    L.pagenum_bottomright(sl, label)


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


# ================================================================ 조립(스펙의 쪽 순서, 쪽 번호는 그린 순서대로)
SKIPPED = []
STATIC_MEDIA = False         # True 이면 GIF 대신 정지 PNG(PDF 용 빌드)


def ready(x):
    if x.get("render") is False:
        return False
    req = x.get("requires")
    if req and not (ROOT / req).exists():
        SKIPPED.append((x["id"], x["title"], req))
        return False
    return True


def deck_order():
    keys = []
    for x in SPEC_D["slides"]:
        if not ready(x):
            continue
        keys += ["S35a", "S35b"] if x["id"] == "S35" else [x["id"]]
    return keys + [x["id"] for x in SPEC_D["appendix"] if ready(x)]


ORDER = deck_order()
MAIN_IDS = {x["id"] for x in SPEC_D["slides"]}


def build(static_media=False, out=None):
    global STATIC_MEDIA
    STATIC_MEDIA = static_media
    out = out or PPTX
    prs = fl.new_deck()
    rcols = ref_columns()
    if len(rcols) > 4:
        raise SystemExit(f"참고문헌이 2쪽(4단)을 넘음: {len(rcols)}단")
    n_main, n_app = 0, 0
    labels = {}
    for key in ORDER:
        sid = key.split("+")[0]
        sid = "S35" if sid.startswith("S35") else sid
        s = SPEC[sid]
        sl = fl.blank(prs)
        if sid == "S01":
            label = None
        elif sid in MAIN_IDS:
            n_main += 1
            label = n_main
        else:
            n_app += 1
            label = f"부록 {n_app}"
        labels[key] = label
        if sid == "S01":
            b_S01(sl, s)
        elif sid == "NTY":
            b_NTY(sl, s, label)
        elif sid == "S35":
            k = 0 if key == "S35a" else 1
            L.header(sl, s["title"], None, label)
            for x, refs in zip((L.COL[0], L.COL[3]), rcols[2 * k: 2 * k + 2]):
                ref_column(sl, x, L.Y0, refs)
            L.notes(sl, None, s["notes"].get("reference_line"), [REF_NOTES] if k == 0 else None)
            continue
        else:
            L.header(sl, s["title"], s.get("subtitle"), label)
            if sid.startswith("SM"):
                b_map(sl, s)
            elif sid.startswith("SX"):
                b_SX(sl, s)
            else:
                globals()["b_" + sid](sl, s)
            if s.get("takeaway"):
                L.takeaway(sl, s["takeaway"])
        n = s["notes"]
        extra = [] if sid in MAIN_IDS else [f"[제작 상태] {x}" for x in STATUS.get(sid, [])]   # 본편 노트에는 파일 이름 없음
        L.notes(sl, n.get("script"), n.get("reference_line"), extra or None)
    RENDER.mkdir(parents=True, exist_ok=True)
    prs.save(out)
    print(f"[pptx] {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}  {len(prs.slides)} 쪽(본편 {n_main + 1}, 부록 {n_app})")
    for sid, title, req in SKIPPED:
        print(f"  [대기] {sid} {title}: {req} 없음")
    (RENDER / "paper_report_page_map.json").write_text(json.dumps(labels, ensure_ascii=False, indent=1), encoding="utf-8")
    return prs


def render(src=None):
    L.export_pdf(src or PPTX, PDF, SCRATCH / "lo")
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
        has_gif = any((ROOT / SPEC[k]["evidence"].get("gif", "")).exists() for k in ORDER if k in SPEC and SPEC[k]["evidence"].get("gif"))
        if has_gif:
            STATUS.clear()
            tmp = SCRATCH / "permafrost_paper_report_static.pptx"
            build(static_media=True, out=tmp)          # PDF 는 정지 5패널
            render(tmp)
        else:
            render()
