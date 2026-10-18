# -*- coding: utf-8 -*-
"""build_v7.py: v7 연구 결과 보고 덱(국문, 16:9). 명세 deck/deck_spec_v7.json. 문법 V7:
주황 제목(명사구) + 회색 부제 한 줄 + 그림이 본문을 채움(y 1.30–6.40) + 맨 아래 회색 한 줄 설명(12 pt). 하단 결론 줄·헤어라인 없음.
강조는 주황 외곽선 상자(채움 없음, 1.5 pt, 잉크 글자 16 pt) 쪽당 최대 1개. 그림: outputs/figures/paper/v7/slide/<stem>__<panels>.png, deck/assets/v7/<name>.png.
수치는 docs/numbers_v6.json 의 {nMacro} 토큰으로만 들어간다.
출력 deck/render/v7_report.{pptx,pdf}, deck/render/v7_report_pages/page-XX.png. 실행: OMP_NUM_THREADS=2 nice -n 10 python3 deck/build_v7.py [--no-pdf]
PDF 변환은 paper_report_lib.export_pdf(GPU 차단 환경변수 적용).
"""
from __future__ import annotations
import argparse, json, os, re, subprocess, sys, tempfile
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pptx.util import Inches, Pt  # noqa: E402
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN  # noqa: E402
from pptx.enum.shapes import MSO_SHAPE  # noqa: E402
from pptx.dml.color import RGBColor  # noqa: E402
import final_lib as fl  # noqa: E402
import paper_report_lib as L  # noqa: E402
from paper_report_lib import ACC, F_M, F_S, F_X, GRAY, GRAY2, INK, run, text  # noqa: E402

ROOT = L.ROOT; DECK = L.DECK; RENDER = DECK / "render"; OUTSTEM = os.environ.get("DECK_OUT", "v8_report"); PAGES = RENDER / f"{OUTSTEM}_pages"
PPTX = RENDER / f"{OUTSTEM}.pptx"; PDF = RENDER / f"{OUTSTEM}.pdf"
FIG = ROOT / "outputs" / "figures" / "paper" / "v7" / "slide"
DECKNATIVE = ROOT / "outputs" / "figures" / "paper" / "v7" / "deck"
DECKFIG = DECK / "assets" / "v7"
DECKFIG8 = DECK / "assets" / "v8"
ASSET = DECK / "assets" / "paper_report"
SCRATCH = Path(os.environ.get("PR_TMP", tempfile.gettempdir()))
NUM = json.load(open(ROOT / "docs" / "numbers_v6.json", encoding="utf-8")) if (ROOT / "docs" / "numbers_v6.json").exists() else {}
Y0, YB = 1.30, 6.40          # 본문 위·아래
CAP_Y = 6.62                 # 설명 줄
BOX_H = 1.20


def subst(obj):
    if isinstance(obj, str):
        return re.sub(r"\{(n[A-Za-z]+)\}", lambda m: NUM.get(m.group(1)[1:], m.group(0)), obj)
    if isinstance(obj, list):
        return [subst(x) for x in obj]
    if isinstance(obj, dict):
        return {k: subst(v) for k, v in obj.items()}
    return obj


SPEC_NAME = os.environ.get("DECK_SPEC", "deck_spec_v8.json"); SPEC = subst(json.load(open(DECK / SPEC_NAME, encoding="utf-8")))


CACHE = DECK / "assets" / "v7" / "_rgb"


def asset(name):
    """그림 파일을 찾고, RGBA 면 흰 바탕에 평탄화한 RGB 사본(캐시)을 돌려준다(PDF 변환 시 smask 를 없애기 위함)."""
    for p in (DECKNATIVE / name, FIG / name, DECKFIG8 / name, DECKFIG / name, ASSET / name):
        if p.exists():
            from PIL import Image
            with Image.open(p) as im:
                if im.mode != "RGBA":
                    return p
                CACHE.mkdir(parents=True, exist_ok=True); q = CACHE / p.name
                if not q.exists() or q.stat().st_mtime < p.stat().st_mtime:
                    bg = Image.new("RGB", im.size, (255, 255, 255)); bg.paste(im, mask=im.split()[3]); bg.save(q, dpi=im.info.get("dpi", (300, 300)))
                return q
    return None


def nominal(p):
    w, h = L.img_size(p); return w / 300.0, h / 300.0


def fit_cap(p, max_w, max_h, cap=1.6):
    w, h = L.fit(p, max_w, max_h); nw, nh = nominal(p)
    if w / nw > cap:
        w, h = nw * cap, nh * cap
    return w, h


def row_place(sl, paths, x, y, max_w, max_h, gap_in=0.30, cap=2.0):
    """여러 그림을 한 줄에 같은 높이로 두고 가로 가운데 정렬."""
    nom = [nominal(p) for p in paths]
    h = min(max_h, cap * min(nh for _, nh in nom))
    widths = [nw * h / nh for nw, nh in nom]; tot = sum(widths) + gap_in * (len(paths) - 1)
    if tot > max_w:
        f = (max_w - gap_in * (len(paths) - 1)) / (tot - gap_in * (len(paths) - 1)); h *= f; widths = [w * f for w in widths]; tot = max_w
    cx = x + (max_w - tot) / 2
    for p, w in zip(paths, widths):
        L.place(sl, p, cx, y, w=w, h=h); cx += w + gap_in
    return tot, h


def meaning_lines(meaning):
    """14 pt Pretendard 로 12 in 폭에 들어가는 글자 수(한글 약 88자)를 기준으로 줄 수를 센다."""
    return 0 if not meaning else max(1, int(np.ceil(len(meaning) / 88.0)))


def meaning_h(meaning):
    return 0.0 if not meaning else 0.30 * meaning_lines(meaning) + 0.08


def caption(sl, t, meaning=None):
    """의미 줄(잉크 14 pt, 그림 아래, 1–2줄)과 회색 설명 줄(12 pt, 맨 아래)."""
    if meaning:
        h = meaning_h(meaning)
        text(sl, L.ML, CAP_Y - h - 0.02, 12.0, h, [run(meaning, 14, INK, F_M)], align=PP_ALIGN.LEFT, line_spacing=1.08)
    if t:
        text(sl, L.ML, CAP_Y, 12.0, 0.30, [run(t, 12, GRAY, F_M)], align=PP_ALIGN.LEFT)


def accent_box(sl, x, y, w, h, lines):
    """주황 외곽선 상자(채움 없음). 줄 머리(첫 두 칸 띄기 전)는 SemiBold, 나머지 Medium 16 pt."""
    shp = sl.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    shp.fill.background(); shp.line.color.rgb = ACC; shp.line.width = Pt(1.5); shp.shadow.inherit = False; shp.name = "accent box"
    tf = shp.text_frame; tf.word_wrap = True; tf.margin_left = tf.margin_right = Inches(0.18); tf.margin_top = tf.margin_bottom = Inches(0.08)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    for i, ln in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph(); p.alignment = PP_ALIGN.LEFT
        head, _, body = ln.partition("  ")
        for t, f in ((head + "  ", F_S), (body, F_M)):
            r = p.add_run(); r.text = t; r.font.size = Pt(14 if w < 6.0 else (15 if len(lines) >= 3 else 16)); r.font.color.rgb = INK; r.font.name = f; L._set_fonts(r, f) if hasattr(L, "_set_fonts") else fl._set_fonts(r, f)
        p.space_after = Pt(4)
    return shp


def gap(sl, x, y, w, h, name):
    L.gap_marker(sl, x, y, w, h, ["그림 대기", name])


def cover(sl, s):
    bg = DECKFIG8 / "cover_bg_v8.png"; dark = bg.exists()
    if not dark:
        bg = ASSET / "cover_bg_v3.png"
    pic = sl.shapes.add_picture(str(bg), 0, 0, Inches(13.333), Inches(7.5)); pic.shadow.inherit = False
    white = RGBColor(0xFF, 0xFF, 0xFF); soft = RGBColor(0xC8, 0xD0, 0xDA)
    L.break_runs(text(sl, L.ML, 1.15, 6.9, 2.3, [run(s["title"], 32, ACC, F_X)], anchor=MSO_ANCHOR.TOP, line_spacing=1.12))
    text(sl, L.ML, 3.75, 6.9, 0.45, [run(s["subtitle"], 18, white if dark else GRAY2, F_M)])
    text(sl, L.ML, 4.55, 6.9, 0.40, [run(s["presenter"], 18, white if dark else INK, F_S)])
    text(sl, L.ML, 4.98, 6.9, 0.34, [run(s["affiliation"], 14, soft if dark else GRAY2, F_M)])
    L.logos_row(sl, x_left=L.ML, y_bottom=6.90, h=L.LOGO_H_COVER, gap=L.LOGO_GAP_COVER)


def refs(sl, s):
    half = (len(s["refs"]) + 1) // 2
    for x, chunk in zip((L.COL[0], L.COL[3]), (s["refs"][:half], s["refs"][half:])):
        text(sl, x, L.Y0, 5.85, 5.2, [[run(r, 12, INK, F_M)] for r in chunk], line_spacing=1.15, space_after=6)


def fig_slide(sl, s):
    paths = [asset(a) for a in s["assets"]]
    if s.get("deck_native") and not all(paths) and s.get("fallback_assets"):
        paths = [asset(a) for a in s["fallback_assets"]]
    box = s.get("box"); box_h = (0.72 if len(box) == 1 else BOX_H) if box else 0.0; body_h = (YB - Y0) - (box_h + 0.18 if box else 0.0) - (meaning_h(s.get("meaning")) + 0.02 if s.get("meaning") else 0.0)
    if all(paths) and (s.get("deck_native") or s.get("deck_only")) and len(paths) == 1:
        p = paths[0]; w, h = L.fit(p, 12.0, body_h); L.place(sl, p, L.ML + (12.0 - w) / 2, Y0, w=w, h=h)
    elif not all(paths):
        gap(sl, L.ML, Y0, 12.0, body_h, ", ".join(a for a, p in zip(s["assets"], paths) if p is None))
    else:
        row_place(sl, paths, L.ML, Y0, 12.0, body_h)
    if box:
        accent_box(sl, L.ML, YB - box_h, 12.0, box_h, box)
    caption(sl, s.get("caption", ""), s.get("meaning"))


def columns_slide(sl, s):
    """결론·의의 쪽: 위에 세 열짜리 덱 전용 그림 한 장(12 × 2.9 in), 아래에 열마다 주황 외곽선 상자(머리 줄 + 본문 2줄)."""
    cols = s["columns"]; n = len(cols); gap_in = 0.30; w = (12.0 - gap_in * (n - 1)) / n
    p = asset(s["assets"][0]) if s.get("assets") else None
    fig_h = 2.90
    if p is not None:
        fw, fh = L.fit(p, 12.0, fig_h); L.place(sl, p, L.ML, Y0, w=fw, h=fh); fig_h = fh
    else:
        gap(sl, L.ML, Y0, 12.0, fig_h, s.get("assets", [""])[0])
    box_y = Y0 + fig_h + 0.18; box_h = (YB - box_y) - (meaning_h(s.get("meaning")) + 0.02 if s.get("meaning") else 0.0)
    for i, c in enumerate(cols):
        x = L.ML + i * (w + gap_in)
        shp = sl.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(box_y), Inches(w), Inches(box_h))
        shp.fill.background(); shp.line.color.rgb = ACC; shp.line.width = Pt(1.5); shp.shadow.inherit = False; shp.name = "accent box"
        tf = shp.text_frame; tf.word_wrap = True; tf.margin_left = tf.margin_right = Inches(0.16); tf.margin_top = tf.margin_bottom = Inches(0.08); tf.vertical_anchor = MSO_ANCHOR.TOP
        para = tf.paragraphs[0]; para.alignment = PP_ALIGN.LEFT
        r = para.add_run(); r.text = c["head"]; r.font.size = Pt(16); r.font.color.rgb = ACC; r.font.name = F_X; fl._set_fonts(r, F_X); para.space_after = Pt(4)
        for ln in c["lines"]:
            q = tf.add_paragraph(); q.alignment = PP_ALIGN.LEFT; r = q.add_run(); r.text = ln; r.font.size = Pt(14); r.font.color.rgb = INK; r.font.name = F_M; fl._set_fonts(r, F_M); q.space_after = Pt(3)
    caption(sl, s.get("caption", ""), s.get("meaning"))


def build(out=PPTX):
    prs = fl.new_deck()
    for i, s in enumerate(SPEC["slides"]):
        sl = fl.blank(prs)
        if s["type"] == "cover":
            cover(sl, s)
        elif s["type"] == "references":
            L.header(sl, s["title"], None, i); refs(sl, s)
        else:
            L.header(sl, s["title"], s.get("subtitle"), i)
            (columns_slide if s["type"] == "columns" else fig_slide)(sl, s)
        L.notes(sl, s.get("notes"), None, None)
    RENDER.mkdir(parents=True, exist_ok=True); prs.save(out)
    print(f"[pptx] {out.relative_to(ROOT)} {len(prs.slides)} 쪽"); return prs


def audit():
    rep = L.pptx_audit(str(PPTX)); ext = dict(L.deck_extra_audit(str(PPTX))); hits = 0
    for i, v in rep:
        e = ext[i]; msgs = []
        for k in ("off_font", "off_size", "big_num"):
            if v.get(k):
                msgs.append(f"{k} {v[k]}")
        bad = [x for x in (v.get("pic_scale_bad") or []) if not (0.85 <= x <= 1.65)]
        if bad:
            msgs.append(f"pic_scale {bad}")
        if e.get("dash"):
            msgs.append(f"dash {e['dash']}")
        if msgs:
            hits += 1; print(f"  [audit] p{i}: {'; '.join(msgs)}")
    print(f"[audit] 지적 쪽 {hits}/{len(rep)}")


def render():
    L.export_pdf(PPTX, PDF, SCRATCH / "lo_v7")
    PAGES.mkdir(parents=True, exist_ok=True)
    for old in PAGES.glob("page-*.png"):
        old.unlink()
    subprocess.run(["nice", "-n", "10", "pdftoppm", "-r", "80", "-png", str(PDF), str(PAGES / "page")], check=True)
    print(f"[pdf] {PDF.relative_to(ROOT)} 쪽 PNG {len(list(PAGES.glob('page-*.png')))}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--no-pdf", action="store_true"); a = ap.parse_args()
    build(); audit()
    if not a.no_pdf:
        render()
