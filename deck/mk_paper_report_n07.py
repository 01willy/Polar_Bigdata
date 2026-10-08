# -*- coding: utf-8 -*-
"""mk_paper_report_n07.py: 새 7쪽(Stefan 물리 모델과 계수 재보정)의 개념 좌표도.

옛 9쪽 그림(mk_paper_report_figs.chart_s09)과 같은 자료·색·글자를 쓰고 높이만 4.75 → 4.15 in 로 줄인다(아래에 재보정 수식 줄과
기호 열쇠 줄을 두기 위함). 다른 작업자가 chart_s09 의 양식(v4 색·글자)을 고쳐도 따라가도록 함수 원문을 실행 때 읽어 높이와 저장 이름만
바꾼다. MANIFEST.json 은 쓰지 않는다(덱 빌더가 PNG 를 경로로 넣는다).
실행: cd deck && OMP_NUM_THREADS=2 nice -n 10 python3 mk_paper_report_n07.py
"""
import inspect
import sys
from pathlib import Path

DECK = Path(__file__).resolve().parent
sys.path.insert(0, str(DECK))
import mk_paper_report_figs as M  # noqa: E402

H_NEW = 4.15
OUT = M.OUT / "N07_stefan_recal.png"


def make():
    src = inspect.getsource(M.chart_s09)
    reps = [("W, H = 12.0, 4.75", f"W, H = 12.0, {H_NEW}"),
            ('finish(fig, "S09", "stefan_concept"', 'finish(fig, "N07", "stefan_recal"'),
            ('slide="S09", page=9', 'slide="N07", page=7'),
            ("def chart_s09(spec):", "def chart_n07(spec):")]
    for a, b in reps:
        assert src.count(a) == 1, f"chart_s09 원문에서 '{a}' 를 찾지 못함(그림 함수가 바뀜)"
        src = src.replace(a, b)
    ns = dict(vars(M))
    exec(compile(src, "chart_n07", "exec"), ns)
    M.setup_mpl()
    ns["chart_n07"](M.load_spec())
    return OUT


if __name__ == "__main__":
    print(make())
