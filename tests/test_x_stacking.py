"""계획 2.2 '구현·시험'이 등록한 시험 파일 이름(tests/test_x_stacking.py). 시험 본문은 작업 지시에 따라 tests/test_x_xb.py 에 있다
(docs/research/2026-10-04/impl_notes/x_multisource_stacking.md 8절). 이 파일은 그 시험을 다시 내보내는 얇은 파일이다.

같은 실행에서 tests/test_x_xb.py 도 모이면(디렉터리, 글롭, 인자 없음) 같은 시험이 두 번 돌지 않게 이 파일의 시험을 건너뛴다. 이 파일 이름만
줄 때(python3 -m pytest tests/test_x_stacking.py) 등록 항목 (a)–(i)와 추가 항목이 모두 돈다. 계획의 개정 이력에 이름 대응을 적는 일은
다음 허용 개정에서 한다(계획은 이 작업에서 고치지 않는다).
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _only_this_file(argv=None) -> bool:
    """pytest 명령의 경로 인자가 모두 이 파일을 가리키는가(그때만 다시 내보낸 시험을 돌린다)."""
    av = list(sys.argv[1:] if argv is None else argv)
    paths = [v for v in av if ".py" in v or os.path.isdir(v)]
    return bool(paths) and all("test_x_stacking" in v for v in paths)


if _only_this_file():
    from test_x_xb import *                                                                  # noqa: F401,F403,E402
else:
    def test_registered_name_maps_to_test_x_xb():
        """이름 대응 확인: 등록 항목의 시험은 같은 실행에서 tests/test_x_xb.py 로 돈다."""
        src = (Path(__file__).resolve().parent / "test_x_xb.py").read_text()
        for item in ("test_a_", "test_b_", "test_c_", "test_d_", "test_e_", "test_f_", "test_g_", "test_h_", "test_i_"):
            assert f"def {item}" in src, item
