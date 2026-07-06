# 시각화 & 산출물 정리 규칙

> 모든 결과 생성 스크립트는 이 규칙을 따른다. 경로는 `src/polar/outputs.py` 헬퍼로 얻는다.
> 한글 폰트는 `src/polar/plotstyle.py`의 `use_korean()`으로 적용한다.

## 1. 폴더 구조

```
outputs/
  figures/                 분석 그래프 (카테고리별 하위폴더)
    00_concept/            개념도·연구 프레이밍
    01_data/               데이터 개요·커버리지·규모
    02_eval/               CV 평가·누설 진단
    03_covariate_upgrade/  공변량 비교(WorldClim vs ERA5 등)
    04_ground_temp/        지중온도 프로파일·지도
    05_alaska_pilot/       초기 알래스카 3D 파일럿(레거시)
    06_spatial_dl/         공간 DL(패치 CNN/Neural Field) 결과
  maps/                    2D 공간 레이어 — 예측 ALT 지도 등 (PNG/GeoTIFF)
  volumes_3d/              3D 지중 열구조 렌더·메시(.vtp)
  animations/              GIF/MP4 — 깊이 슬라이스, 계절 변화, 회전 등
  models/                  DL 체크포인트(.pt)
```

파일명: 폴더가 맥락을 주므로 **전역 번호 접두 없이** 서술형(kebab/snake)으로. 예 `maps/alt_alaska_pred.png`.

## 2. 직관적 시각화 필수 (그래프만으로 끝내지 말 것) — 기본 원칙

**★ 절대 원칙: 어떤 실험/모델 결과든 막대·산점도 같은 "그래프"만 내놓지 말 것.**
공간 데이터를 다루므로, 결과는 반드시 **사람이 한눈에 이해하는 공간 시각화**로 보여준다.
예: 공간블록 CV를 했으면 → 블록을 지도에 색칠, 예측 ALT를 지도에 레이어로, 오차를 지도에 빨강/파랑으로.
"모델이 예측한 ALT가 공간적으로 어떻게 생겼나"를 항상 png/gif로 만든다. 이걸 빠뜨리면 결과 미완성으로 간주.

결과 분석 시 **그래프에 더해**, 사람이 한눈에 이해할 **직관적 공간 시각화를 최소 2종 이상** 함께 제시한다:

- **2D 레이어** (`maps/`): 예측 ALT/지온을 실제 지도 위에 색으로. 관측점 오버레이, 오차 지도.
- **3D 레이어** (`volumes_3d/`): 0~20m 지중 열구조 볼륨, 0°C 등온면(동토 경계), 시추공 프로파일 3D 배치.
- **애니메이션** (`animations/`): 깊이 슬라이스 스캔(위→아래), 계절별 융해 전선 이동, 3D 회전.

원칙: "이 결과가 공간적으로 어떻게 생겼나"를 비전문가도 보게 만든다. 수치표는 근거로 남기되, 헤드라인은 직관적 이미지로.

## 3. 코드 규약

```python
import sys; sys.path.insert(0, "src")
from polar.plotstyle import use_korean
from polar.outputs import figpath, mappath, volpath, animpath
plt = use_korean()
...
figpath("covariate_upgrade", "era5_rescore")  # outputs/figures/03_covariate_upgrade/era5_rescore.png
mappath("alt_alaska_pred")                    # outputs/maps/alt_alaska_pred.png
```
