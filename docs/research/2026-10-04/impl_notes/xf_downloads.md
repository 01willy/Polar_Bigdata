# XF 자료 취득 기록(2026-10-04)

대상: `docs/EXPERIMENT_PLAN_FINAL_BATCH_2026-10-04.md`(개정 1, T0 = 2678100, 2026-10-04 14:22:39 +09:00) 2.6절 '자료와 우선순위' 표와 `open_regions.md` 8절. 작업 시각 2026-10-04 14:24–14:40 +09:00. 계획 본문은 고치지 않았다.

## 1. 취득 결과

| 순서(2.6 표) | 폴더(`data/raw/`) | 자료 | 약관(확인 근거) | 받은 크기 | 상태 |
|---|---|---|---|---|---|
| 1 | swe_stordalen_crill | Stordalen 자동 챔버 지점 ALT·지하수위 2003–2017(Zenodo 10420396) | CC BY 4.0(Zenodo API) | xlsx 194,902 B, pdf 2,245,763 B | 받음, md5 Zenodo 와 같음 |
| 2 | sjm_adventdalen_wendt2023 | Adventdalen 코어 12지점, 2023 ALT(Zenodo 11187360) | CC BY 4.0(Zenodo API) | xlsx 51,908 B | 받음, md5 같음 |
| 3 | (해당 없음) | NAtlantic 약관 확인분 판 재계산 | | | 로컬 `--count-only` 몫. 이 작업 범위 밖 |
| 4 | ak_bnz_blackspruce150 | BNZ LTER knb-lter-bnz.140 | 제한 약관(DataONE EML 사본, 아래 3절) | 메타데이터 11,012 B 만 | **보류(차단)** |
| 5 | (폴더 없음, 4번 폴더에 기록) | BNZ LTER knb-lter-bnz.605 | 미확인 | 없음 | **보류(차단)** |
| 6 | ak_viper_td | ViPER 2015·2017·2018(ADC A22J6848J) | CC BY 4.0(EML) | csv 9개 81,433 B | 받음, EML 체크섬 같음 |
| 7 | ak_anaktuvuk_selawik | Anaktuvuk Pass·Selawik 2011(ADC A2N596) | CC BY 4.0(EML) | xlsx 868,524 B, pdf 262,148 B | 받음, SHA-256 같음 |
| 8 | ca_naat_ornl1386 | NAAT·북사면 격자 융해 깊이(ORNL 1386) | NASA 공개 자료 지침(UMM-C LicenseURL, 원문 주소 404) | 융해 깊이 csv 20개 126,562 B, 안내 pdf 1,975,314 B | 받음, ORNL sha256 같음. Earthdata 로그인 사용 |
| 9 | ca_parks_qausuittuq | Parks Canada Qausuittuq(Dundee Bight) | Open Government Licence - Canada(CKAN) | csv 8,686 B + 1,379 B | 받음. 좌표 없음 |
| 10 | ca_moses_tuk2021 | MOSES 2021 Tuktoyaktuk·Lake 3(PANGAEA 949181) | CC BY 4.0(원자료 머리말) | tab 14,123 B, README pdf 1,499,055 B | 받음. 열람 사고(2절) |
| 11 | ca_usgs_aln | USGS 유콘강 유역 ALN | 제약 없음(FGDC accconst·useconst 'none') | 메타데이터 10,372 B 만 | **보류(ScienceBase Cloudflare 403)** |
| 12 | grl_petrone_pangaea | Petrone 2016 캉에를루수아크(PANGAEA 845258) | CC BY-NC-SA 3.0(metainfo) | zip 2,534,367 B(풀면 21.3 MB) | 받음. CC BY 공개 파생 표에 넣지 않음 |
| 13 | qtp_xiao_wenquan | Xiao 2026 원취안 GPR 810점(Zenodo 21366503) | CC BY 4.0(Zenodo API) | csv 34,559 B + 코드 3개 | 받음, md5 같음 |

전체 디스크 사용 약 31.5 MB(/home). 각 폴더에 SOURCE.md(약관 인용, 방법, 측정 월, md5·sha256)를 두었다.

`open_regions.md` 4.3 의 Tarnocai & Bockheim 2011, Tulemalu Lake 는 계획 2.6 표에 없어서 받지 않았다.

## 2. 열람 범위와 사고

- 모든 xlsx·csv 는 수치 칸을 자료형 표지로 가린 상태에서 시트 이름, 크기, 머리글, 날짜 칸의 연·월 구성, 칸 수만 출력했다. 평균·분산·계수 비 등 라벨 통계는 계산하지 않았다.
- **사고 1건**: 2026-10-04 14:30 무렵, MOSES README PDF(`ca_moses_tuk2021/Readme_mcan21_Permafrost_Probe.pdf`)의 텍스트 추출본에 방법 문장을 찾는 grep 을 했고, 검색어가 PDF 의 표 1 에도 걸려 1–17번 행 가운데 14행(모두 Lake 3, 2021-09-14)의 평균·개별 ALT 값과 절단 표기가 작업 화면에 출력되었다. 통계 계산과 파일 기록은 없었다. Tuktoyaktuk Island 행은 출력되지 않았다.
  - 영향: MOSES 는 새 macro 지역이 아니라 CA-2 셀 보강(L40 형식 민감도)이다. XF-1–XF-3 의 맹검 대상(이번 조사 뒤 처음 확보한 적격 새 macro 지역)에는 해당하지 않는다. 다만 MOSES Lake 3 행은 '비맹검 부분 포함'으로 표시해야 한다(계획 0.3 의 출력 제한 위반으로 기록).
  - 재발 방지: 이후 PDF·문서의 방법 확인은 줄 단위 grep 대신 쪽 범위를 지정해 본문 절만 읽고, 표가 있는 쪽은 열지 않는다.
- 다른 자료원에서 깊이 값이 화면에 나온 일은 없다. Petrone zip 을 처음 `.tab` 이름으로 받았을 때 압축 바이트가 출력되었으나 압축 상태라 읽을 수 있는 값은 없다.

## 3. BNZ(EDI) 상태

- EDI PASTA: `listDataPackageRevisions` 가 공개 사용자에게 HTTP 403(140, 605 모두). 조사 당시의 `readMetadata` 거부와 같은 계열로, **여전히 차단**이다.
- DataONE 조정 노드의 knb-lter-bnz.140.18(obsoletedBy 없음, pubDate 2005-12-10) 메타데이터는 공개로 읽혔다. 그 `intellectualRights` 는 'Data will not be released without proper permission first being obtained from the investigator who generated the data ... data is not to be sold or redistributed' 이다. 계획의 '약관 확인분만 주 판정' 규칙에 맞는 공개 약관이 아니므로 자료 본체(`140_activelayerdpth.txt`)는 요청하지 않았다.
- knb-lter-bnz.605 는 DataONE 색인에 없다.
- 남은 사용자 행동: EDI 포털에서 현재 개정판의 약관과 측정 월 확인(계획 2.6, 7.2-4). BNZ 는 어떤 경우에도 XD 개발 과제에 넣지 않고 XC F2 서술에만 쓴다.

## 4. 계획과의 불일치·주의(문자 그대로 읽은 처리)

1. Stordalen 제목은 2003–2017 이나 DATA 시트에 2018년 6월 A L 기록 1행이 있다. 처리: 원파일을 그대로 두고 파서 단계에서 연도 규칙(1990년 이후)만 적용한다.
2. Adventdalen 좌표(UTM X, Y)의 측지계·구역이 파일에 없다. UTM 33N 가정 시 78.185–78.227°N, 15.781–16.150°E. 라벨 열 후보가 셋(현장 융해 깊이, 침하 보정, ALT)이다. 처리: 파서가 값 계산 전에 라벨 열을 등록한다. 이 기록은 결정을 하지 않는다.
3. Qausuittuq CSV 와 CKAN 메타데이터에 좌표가 없다. 2.6 표의 기대 효과(CA-1 보강)는 외부 좌표 확보 전에는 조립할 수 없다.
4. NAAT 의 UMM-C 약관 주소가 404 다. NASA Earthdata 의 공개 자료 이용 지침 문구로 대신 적었다. 계획 2.6 표의 '공개'와 맞는다.
5. Xiao 원취안 CSV 에 날짜와 ALT 단위가 없다. 계획 2.6 표는 이 자료를 '티베트 블록 안 밀도' 용도로만 둔다.
6. Petrone 은 CC BY-NC-SA 3.0 이다. 계획 2.6 표대로 CC BY 로 공개할 v5 파생 표에 넣지 않는다. 원자료는 `data/raw/`(gitignore 대상)에만 둔다.
7. USGS ALN 은 약관상 제약이 없으나 ScienceBase 가 서버 요청을 막는다. 봇 확인 우회는 하지 않았다.
8. 계획 0.3 의 '약관 문구와 md5 를 메타에 적는다'는 각 SOURCE.md 의 약관 인용과 md5·sha256 표로 이행했다.

## 5. 다음 단계(계획 2.6 마감)

- T0 + 24 h 안: `parse_ext_swe_stordalen_crill.py`, `parse_ext_sjm_adventdalen_wendt2023.py` 파싱 뒤 라벨 값 없이 NAtlantic v5 의 `--count-only`. 합집합이 8 미만이면 NAtlantic v5 민감도 행을 점 추정으로 확정한다.
- T0 + 7일: BNZ 약관·측정 월(사용자), USGS ALN 브라우저 내려받기(사용자), 약관 회신. 적격 새 macro 지역이 없으면 R4 의 XF 부분과 XC-F3 를 '시험하지 못함'으로 확정한다.
