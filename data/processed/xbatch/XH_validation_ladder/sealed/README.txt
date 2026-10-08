봉인 폴더(계획 docs/EXPERIMENT_PLAN_FINAL_BATCH_2026-10-04.md 0.3 열람 순서)

이 폴더의 표는 열람 순서보다 먼저 만들어진 집계 표다(R1a·R1b 집계, XD-alg 의 비알래스카 행 등). 0.3 의 열람 순서에 정한 시점 전에는
열지 않는다. 봉인을 푼 시각은 계획 8절에 적는다. 쓰기는 xbatch_core.write_sealed 만 하며 화면에는 파일 이름, 행 수, sha256 앞 16자만 남긴다.
sealed_manifest.json 에는 파일별 행 수, sha256, 기록 시각만 있다(표의 내용은 없다).
