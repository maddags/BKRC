# brkc_predict_ml

기업 부실 징후 예측을 ML로 수행하고, Permutation Importance 기반 변수 중요도를 산출하는 프로젝트입니다.

## 프로젝트 구조

```
brkc_predict_ml/
├── run.py                  # 메인 실행 스크립트 (전처리 → 학습 → 모델 선택 → 변수 중요도)
├── src/
│   ├── preprocessing.py    # 데이터 전처리 (changeCdtStatus, dataSetting)
│   ├── modeling.py         # 모델 학습 및 평가 (split_train_test, createModel)
│   └── scoring.py          # 스코어링
├── final_modeling.ipynb    # 모델링 과정 노트북 (참고용)
├── final_scoring.ipynb     # 스코어링 과정 노트북 (참고용)
├── requirements.txt        # 패키지 의존성
└── README.md
```

## 실행 방법

```bash
python run.py
```

## 파이프라인 흐름

1. **데이터 호출** — parquet 파일 로드
2. **전처리** — `changeCdtStatus()` : 폐업 여부(`isClosed`) 라벨 생성
3. **데이터 셋팅** — `dataSetting()` : 피처 선택 및 필터링
4. **Train/Test 분리** — `split_train_test()` : 기간 기반 분리 (학습: 2022-01~03, 테스트: 2022-04~12)
5. **모델 학습** — `createModel()` : PyCaret 기반 다중 모델 학습 (lr, ridge, xgboost)
6. **최적 모델 선택** — Recall 기준 최고 성능 모델 자동 선택
7. **변수 중요도** — `pmi()` : Permutation Importance (scoring=recall) 산출

## 사용 피처

| 피처명 | 설명 |
|--------|------|
| AVG_NMB_LST_YoY | 평균 상실 수 전년 대비 |
| COUNT_COM_RGNO | 사업장 수 |
| COUNT_COM_RGNO_YoY | 사업장 수 전년 대비 |
| AVG_NMB_ACQ_YoY | 평균 취득 수 전년 대비 |
| AVG_EMP_FLUX_NET | 평균 고용 순변동 |
| EMP_FLUX_VOL | 고용 변동 규모 |
| AVG_NMB_SBS_MoM | 평균 가입자 수 전월 대비 |
| AVG_AMT_RATE | 평균 금액 비율 |
| 동종업계_대비_고용규모 | 동종업계 대비 고용 규모 |
| AVG_AMT_MoM | 평균 금액 전월 대비 |
| AVG_EMP_FLUX_NET_MoM | 평균 고용 순변동 전월 대비 |

## 데이터 위치

실제 데이터 위치 수정 후 사용할 것

    hdfs : /user/hhan/shared_folder/bkrc/causality/keta_features/newBkrcPensionInfo_atan202201_202212_12.parquet

## 주요 의존성

- Python 3.x
- pandas 1.5.3
- scikit-learn 1.4.2
- pycaret 3.3.1
- imbalanced-learn 0.12.0
- xgboost 2.1.4
- lightgbm 4.6.0

## Version

- **Ver 1.0.0** : 초기버전
- **Ver 1.0.1** : run.py 사용 import 및 dead code 제거, pmi() 함수 내 불필요한 변수·중복 정렬 로직 경량화