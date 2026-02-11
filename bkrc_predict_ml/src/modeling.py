import time

# ===== Data =====
import numpy as np
import pandas as pd

# ===== Modeling utils =====
from sklearn.model_selection import TimeSeriesSplit
from imblearn.over_sampling import SMOTE

# ===== PyCaret (Classification) =====
from pycaret.classification import (
    setup, compare_models, create_model, tune_model, finalize_model,
    predict_model, save_model, pull, get_config, evaluate_model
)

# ===== (Optional) External model backends used by PyCaret =====
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

def split_train_test(
    df:pd.DataFrame,
    trainYm: str,
    testYm: str
):
    """
    df : 전체 데이터
    trainYm : 학습 데이터 시작일자, "YYYY-MM-DD"
    testYm : 테스트 데이터 시작일자, "YYYY-MM-DD"
    """
    
    # ===== 1) 날짜 파생 + 피처 선택 =====
    df["MONTH_dt"] = pd.to_datetime(df["MONTH"].astype(str) + "01", format="%Y%m%d")
    df_use = df.sort_values(["COM_RGNO","MONTH_dt"]).reset_index(drop=True)
    
    # ===== 3) 고정 기간 분리 =====
    train_start = pd.Timestamp(trainYm)
    train_end   = pd.Timestamp(testYm)    
    train_mask  = (df_use["MONTH_dt"] >= train_start) & (df_use["MONTH_dt"] < train_end)
    train_df    = df_use.loc[train_mask].copy()
    test_df     = df_use.loc[~train_mask].copy() 
    
    train_df = train_df.dropna()
    test_df = test_df.dropna()
    
    # 점검(선택): 기간/행수 확인
    print(train_df["MONTH_dt"].min(), "→", train_df["MONTH_dt"].max(), len(train_df))
    print(test_df["MONTH_dt"].min(),  "→", test_df["MONTH_dt"].max(),  len(test_df))

    return train_df, test_df

def createModel(train_df,test_df,target) :
    if target == "isClosed" :
        ignore_cols = ['COM_RGNO','MONTH','BZCD','MONTH_dt','CRI_NEW']
    else : 
        ignore_cols = ['COM_RGNO','MONTH','BZCD','MONTH_dt','isClosed','CRI_NEW']
    
    sm = SMOTE(sampling_strategy='auto', k_neighbors=5)
    
    exp = setup(
        data=train_df,
        target=target,
        test_data=test_df,         # ← 최종 평가용(2022-04~12)
        session_id = 25,
        train_size=0.8,            # ← 학습/검증 분리 비율 (예: 80% train, 20% val)
        fold = 4,
        fold_strategy = 'stratifiedkfold',   # fold 안에서 train/val는 (k-1 / k) : 1/k로 나눔
        data_split_shuffle=True,  # 셔플 O
        imputation_type="simple",  # data자체에 NaN값이 없음
        normalize=False,
        remove_multicollinearity=True,
        multicollinearity_threshold=0.75,
        fix_imbalance=True,
        fix_imbalance_method=sm,
        ignore_features=ignore_cols,
        
        # 기타설정
        use_gpu = False , # GPU 사용 (CUDA 미지원 빌드)
        log_plots = True, # 중요 그래프(Confusion Matrix, ROC 등 )로깅
        log_data = True, # 샘플 데이터 로깅
        verbose=True, # 로그 최소화
    )

    candidates = {}
    timing_results = {}

    model_names = [
        "lr",
        "ridge",
        "xgboost",
        # "nb",
        # "lda",
        # "qda",
        # "catboost",
        # "et",
        # "rf",
        # "svm",
        # "dt",
        # "knn",
        # "gbc"
    ]
    
    print("=== 모델별 학습 시작 ===\n")
    
    for name in model_names:
        print(f"[{name}] 학습 시작...")
        start = time.time()
    
        try:
            if name == "lr" : 
                model = create_model(name,max_iter = 2000)
            else:
                model = create_model(name)
            
            candidates[name] = model
    
            elapsed = time.time() - start
            timing_results[name] = round(elapsed, 2)
            print(f"✅ {name} 학습 완료 (소요시간: {elapsed:.2f}초)\n")
    
        except Exception as e:
            elapsed = time.time() - start
            timing_results[name] = None
            print(f"❌ {name} 학습 실패 (소요시간: {elapsed:.2f}초)")
            print(f"   오류 메시지: {str(e)}\n")
    
    print("=== 모든 모델 학습 완료 ===\n")
    
    # 📊 모델별 학습 소요시간 요약
    timing_df = pd.DataFrame(
        list(timing_results.items()),
        columns=["model", "train_time_sec"]
    )
    
    # ✅ holdout 예측 및 지표 수집
    holdout_metrics = []
    for name, model in candidates.items():
        _ = predict_model(model)
        met = pull().copy()
        met['model'] = name
        holdout_metrics.append(met)
    
    holdout_results = pd.concat(holdout_metrics, ignore_index=True)
    
    # ✅ 모델별 평균 지표 계산
    avg_metrics = (
        holdout_results
        .groupby('model')[['Accuracy', 'Recall', 'Prec.', 'F1', 'AUC']]
        .mean()
        .reset_index()
    )
    
    # ✅ 시간 정보 병합
    summary = avg_metrics.merge(timing_df, on='model', how='left')
    summary.sort_values('Recall', ascending=False, inplace=True)
    print(summary.columns)
    # ✅ 최종 결과 출력
    print("\n=== 모델별 Holdout 성능 및 학습시간 요약 ===")

    return candidates, summary



# 2) 공통 그리드 서치 함수
def tune_with_grid(model_name, grid, optimize="Recall", fold=4):
    print(f"\n=== [{model_name}] 기본 모델 생성 ===")
    base = create_model(model_name, fold=fold, verbose=False)

    print(f"=== [{model_name}] GridSearch 튜닝 시작 ===")
    tuned = tune_model(
        base,
        optimize=optimize,
        fold=fold,
        search_library="scikit-learn",   # GridSearchCV
        search_algorithm="grid",
        custom_grid=grid,
        choose_better=True,
        verbose=False
    )

    # 교차검증 결과표
    cv_table = pull()
    print(f"=== [{model_name}] CV 결과 ===")
    print(cv_table.head())

    return tuned, cv_table

def gridSearchModel(trainSet,testSet) :
    target = 'isClosed'
    ignore_cols = ['COM_RGNO','MONTH','MONTH_dt']
    
    exp = setup(
        data=train_df,
        target=target,
        test_data=test_df,         # ← 최종 평가용(2022-04~12)
        session_id = 25,
        train_size=0.8,            # ← 학습/검증 분리 비율 (예: 80% train, 20% val)
        fold = 4,
        fold_strategy = 'kfold',   # fold 안에서 train/val는 (k-1 / k) : 1/k로 나눔
        data_split_shuffle=True,  # 셔플 O
        imputation_type="simple",  # data자체에 NaN값이 없음
        normalize=False,
        remove_multicollinearity=False,
        multicollinearity_threshold=0.75,
        fix_imbalance=True,
        ignore_features=ignore_cols,
        
        # 기타설정
        use_gpu = False , # GPU 사용 (CUDA 미지원 빌드)
        log_plots = True, # 중요 그래프(Confusion Matrix, ROC 등 )로깅
        log_data = True, # 샘플 데이터 로깅
        verbose=True, # 로그 최소화
    )

    # 1) 모델별 그리드 정의 (LR, XGBoost만)
    grids = {
        "lr": {  # LogisticRegression
            "C": [0.1, 1.0, 3.0, 10.0],
            "penalty": ["l2"],                  # l1은 solver 제약
            "solver": ["lbfgs", "liblinear"],   # 데이터 특성에 따라 선택
            "max_iter": [2000, 2000]
        },
        "xgboost": {
            "n_estimators": [300, 600, 900],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.03, 0.1],
            "subsample": [0.7, 1.0],
            "colsample_bytree": [0.7, 1.0],
            "min_child_weight": [1, 5]
            # 필요시 규제항도 추가 가능: "reg_lambda": [1.0, 5.0]
        }
    }

    # 3) 두 모델만 튜닝
    opt_metric = "Recall"
    kfolds = 4
    
    tuned_models = {}
    cv_results = {}
    
    for name in ["lr", "xgboost"]:
        tuned, cv_tbl = tune_with_grid(name, grids[name], optimize=opt_metric, fold=kfolds)
        tuned_models[name] = tuned
        cv_results[name] = cv_tbl
    
    # 4) 테스트셋 성능 비교 (setup에 test_data 지정됨)
    test_scores = {}
    for name, mdl in tuned_models.items():
        print(f"\n=== [{name}] 테스트셋 평가 ===")
        _ = predict_model(mdl)          # test_data 기준 성능 산출
        score_tbl = pull()
        test_scores[name] = score_tbl
        cols = [c for c in ["Model","Accuracy","AUC","Recall","Precision","F1"] if c in score_tbl.columns]
        print(score_tbl[cols])
    
    # 5) Recall 기준 최종 모델 선택
    def get_recall(df):
        # PyCaret 표에서 단일 행으로 귀결됨(테스트셋). 컬럼 존재 가정.
        return float(df["Recall"].values[0]) if "Recall" in df.columns else -1.0
    
    best_name = max(test_scores.keys(), key=lambda k: get_recall(test_scores[k]))
    final_model = finalize_model(tuned_models[best_name])
    print(f"\n🎯 최종 선택 모델: {best_name}")