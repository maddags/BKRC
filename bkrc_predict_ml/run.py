import pandas as pd

from src.preprocessing import changeCdtStatus, dataSetting
from src.modeling import split_train_test, createModel

from sklearn.inspection import permutation_importance
from pycaret.classification import get_config


def pmi(selected_model):
    """Permutation Importance를 계산하여 중요도 내림차순 DataFrame을 반환한다."""
    X_train = get_config("X_train")
    y_train = get_config("y_train")

    imp = permutation_importance(
        selected_model,
        X_train[selected_model.feature_names_in_],
        y_train,
        scoring="recall",
        n_repeats=20,
    )

    imp_df = pd.DataFrame({
        "feature": selected_model.feature_names_in_.tolist(),
        "importance_mean": imp.importances_mean,
        "importance_std": imp.importances_std,
    }).sort_values("importance_mean", ascending=False)

    return imp_df

def main():
    # 데이터 호출
    fpath = '/home/hhan/workspace/hhan/01_bkrc/ML/data/newBkrcPensionInfo_atan202201_202212_12.parquet'
    data = pd.read_parquet(fpath)

    # 데이터 전처리
    df = changeCdtStatus(data)
    
    # 모델 파라미터 설정
    target = 'isClosed'
    trainYm = "2022-01-01"
    testYm = "2022-04-01"
    feature_cols = ['AVG_NMB_LST_YoY','COUNT_COM_RGNO' ,'COUNT_COM_RGNO_YoY', 'AVG_NMB_ACQ_YoY', 'AVG_EMP_FLUX_NET',
                    'EMP_FLUX_VOL','AVG_NMB_SBS_MoM','AVG_AMT_RATE','동종업계_대비_고용규모','AVG_AMT_MoM','AVG_EMP_FLUX_NET_MoM']
    
    # 모델 데이터 셋팅
    model_set = dataSetting(df,feature_cols,target)

    # 학습 데이터, 테스트 데이터 분리
    train_df, test_df = split_train_test(model_set, trainYm, testYm)

    # ML별 모델 학습
    model_list, summary = createModel(train_df, test_df,target)

    # 가장 성능이 좋은 모델 선택 (Recall 기준, 이미 summary가 Recall 내림차순 정렬됨)
    best_model_name = summary.iloc[0]['model']
    selected_model = model_list[best_model_name]
    print(f"최적 모델: {best_model_name}")

    # permutation Important 생성
    pmi_values = pmi(selected_model)

    print(pmi_values)

if __name__ == '__main__':
    main()