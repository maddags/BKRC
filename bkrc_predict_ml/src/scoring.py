import pandas as pd
import numpy as np

def bin_edges_define(data, f_cols, num_bins=10):
    """
    변수별 10등급 분류 기준을 설정하는 함수
    
    Args:
        data (DataFrame): 입력 데이터
        outliers_selected (DataFrame): 이상치 판단 결과
        f_cols (list): 분석할 변수 리스트
        num_bins (int): 등급 개수 (기본값: 10)
        
    Returns:
        dict: 변수별 등급 분류 기준 (bin edges)
    """
    # 각 변수별로 처리
    bin_edges={}
    for col in f_cols:
        # 동일한 데이터 개수로 등급 분류 (quantile 기반)
        bin_edges_equal_count = pd.qcut(data[col], q=num_bins, duplicates='drop', retbins=True)[1]
        bin_edges_equal_count[0] = -np.inf  # 최소값을 무한대로 설정
        bin_edges_equal_count[-1] = np.inf  # 최대값을 무한대로 설정
        bin_edges[col] = bin_edges_equal_count
    
    return bin_edges

def get_rank_scores(data, f_cols, y_col):
    """
    등급별 점수를 생성하는 함수 (부도율이 높은 등급일수록 높은 점수 부여)
    
    Args:
        data (DataFrame): 학습 데이터
        f_cols (list): 분석할 변수 리스트
        y_col (str): 타겟 변수명
        
    Returns:
        dict: 변수별 등급 점수 딕셔너리
    """
    # 등급별 점수 생성 (등급의 개수에 따른 점수 생성) 
    result_dict = {}
    
    for i in [i+'_BIN' for i in f_cols]: 
        # 등급별 부도율 계산( 중복값도 counting 한다.)
        t = data.groupby(i).agg(count_data=("MONTH", 'count'), y_sum=(y_col, 'sum'))
        t['bkrc_ratio'] = t.y_sum / t.count_data
        result_dict[i] = {}

        # 부도율이 낮은 등급부터 높은 등급 순으로 점수 부여
        if t.iloc[0, -1] > t.iloc[-1, -1]:
            for idx, row in enumerate(t.iterrows()): 
                key = row[0]
                value = round(100/len(f_cols)) // len(t) * (idx + 1) 
                result_dict[i][key] = value
        
        # 부도율이 높은 등급부터 낮은 등급 순으로 점수 부여
        elif t.iloc[0, -1] < t.iloc[-1, -1]:
            for idx, row in enumerate(reversed(list(t.iterrows()))):
                key = row[0]
                value = round(100/len(f_cols)) // len(t) * (idx + 1)
                result_dict[i][key] = value     
        else: 
            raise ValueError("bkrc_ratio of the first and last rows are equal.")

    return result_dict

def assign_group(value, bins, labels):
    for i in range(len(bins)):
        if value <= bins[i]:
            return labels[i]
    return labels[-1]

def summarize_by_group(df, target_col, group_col='group', prefix='real'):
    """
    특정 그룹(group_col) 내 target_col의 분포(개수, 비율)를 계산합니다.
    """
    summary = (
        df.groupby([group_col, target_col])
          .size()
          .rename(f'new_dist_{prefix}_count')
          .reset_index()
    )
    # 그룹별 total 계산
    summary['total'] = summary.groupby(group_col)[f'new_dist_{prefix}_count'].transform('sum')
    # 비율(%) 계산
    summary[f'new_dist_{prefix}_ratio'] = (
        summary[f'new_dist_{prefix}_count'] / summary['total'] * 100
    )
    return summary
    