import numpy as np
import pandas as pd

def changeCdtStatus(
    df: pd.DataFrame,
    y_prefix: str = "y_",
) -> pd.DataFrame:
    """
    - y_prefix로 시작하는 모든 칼럼에서 값이 7 또는 8이면 0으로 변경 (완전 벡터화)
    - y_prefix로 시작하는 값들의 합으로 영업인지 폐업인지 결정하는 칼럼 생성

    Parameters
    ----------
    df : pd.DataFrame
    y_prefix : str
        치환 대상이 되는 숫자 칼럼의 접두사 (기본: 'y_')

    Returns
    -------
    pd.DataFrame
        원본 df를 수정하여 반환
    """
    
    # 1) y_로 시작하는 모든 칼럼 일괄 치환 (7,8 -> 0)
    y_cols = [c for c in df.columns if c.startswith(y_prefix)]
    if y_cols:
        # 숫자가 문자열로 들어있어도 안전하게 처리 (가능하면 숫자로 변환)
        df[y_cols] = df[y_cols].apply(pd.to_numeric)

        # 벡터화 마스킹으로 한 번에 치환
        mask = df[y_cols].isin([7, 8])
        df[y_cols] = df[y_cols].where(~mask, 0)

    # 2) 선택된 칼럼들의 합을 계산해서 새로운 컬럼 생성
    ## 'y_'로 시작하고 숫자가 붙은 칼럼만 선택
    y_cols = [col for col in df.columns if col.startswith('y_') and col[2:].isdigit()]
    
    # 'y_'의 변수들 합으로 폐업 여부 결정
    df['isClosed'] = df[y_cols].sum(axis=1)
    df['isClosed'] = df['isClosed'].apply(lambda x : 1 if x > 0 else 0)
    
    return df

def dataSetting(
    df:pd.DataFrame,
    f_cols : list,
    target : str
)-> pd.DataFrame:
    """
    초기 데이터 세팅
    - 데이터 변수 선택
    df : 전처리 끝난 DataFrame
    feature_cols : 모델링에 쓰일 변수들
    """
    # target_col, ignore_col은 고정
    target_col = [target]
    ignore_col = ['COM_RGNO','MONTH','BZCD','CRI_NEW']

        
    # 데이터 기본 세팅(ML, Scoring 공통 부분)
    df['MONTH'] = df['MONTH'].astype(str)
    df['BZCD'].fillna('-',inplace = True)

    mask = (df['y_1'] != 9)
    df = df[mask][ignore_col+f_cols+target_col].copy()
    
    return df

def scoringDataSetting(
    df : pd.DataFrame
) :
    """
    df : 초기 데이터 세팅 끝난 데이터
    """
    
    df['MONTH'] = df['MONTH'].astype(str)
    df['BZCD'].fillna('-',inplace = True)

    mask = (df['CRI_NEW'].notnull()) & (df['y_1'] != 9)
    result = df[mask]

    return result