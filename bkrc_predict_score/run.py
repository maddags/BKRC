import gc
import pandas as pd
import numpy as np
import sys

from clickhouse_driver import connect

from src.scoring import get_rank_scores, bin_edges_define
from src.preprocessing import changeCdtStatus, dataSetting, scoringDataSetting
from src.db import save_to_clickhouse

def making_bin_edges(
    df : pd.DataFrame,
    f_cols : list[str]
) :
    """
        df : 2022년 데이터
        f_cols : 모델에 사용된 변수
    """
    target = 'isClosed'
    trainYm = 202201

    # 01 ~ 03월 데이터를 통해 등급 경계 구하기
    three_month_sample = df[(df['MONTH'] >= str(trainYm)) & (df['MONTH'] <= str(trainYm + 2))][["MONTH"] + f_cols + [target]].dropna()
    
    # train data에서 칼럼 등급 결정하기
    bin_edges = bin_edges_define(three_month_sample,f_cols, num_bins=round(100/len(f_cols)))
    
    return bin_edges

def making_bizscore(
    df: pd.DataFrame,
    f_cols : list[str],
    bin_edges : dict,
    year : str = '2022'
) :
    """
        df : 전처리 끝난 데이터
        f_cols : 모델에 사용된 변수
        bin_edges : 모델에 사용된 변수별 경계
        year : 등급을 매길 연도
    """

    target = 'isClosed'
    trainYm = int(year + '01')

    # 01 ~ 03월 데이터를 통해 등급 경계 구하기
    three_month_sample = df[(df['MONTH'] >= str(trainYm)) & (df['MONTH'] <= str(trainYm + 2))][["MONTH"] + f_cols + [target]].dropna().copy()
    
    # 경계에 따른 칼럼별 등급 만들기
    for col in f_cols:
        if col == 'COUNT_COM_RGNO' :
            three_month_sample[f'{col}_SCORE'] = three_month_sample[f'{col}'].astype(int)
        else :
            # 학습 데이터에 등급 분류 적용
            three_month_sample[f'{col}_SCORE'] = 1 + pd.cut(three_month_sample[col], bins=bin_edges[col], labels=False)
    
    # 등급별 점수를 dict 생성
    result_dict = get_rank_scores(data = three_month_sample, f_cols = f_cols, y_col = target)
    
    # 최종 점수 저장용 DataFrame 초기화
    test_scores = pd.DataFrame(columns=["COM_RGNO", "BZCD", "MONTH", "CRI_GRD","CRI_NEW",'isClosed']+[i+'_SCORE' for i in f_cols] + ["bizdata_score"])
    
    months = [f"{year}{month:02d}" for month in range(1, 13)]
    
    # 월별 점수 총합 구하기
    for j in months:
        sample = df[(df.MONTH == str(j)) & (df[target].isin([0,1]))]\
        [["COM_RGNO", "BZCD", "MONTH","CRI_GRD","CRI_NEW"] + f_cols + [target]].dropna() 
    
        tmp2 = sample.loc[:, ['COM_RGNO', 'BZCD', 'MONTH',"CRI_GRD","CRI_NEW",target]]
        
        for col in f_cols: 
            # 테스트 데이터에 학습 데이터 기준 등급 분류 적용
            tmp2[f'{col}_SCORE'] = 1 + pd.cut(sample[col], bins=bin_edges[col], labels=False)
    
        # 최종 점수 계산 (모든 변수 등급 점수 합계)
        ans = [0] * len(tmp2)
        
        for col in [i+'_SCORE' for i in f_cols]:
            ans = [a+b for a,b in zip(ans, [result_dict[col][j] for j in tmp2[col]])]
        
        tmp2['FINAL_SCORE'] = ans
    
        if test_scores.empty:
            test_scores = tmp2
        else :
            test_scores = pd.concat([test_scores, tmp2], ignore_index=True)

    return test_scores  

def loadBZIDX(year):
    # DB(데이터 가져올 DB 정보 입력)
    host = "172.16.220.34"
    port = "29000"
    database = "DA_DATA"
    user = "hhan"
    password = "hhan"

    conn = connect(host=host,
                   port=port,
                   database=database,
                   user=user,
                   password=password)

    cursor = conn.cursor()

    bf_one_year = str(int(year) - 1)
    startYm = int(str(bf_one_year) + '01')
    endYm = int(str(year) + '12')

    # clikcehouse여서 DT = '{}'에서 작은따옴표 필요
    sql = f"""
    WITH T1 AS (
        SELECT * FROM DA_DATA.ATCP_NPS WHERE DATE >= '{startYm}' AND DATE <= '{endYm}'
    ),
    T2 AS (
        SELECT RGNO, MIN(BZ_IDX) AS BZ_IDX 
        FROM DA_DATA.NPS_MASTER 
        WHERE DATE >= '{startYm}' AND DATE <= '{endYm}' AND LENGTH(RGNO) = 10 
        GROUP BY RGNO
    )
    SELECT T2.BZ_IDX AS BZ_IDX, T1.*FROM T1 LEFT JOIN T2 ON T1.RGNO = T2.RGNO
    """

    cursor.execute(sql)
    loadData = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]
    conn.close()

    npsMaster = pd.DataFrame(loadData, columns=columns)

    # KSIC_BZCD의 NaN을 '-'로 변경
    npsMaster['KSIC_BZCD'] = npsMaster['KSIC_BZCD'].fillna('-')
    npsMaster['KSIC_BZCD'] = npsMaster['KSIC_BZCD'].replace('', '-')

    return npsMaster


def main(year):
    print(f'\n[시작] {year}년 데이터 처리를 시작합니다...\n')
    
    # 현재 연도 데이터 호출
    print(f'[1/12] {year}년 데이터 로딩 중...')
    fpath = f'./data/newBkrcPensionInfo_atan{year}01_{year}12_12.parquet'
    present_df = pd.read_parquet(fpath)
    present_data = scoringDataSetting(changeCdtStatus(present_df))
    print(f'       ✓ {year}년 데이터 로드 완료')

    # 1년 전 데이터 호출
    print(f'\n[2/12] {bf_one_year}년 데이터 로딩 중...')
    period_one_bf_fpath = f'./data/newBkrcPensionInfo_atan{bf_one_year}01_{bf_one_year}12_12.parquet'
    period_one_bf_df = pd.read_parquet(period_one_bf_fpath)
    period_one_bf_data = scoringDataSetting(changeCdtStatus(period_one_bf_df))
    print(f'       ✓ {bf_one_year}년 데이터 로드 완료')

    # 2개 데이터 concat
    print(f'\n[3/12] 데이터 병합 중...')
    two_year_df = pd.concat([period_one_bf_df, present_df], axis=0, ignore_index=True)
    del period_one_bf_df, present_df  # 메모리 해제
    gc.collect()
    print(f'       ✓ 데이터 병합 및 메모리 해제 완료')

    # feature 설정
    f_cols = ['AVG_NMB_LST_YoY','COUNT_COM_RGNO' ,'COUNT_COM_RGNO_YoY', 'AVG_NMB_ACQ_YoY', 'AVG_EMP_FLUX_NET',
            'EMP_FLUX_VOL','AVG_NMB_SBS_MoM','AVG_AMT_RATE','동종업계_대비_고용규모','AVG_EMP_FLUX_NET_MoM']

    target = 'isClosed'

    # 점수 범위 생성
    print(f'\n[4/12] 점수 범위(bin_edges) 생성 중...')
    fpath_2022 = './data/newBkrcPensionInfo_atan202201_202212_12.parquet'
    data_2022 = scoringDataSetting(changeCdtStatus(pd.read_parquet(fpath_2022)))
    bin_edges = making_bin_edges(data_2022, f_cols)
    del data_2022  # 메모리 해제
    gc.collect()
    print(f'       ✓ 점수 범위(bin_edges) 생성 완료')

    # bizscore 데이터 생성
    print(f'\n[5/12] BizScore 데이터 생성 중...')
    past_year_biz_scores = making_bizscore(period_one_bf_data,f_cols,bin_edges,year=bf_one_year)
    present_year_biz_scores = making_bizscore(present_data,f_cols,bin_edges,year=year)
    del period_one_bf_data, present_data, bin_edges  # 메모리 해제
    gc.collect()
    print(f'       ✓ BizScore 데이터 생성 완료')

    print(f'\n[6/12] BizScore 데이터 병합 중...')
    biz_scores = pd.concat([past_year_biz_scores, present_year_biz_scores], axis=0, ignore_index=True)
    del past_year_biz_scores, present_year_biz_scores  # 메모리 해제
    gc.collect()
    print(f'       ✓ BizScore 데이터 병합 완료')

    biz_scores = biz_scores.rename(columns = {
        'COM_RGNO' : 'RGNO',
        'MONTH' : 'DATE',
        'BZCD' : 'KSIC_BZCD'
    })

    # BIZ_IDX 데이터 생성
    print(f'\n[7/12] BZ_IDX 데이터 로딩 및 병합 중...')
    bzidx_df = loadBZIDX(year)
    bzidx_df['DATE'] = bzidx_df['DATE'].astype(str)

    # # 결과 테이블 생성
    result = pd.merge(biz_scores, bzidx_df, on = ['RGNO','DATE','KSIC_BZCD'], how = 'inner')
    del bzidx_df  # 메모리 해제
    gc.collect()
    print(f'       ✓ BZ_IDX 데이터 병합 완료')

    # Q_tile, Rank 칼럼 생성
    print(f'\n[8/12] Q_tile 및 RANK 칼럼 생성 중...')
    result["Q_tile"] = (
        result
        .groupby(["DATE", "KSIC_BZCD"])["FINAL_SCORE"]
        .rank(method="average", ascending=True, pct=True)
    )

    result["RANK"] = (
        result
        .groupby(["DATE", "KSIC_BZCD"])["FINAL_SCORE"]
        .rank(method="min", ascending=False)
    ).astype(int)

    columns_order = ['BZ_IDX','DATE','RGNO','BZ_NM','APL_DT','NMB_SBS','NMB_ACQ','NMB_LST','AMT','EMP_FLUX_VOL','EMP_FLUX_NET','AMT_RATE',
                    'AVG_NMB_LST_YoY_SCORE','COUNT_COM_RGNO_SCORE', 'COUNT_COM_RGNO_YoY_SCORE','AVG_NMB_ACQ_YoY_SCORE', 'AVG_EMP_FLUX_NET_SCORE',
                    'EMP_FLUX_VOL_SCORE','AVG_NMB_SBS_MoM_SCORE', 'AVG_AMT_RATE_SCORE', '동종업계_대비_고용규모_SCORE',
                    'AVG_EMP_FLUX_NET_MoM_SCORE', 'FINAL_SCORE','Q_tile','RANK','SD_CD','SGG_CD', 'UMD_CD', 'BJD_CD', 'HJD_CD', 'ZIP_CD',
                    'JIBN_ADRS','RD_ADRS', 'KSIC_GEN', 'KSIC_BZCD', 'KSIC_LCLS_CD', 'KSIC_LCLS_NM','KSIC_MCLS_CD', 'KSIC_MCLS_NM',
                    'KSIC_SCLS_CD', 'KSIC_SCLS_NM','KSIC_DCLS_CD', 'KSIC_DCLS_NM', 'KSIC_DDCLS_CD', 'KSIC_DDCLS_NM','RGST_CD', 'RE_RG_DT',
                    'WTH_DT']

    result = result[columns_order]
    print(f'       ✓ Q_tile 및 RANK 칼럼 생성 완료')

    # FINAL SCORE의 6개월 ROC 평균 구하는 코드
    print(f'\n[9/12] SCORE_DESC 칼럼 생성 중...')
    cols = ['RGNO', 'DATE', 'FINAL_SCORE']
    result_2 = result[cols].copy()

    result_2['DATE'] = pd.to_datetime(result_2['DATE'], format='%Y%m')
    result_2 = result_2.sort_values(['RGNO','DATE'])
    result_2['FINAL_SCORE_DIFF'] = result_2.groupby('RGNO')['FINAL_SCORE'].diff()
    result_2['ROLLING6_MEAN_FINAL_SCORE'] = (
        result_2.groupby('RGNO')['FINAL_SCORE_DIFF'].
        rolling(window=6, min_periods=6).
        mean().
        reset_index(level=0, drop=True)
    )

    result_2['ROLLING6_STD_FINAL_SCORE'] = (
        result_2.groupby('RGNO')['FINAL_SCORE_DIFF'].
        rolling(window=6, min_periods=6).
        std().
        reset_index(level=0, drop=True)
    )

    # ROLLING6_MEAN_FINAL_SCORE 양수, 음수인거 구분 (NaN 처리 추가)
    result_2['option1'] = np.where(
        result_2['ROLLING6_MEAN_FINAL_SCORE'].isna(),
        None,
        np.where(result_2['ROLLING6_MEAN_FINAL_SCORE'] > 0, '양수', '음수')
    )
    # ROLLING6_STD_FINAL_SCORE 상위 50%인지 하위 50%인지 구분
    result_2['option2'] = np.where(
        result_2['ROLLING6_STD_FINAL_SCORE'].isna(),
        None,
        np.where(
            result_2.groupby('DATE')['ROLLING6_STD_FINAL_SCORE'].rank(pct=True) >= 0.5,
            '상위',
            '하위'
        )
)

    # SCORE_DESC 칼럼 생성
    conditions = [
        (result_2['option1'] == '양수') & (result_2['option2'] == '상위'),
        (result_2['option1'] == '양수') & (result_2['option2'] == '하위'),
        (result_2['option1'] == '음수') & (result_2['option2'] == '상위'),
        (result_2['option1'] == '음수') & (result_2['option2'] == '하위')
    ]

    choices = [
        '평가 결과를 참고 수준으로 확인하세요',
        '평가 결과가 안정적입니다',
        '평가 결과 판단에 각별한 주의가 필요합니다',
        '평가 결과 판단에 주의하세요'
    ]

    result_2['SCORE_DESC'] = np.select(conditions, choices, default=None)

    # 중간 컬럼 삭제 (더 이상 필요 없음)
    result_2 = result_2.drop(['FINAL_SCORE', 'FINAL_SCORE_DIFF', 'ROLLING6_MEAN_FINAL_SCORE', 
                          'ROLLING6_STD_FINAL_SCORE', 'option1', 'option2'], axis=1)

    # DATE 타입 확인 후 변환
    if result_2['DATE'].dtype != 'object':
        result_2['DATE'] = result_2['DATE'].dt.strftime('%Y%m')

    # 결과 DF과 Join하기
    result3 = pd.merge(result, result_2[result_2['DATE'] >= str(str(year) + '01')][['RGNO','DATE','SCORE_DESC']], on=['RGNO','DATE'], how='left').copy()

    # SCORE_DESC가 None과 NaN이면 모두 제거(데이터 수 부족으로 인해 NaN,None 생성됨)
    result3 = result3[
        (result3['SCORE_DESC'].notna()) & 
        (result3['SCORE_DESC'] != None)
    ]

    # result_2 삭제
    del result_2
    gc.collect()
    print(f'       ✓ SCORE_DESC 칼럼 생성 완료')

    # 최종 result만 남음
    # 최종 DATE의 year 부분이 year만 출력 
    print(f'\n[10/12] 최종 데이터 필터링 중...')
    result3 = result3[result3['DATE'].str[:4] == year]
    print(f'       ✓ 최종 데이터 필터링 완료 (레코드 수: {len(result3):,})')

    # BZ_IDX 컬럼을 Int64로 변경
    result3['BZ_IDX'] = result3['BZ_IDX'].astype('Int64')

    # 컬럼명 변경
    result3 = result3.rename(columns={
        '동종업계_대비_고용규모_SCORE': 'IDST_RELATIE_EMPLOYMENT_SIZE_SCORE'
    })

    # DATE를 Int32로 변환 (String '202301' → Int 202301)
    print(f'\n[11/12] 데이터 타입 변환 중...')
    result3['DATE'] = result3['DATE'].astype('int32')

    # Q_tile을 100분위수 Int32로 변환 (0.5 → 50)
    result3['Q_tile'] = (result3['Q_tile'] * 100).astype('int32')

    # 기타 정수형 컬럼 타입 변환
    result3['NMB_SBS'] = result3['NMB_SBS'].astype('int32')
    result3['NMB_ACQ'] = result3['NMB_ACQ'].astype('int16')
    result3['NMB_LST'] = result3['NMB_LST'].astype('int16')
    result3['AMT'] = result3['AMT'].astype('int64')

    # SCORE 컬럼들 Int32로 변환
    score_columns = [
        'AVG_NMB_LST_YoY_SCORE', 'COUNT_COM_RGNO_SCORE', 'COUNT_COM_RGNO_YoY_SCORE',
        'AVG_NMB_ACQ_YoY_SCORE', 'AVG_EMP_FLUX_NET_SCORE', 'EMP_FLUX_VOL_SCORE',
        'AVG_NMB_SBS_MoM_SCORE', 'AVG_AMT_RATE_SCORE', 'IDST_RELATIE_EMPLOYMENT_SIZE_SCORE',
        'AVG_EMP_FLUX_NET_MoM_SCORE', 'FINAL_SCORE', 'RANK'
    ]
    for col in score_columns:
        if col in result3.columns:
            result3[col] = result3[col].astype('int32')
    print(f'       ✓ 데이터 타입 변환 완료')

    # 최종 컬럼 순서 정렬
    final_columns = [
        'BZ_IDX', 'DATE', 'RGNO', 'BZ_NM', 'APL_DT', 
        'NMB_SBS', 'NMB_ACQ', 'NMB_LST', 'AMT',
        'AVG_NMB_LST_YoY_SCORE', 'COUNT_COM_RGNO_SCORE', 'COUNT_COM_RGNO_YoY_SCORE',
        'AVG_NMB_ACQ_YoY_SCORE', 'AVG_EMP_FLUX_NET_SCORE', 'EMP_FLUX_VOL_SCORE',
        'AVG_NMB_SBS_MoM_SCORE', 'AVG_AMT_RATE_SCORE', 'IDST_RELATIE_EMPLOYMENT_SIZE_SCORE',
        'AVG_EMP_FLUX_NET_MoM_SCORE', 'FINAL_SCORE', 'Q_tile', 'RANK',
        'SD_CD', 'SGG_CD', 'UMD_CD', 'BJD_CD', 'HJD_CD', 'ZIP_CD',
        'JIBN_ADRS', 'RD_ADRS', 
        'KSIC_GEN', 'KSIC_BZCD', 'KSIC_LCLS_CD', 'KSIC_LCLS_NM',
        'KSIC_MCLS_CD', 'KSIC_MCLS_NM', 'KSIC_SCLS_CD', 'KSIC_SCLS_NM',
        'KSIC_DCLS_CD', 'KSIC_DCLS_NM', 'KSIC_DDCLS_CD', 'KSIC_DDCLS_NM',
        'RGST_CD', 'RE_RG_DT', 'WTH_DT', 'SCORE_DESC'
    ]

    result3 = result3[final_columns]

    # result DataFrame을 ClickHouse에 저장
    print(f'\n[12/12] ClickHouse에 데이터 저장 중...')
    save_result = save_to_clickhouse(
        df=result3,
        table_name='ATCP_SCORE_DM',
        database='hhan_rgno',          # 기본값 (생략 가능)
        if_exists='append',        # 기본값: 기존 데이터에 추가 (생략 가능)
        batch_size=10000           # 기본값 (생략 가능)
    )

    # 저장 결과 확인
    print(save_result)
    print(f'\n[완료] {year}년 데이터 처리 및 저장이 완료되었습니다!\n')

    return print(f'Complete to save {year} file') 

if __name__ == '__main__' :
    year = sys.argv[1]
    bf_one_year = str(int(year) - 1)
    main(year)