import pandas as pd
from clickhouse_driver import connect, Client

def save_to_clickhouse(df,table_name, database='KTSPI', 
                       if_exists='append', batch_size=10000,
                       host="172.16.220.34", port="29000",
                       user="hhan", password="hhan"):
    """
    DataFrame을 ClickHouse 테이블에 저장
    
    Parameters:
    -----------
    df : pd.DataFrame
        저장할 데이터프레임
    table_name : str
        저장할 테이블 명
    database : str, optional
        데이터베이스 명 (기본값: 'KTSPI')
    if_exists : str, optional
        테이블이 존재할 때 동작 방식:
        - 'append': 데이터 추가 (기본값)
        - 'replace': 테이블 삭제 후 재생성
        - 'fail': 에러 발생
    batch_size : int, optional
        한 번에 insert할 행 수 (기본값: 10000)
    host : str, optional
        ClickHouse 호스트
    port : str, optional
        ClickHouse 포트
    user : str, optional
        사용자명
    password : str, optional
        비밀번호
    
    Returns:
    --------
    dict
        저장 결과 정보 딕셔너리
    
    Raises:
    -------
    ValueError
        if_exists가 유효하지 않은 값일 경우
    """
    
    # if_exists 유효성 검사
    if if_exists not in ['append', 'replace', 'fail']:
        raise ValueError("if_exists는 'append', 'replace', 'fail' 중 하나여야 합니다.")
    
    # ClickHouse 클라이언트 생성
    client = Client(
        host=host,
        port=int(port),
        database=database,
        user=user,
        password=password
    )
    
    try:
        # 테이블 존재 여부 확인
        table_exists_query = f"""
            SELECT count() 
            FROM system.tables
            WHERE database = '{database}' AND name = '{table_name}'
        """
        table_exists = client.execute(table_exists_query)[0][0] > 0
        
        # if_exists 처리
        if table_exists:
            if if_exists == 'fail':
                raise ValueError(f"테이블 '{table_name}'이 이미 존재합니다.")
            elif if_exists == 'replace':
                print(f"기존 테이블 '{table_name}' 삭제 중...")
                client.execute(f"DROP TABLE IF EXISTS {database}.{table_name}")
                table_exists = False
        
        # 테이블이 없으면 생성 (자동으로 DataFrame의 dtype 기반)
        if not table_exists:
            print(f"테이블 '{table_name}'을 생성해 주세요. ")
            raise ValueError(f"테이블 '{table_name}'이 존재하지 않습니다.")
            
        
        # 데이터 삽입
        print(f"데이터 삽입 중... (총 {len(df):,}행)")
        
        # DataFrame을 딕셔너리 리스트로 변환
        records = df.to_dict('records')
        
        # 컬럼명 리스트
        columns = df.columns.tolist()
        
        # batch 단위로 삽입
        total_inserted = 0
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            
            # INSERT 쿼리 실행
            insert_query = f"INSERT INTO {database}.{table_name} ({', '.join(columns)}) VALUES"
            client.execute(insert_query, batch)
            
            total_inserted += len(batch)
            print(f"  진행률: {total_inserted:,}/{len(df):,} ({total_inserted/len(df)*100:.1f}%)")
        
        # 삽입 확인
        count_query = f"SELECT count() FROM {database}.{table_name}"
        total_rows = client.execute(count_query)[0][0]
        
        print(f"\n✓ ClickHouse 저장 완료!")
        print(f"  - 데이터베이스: {database}")
        print(f"  - 테이블: {table_name}")
        print(f"  - 삽입된 행 수: {total_inserted:,}")
        print(f"  - 테이블 전체 행 수: {total_rows:,}")
        print(f"  - 열 수: {len(df.columns)}")
        
        return {
            'database': database,
            'table': table_name,
            'inserted_rows': total_inserted,
            'total_rows': total_rows,
            'columns': len(df.columns)
        }
        
    except Exception as e:
        print(f"✗ ClickHouse 저장 중 오류 발생: {e}")
        raise
    finally:
        client.disconnect()