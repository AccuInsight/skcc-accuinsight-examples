import os

ROOT_DIR = os.getcwd()
DATA_DIR = os.path.join(ROOT_DIR, 'data')
PROCESS_DIR = os.path.join(os.path.dirname(ROOT_DIR), 'data', 'processed')

# 제품 분류 카테고리
product_main_code = {
    '#': '미정의',
    '1': '기초화장품',
    '2': '색조화장품',
    '3': '프래그런스',
    '4': '바디용품',
    '5': '헤어용품',
    '6': '건강/위생용품',
    '7': '건강식품',
    '8': '일반식품',
    '9': '미용소품',
    '10': '잡화',
    '80': '온라인_위수탁_전용',
    '90': '기타',
}

# 매장 식별 정보
store_code = {
    'S023': '온라인',
    'D428': '씨지브이여의도점',
    'D127': '오목교점',
    'D579': '홈플러스오산점',
    'D368': '연신내범서점',
    'D338': '관악로점',
    'D120': '선릉역점',
    'D176': '명동본점',
}


# EOF
