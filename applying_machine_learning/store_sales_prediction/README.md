# 매장의 상품 판매량 예측

# 개요
- 온/오프라인 매장에서 판매되는 상품의 판매량을 예측하는 모델을 머신러닝 및 딥러닝 알고리즘을 이용하여 개발하고 평가한다.
- 온라인 매장의 경우는 상품 판매량 단일 변수에 대해 단기간 미래를 예측하는 사례이다.

---
# 개발 환경
- python : 3.7
- tensorflow : 2.0.x

---
# 개발 과제

### 오프라인 매장

* [오프라인 상점 판매량 예측 모델](./offline_sales_prediction.ipynb)
  * 오프라인 샵에 대해 제품별 판매량 예측
  * Feature Engineering, ML 모델 활용 예측

### 온라인 쇼핑몰

* [온라인 쇼핑몰 제품 판매량 예측 (1)](./online_sales_prediction.ipynb)
  * 과거 sales 데이터를 이용 단기간 미래의 sales 데이터를 예측한다.
  * 기본적인 데이터를 탐색하고 RNN 기반의 여러 딥러닝 모델을 이용해 예측하고 성능을 비교하여 본다.

### 행사 판매대

* [디스플레이 매대 판매량 예측 모델](./display_stand_sales_prediction.ipynb)
  * TensorFlow 2.0 기반으로 구현한 RNN 기반의 행사 매대 판매량 예측 모델
  * 과거 30일(t-30, t-29, ..., t-1) 판매량 관련 데이터를 이용하여, 이후 3일(t, t+1, t+2) 판매량 예측 (4.1~4.5)
  * 과거 30일(t-30, t-29, ..., t-1) 판매량 관련 데이터를 이용하여, 다음 날(t) 판매량 예측 (4.6)

---
## 폴더 구조

```
└── project root dir
    ├── data : 데이터 저장소
    │   ├── raw : 모델 훈련 및 테스트에 사용할 파일 저장 폴더
    │   │   ├── train.csv
    │   └── prodcessed : 처리 중 생성되는 데이터 파일, 파일명은 실행하는 날짜
    │       ├── online_20210115132038.csv
    │       └── ...
    ├── display_stand : p매대 제품 판매량 예측 python 코드
    │   ├── core
    │   │   ├── data_prepare.py : 데이터 전처리 코드
    │   │   ├── AutoRegressionLSTM.py : Auto-Regression LSTM(seq2seq) 모델 코드
    │   │   ├── data_prepare_DARNN.py : DA-RNN 모델 생성을 위한 데이터 전처리 코드
    │   │   ├── Dual_stage_attention_model.py : DA-RNN 모델 코드
    │   │   └── train_DARNN.py : DA-RNN 학습 코드
    │   ├── Auto-regression LSTM.png
    │   └── DARNN.png
    ├── run : 코드 실행할 때 생성되는 파일
    ├── common.py : 공통 설정
    ├── display_stand_sales_prediction.ipynb : 행사 매대 판매량 예측 메인 노트북
    ├── offline_sales_prediction.ipynb : 오프라인 판매량 예측 메인 노트북
    └── online_sales_prediction.ipynb : 온라인 판매량 예측 메인 노트북
 ```
