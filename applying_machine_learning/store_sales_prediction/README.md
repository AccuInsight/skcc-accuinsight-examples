# 매장의 상품 판매량 예측

# 개요
- 온/오프라인 매장에서 판매되는 상품의 판매량 데이터를 예측하기 위해 데이터를 분석하고 예측 모델을 구현하는 시계열 데이터 예측 모델을 구현해 본다.
- 단일 변수 혹은 다중 변수를 활용하며 Machine Learning 및 Deep Learning 알고리즘을 이용해 미래 판매량을 예측하고 모델들의 성능을 비교해 본다.

---
# 개발 환경
- python : 3.7
- tensorflow : 2.0.x

---
# 개발 과제

### [다중 변수 활용 ML기반 판매량 예측](./multi_variable_sales_prediction_ml.ipynb)
  * 판매량 포함 요일, 프로모션 등 다중 변수를 활용해 가까운 미래의 판매량을 예측한다.
  * Feature Engineering을 통해 변수의 특성을 이해하고, ML 모델 활용해 예측을 수행하고 성능을 확인한다.

### [다중 변수 활용 DL기반 판매량 예측](./multi_variable_sales_prediction_dl.ipynb)
  * 판매량 포함 요일, 프로모션 등 다중 변수를 활용해 가까운 미래의 판매량을 예측한다.
  * 과거 30일(t-30, t-29, ..., t-1) 판매량 관련 데이터를 이용하여, 이후 3일(t, t+1, t+2) 판매량 예측 (4.1~4.5)
  * 과거 30일(t-30, t-29, ..., t-1) 판매량 관련 데이터를 이용하여, 다음 날(t) 판매량 예측 (4.6)

### [단일 변수 활용 DL기반 판매량 예측](./single_variable_sales_prediction_dl.ipynb)
  * 과거 sales 데이터를 이용 단기간 미래의 sales 데이터를 예측한다.
  * 기본적인 데이터를 탐색하고 RNN 기반의 여러 딥러닝 모델을 이용해 예측하고 성능을 비교하여 본다.
  
---
## 폴더 구조

```
└── project root dir
    ├── data : 데이터 저장소
    │   ├── raw : 모델 훈련 및 테스트에 사용할 파일 저장 폴더
    │   │   ├── train.csv
    │   ├── prodcessed : 처리 중 생성되는 데이터 파일, 파일명은 실행하는 날짜
    │   │   ├── online_20210115132038.csv
    │   │   └── ...
    │   ├── core : multi_variable_sales_prediction_dl에서 사용하는 모델 생성 및 관리를 위한 python module
    │   │   ├── data_prepare.py : 데이터 전처리 코드
    │   │   ├── AutoRegressionLSTM.py : Auto-Regression LSTM(seq2seq) 모델 코드
    │   │   ├── data_prepare_DARNN.py : DA-RNN 모델 생성을 위한 데이터 전처리 코드
    │   │   ├── Dual_stage_attention_model.py : DA-RNN 모델 코드
    │   │   └── train_DARNN.py : DA-RNN 학습 코드
    ├── pic
    │   └── DARNN.png
    ├── run : 코드 실행할 때 생성되는 파일
    ├── multi_variable_sales_prediction_dl.ipynb : 다중 변수 활용 DL 기반 판매량 예측 Jupyter Notebook 파일
    ├── multi_variable_sales_prediction_ml.ipynb : 다중 변수 활용 ML 기반 판매량 예측 Jupyter Notebook 파일
    └── single_variable_sales_prediction_dl.ipynb : 단일 변수 활용 DL 기반 판매량 예측 Jupyter Notebook 파일
 ```
