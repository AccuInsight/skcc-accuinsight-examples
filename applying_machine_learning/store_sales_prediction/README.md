# 매장의 상품 판매량 예측

# 개요
- 온/오프라인 매장에서 판매되는 상품의 판매량을 예측하는 모델을 머신러닝 및 딥러닝 알고리즘을 이용하여 개발하고 평가한다.
- 온라인 매장의 경우는 상품 판매량 단일 변수에 대해 단기간 미래를 예측하는 사례이다.

---
# 개발 환경
### 과제 개발 당시 환경
- python : 3.7
- tensorflow : 2.0.x

---
# 개발 과제

### offline 매장

* [Offline 판매량 예측 모델](./store_offline_rebuild.ipynb)
  * 오프라인 샵에 대해 제품별 판매량 예측
  * Feature Engineering, ML 모델 활용 예측

### online 쇼핑몰

* [온라인 제품 판매량 예측 (1)](./store_online.ipynb)
  * 과거 sales 데이터를 이용 단기간 미래의 sales 데이터를 예측한다.
  * 기본적인 데이터를 탐색하고 RNN 기반의 여러 딥러닝 모델을 이용해 예측하고 성능을 비교하여 본다.

### P매대

* [P매 판매량 예측 모델](./store_pmd.ipynb)
  * TensorFlow 2.0 기반으로 구현한 RNN 기반의 P매대 판매량 예측 모델
  * 과거 30일(t-30, t-29, ..., t-1) 판매량 관련 데이터를 이용하여, 이후 3일(t, t+1, t+2) 판매량 예측 (4.1~4.5)
  * 과거 30일(t-30, t-29, ..., t-1) 판매량 관련 데이터를 이용하여, 다음 날(t) 판매량 예측 (4.6)
---
## 폴더 구조

```
└── root dir
    ├── data : 데이터 풀
    │   ├── raw
    │   │   ├── rossmann)_train.csv
    │   └── prodcessed : 처리 중 생성되는 데이터 파일, 파일명은 실행하는 날짜
    │       ├── online_20210115132038.csv
    │       └── ...
    ├── runs : 시행시 생성되는 파일 저장
    ├── store_offline : 오프라인 매장 제품 판매량 예측 python 코드
    │   ├── source
    │   │   ├── bidirectional.py
    │   │   └── data_prepare.py
    │   ├── inference.py
    │   └── train.py
    ├── store_online : 온라인 제품 판매량 예측 python 코드
    ├── store_pmd : p매대 제품 판매량 예측 python 코드
    │   ├── data
    │   │   ├── raw : pmd 관련 모델 생성 위한 raw data
    │   │   └── processed : 전처리 완료 파일 저장
    │   ├── source
    │   │   ├── data_prepare.py : 데이터 전처리 코드
    │   │   ├── AutoRegressionLSTM.py : Auto-Regression LSTM(seq2seq) 모델 코드
    │   │   ├── data_prepare_DARNN.py : DA-RNN 모델 생성을 위한 데이터 전처리 코드
    │   │   ├── Dual_stage_attention_model.py : DA-RNN 모델 코드
    │   │   └── train_DARNN.py : DA-RNN 학습 코드
    │   ├── Auto-regression LSTM.png
    │   └── DARNN.png
    ├── common.py : 공통 설정
    ├── store_offline.ipynb : 오프라인 판매량 예측 메인 노트북
    ├── store_online.ipynb : 온라인 판매량 예측 메인 노트북
    └── store_pmd.ipynb : P매대 판매량 예측 메인 노트북
 ```
