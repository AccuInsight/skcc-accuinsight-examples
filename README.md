# skcc-accuinsight-examples

이 리포지토리에는 [Accuinsight+ Modeler](http://gcp.accuinsight.net)에서 기계 학습 및 딥 러닝을 적용하는 방법을 보여주는 예제 노트북이 포함되어 있습니다.


## Examples

###  머신러닝 적용 소개

이 예제는 다양한 분야에서 적용되고 있는 머신 러닝의 실제 사용 사례 및 개념을 소개합니다.



### 모델 튜닝 자동화 

이 예제는 하이퍼 파라미터 튜닝 기능을 소개합니다. 
많은 수의 훈련 작업을 실행하여 어떤 하이퍼 파라미터 값이 가장 영향력이 큰지 결정함으로써 Accuinsight+에서 최상의 예측을 제공하는 데 도움이됩니다. 

 - [hyperparameter_optimization_algorithms](hyperparameter_tuning/hyperparameter_optimization_algorithms) : 파이썬 패키지 모델을 이용한 최근 하이퍼 파라미터 최적화 알고리즘에 대한 특징과 모델학습 정확도를 개선하는 방법을 보여줍니다.
 - [autodl_cnn_ecg_classification](hyperparameter_tuning/autodl_cnn_ecg_classification) : 심전도(ECG) 데이터를 분류하는 CNN 모델과 Accuinsight+ Modeler의 AutoDL 기능을 이용해 모델의 하이퍼 파라미터를 최적화하는 방법을 보여줍니다.
 - [autodl_cnn_review_text_classification](hyperparameter_tuning/autodl_cnn_review_text_classification) : 영화 리뷰 데이터를 분류하는 CNN 모델과 Accuinsight+ Modeler의 AutoDL 기능을 이용해 모델의 하이퍼 파라미터를 최적화하는 방법을 보여줍니다.
 - [autodl_mlp_user_classification](hyperparameter_tuning/autodl_mlp_user_classification) : 스팸 계정과 정상 계정을 분류하는 MLP 모델과 Accuinsight+ Modeler의 AutoDL 기능을 이용해 모델의 하이퍼 파라미터를 최적화하는 방법을 보여줍니다.
 - [autodl_rnn_temperature_regression](hyperparameter_tuning/autodl_rnn_temperature_regression) : 서울의 평균 기온을 예측하는 RNN 모델과 Accuinsight+ Modeler의 AutoDL 기능을 이용해 모델의 하이퍼 파라미터를 최적화하는 방법을 보여줍니다.



### Accuinsight+ Data Drift 분석 및 모니터링 

이 예제는 Accuinsight+를 사용하여 사전 구축한 노트북 워크스페이스에서 Data Drift를 설정하는 방법을 보여줍니다. 