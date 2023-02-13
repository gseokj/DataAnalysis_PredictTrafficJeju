#  :oncoming_automobile: Jeju Road Traffic Prediction with ML :sunrise_over_mountains:
[제주 도로 교통량 예측 경진대회](https://dacon.io/competitions/official/235985/overview/description)

![image](https://user-images.githubusercontent.com/84713532/206641256-137974fc-f097-4c6f-843c-d23976861409.png)

## :bookmark_tabs: Mini Project (2022/12/06 ~ 2022/12/09) :date:

> :family: 팀명: 차탄갑서 
- [고석주](https://github.com/SeokJuGo)
- [지우근](https://github.com/UGeunJi)
- [이재영](https://github.com/JAYJAY1005)
- [양효준](https://github.com/HyoJoon-Yang)

---

# :scroll: 프로젝트에 대한 전반적인 설명

### 주제 : <머신러능 성능 올리기>

#### 1. 문제 정의
```
- 문제 유형 (수치 예측, 분류)
- 성능 평가 지표 (수치 예측  : MSE, RMSE, MAE, RMSLE,  분류 : 정확도, f1 score, roc_auc )
```
#### 2. 탐색적 데이터 분석
```
2. 1 데이터 훑어보기
  - 데이터 양 (샘플수, 특성 수...)
  - 특성 이해
  - 타깃값 이해 (<--- 예측해야 하는 값)
  - 데이터세트(훈련/테스트) 분리
    (1) 합쳐서 전체를 전처리 한 후에 데이터 세트 분리 (bike sharing)
    (2) 훈련데이터 전처리한 후에, 동일한 과정을 테스트 세트에 적용 (california housing)
```
```
2.2 데이터 시각화
    (1) 수치형 데이터
       - 히스토그램 : 빠르게 데이터의 분포를 파악
                    : 상한~하한, 많이 분포한 데이터의 위치
                         : 스케일 여부
                         : 왜곡도 여부(꼬리가 긴 분포는 이후에 정규분포 형태로 로그 변환)

    (2) 범주형 데이터 : 범주형 데이터에 따른 수치값을 확인
       - 바플롯 : 타깃값과의 관계 확인
          ( 예 : 성별이 생존에 어떤 영향을 미치는지, 성별이 중요한특성인지, 예측력이 있는지..)
          ( 예 : Pclass가 생존에 어떤 영향을 미치는지)
          ( 참고 : Age와 같이 수치형 데이터의 예측력을 알고 싶다면 범주화(pd.cut)해서 시각화 해볼수 있음)
       - 박스 플롯 : 데이터의 사분위 분포, 이상치 검출
        
    (3) 상관관계 파악
       - 산점도
       - 상관계수
```

#### 3. 전처리 

```
   : 탐색적 데이터 분석을 통해 전처리 전략 적용
   : pandas 사용, 또는 scikit-learn pipeline 사용 가능
```
```
3.1 특성 선택 / 특성 삭제 / 특성 조합
```
```
3.2 인코딩
    (1) 레이블 인코딩
    (2) 원핫인코딩
```
```
3.3 스케일링
    (1) MinMax 스케일링(정규화) : 0~1사이로
    (2) Standardized 스케일링(표준화) : 평균 0, 표준편차 1
    (3) 로그 스케일링 : 왜곡된 데이터를 정규 분포 형태로 (특성, 타깃 모두 적용 가능)
```
#### 4. 베이스라인 모델
```
4. 1 교차검증을 통한 모델 간 성능 비교
    : 교차 검증 = 훈련 + 검증을 교번하며 진행
    : 성능 측정 지표 설정해줘야 함

   (1) 수치 예측
       선형 회귀 (회귀 계수 : 특성의 영향력)
       트리 모델 (특성의 중요도)
       앙상블 모델
       서포트 벡터 머신
       kNN

   (2) 분류
      로지스틱 회귀 : 이진분류 (회귀 계수 : 특성의 영향력)
      소프트맥스 회귀 : 다중 분류 (회귀 계수 : 특성의 영향력)
      트리 모델 (특성의 중요도)
      앙상블 모델
      서포트 벡터 머신
      kNN
```
#### 5. 성능 올리기
```
5.1 데이터 측면
    탐색적 데이터 분석을 통해 나온 결과를 다양하게 실험(적용)
     : 결측치처리, 이상치 제거
     : 특성 추가, 삭제, 파생 특성 추가
     : 특성 스케일링, 특성 인코딩
```
```
5.2 모델 측면
     : 앙상블 모델 강력
     : 하이퍼파라미터 튜닝 : 그리드 탐색, 랜덤 탐색
```
#### 6. 최종 예측과 성능 평가
```
   테스트 데이터로 예측
   : 테스트 데이터는 훈련데이터의 전처리된 형식과 동일해야함
   (1) 테스트 데이터에 대한 정답이 있으면 바로 평가 가능
   (2) 경진대회용 테스트 데이터의 경우 예측 결과를 제출함으로 평가를 받을 수 있음
```

---

# :computer: 실행 코드

```python
import random
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
%matplotlib inline
import seaborn as sns
import warnings
import plotly.express as px
import gc
import folium
from tqdm import tqdm
from math import radians, cos, sin, asin, sqrt
from sklearn.model_selection import train_test_split
```

# csv to parquet 데이터 불러오기

```python
def csv_to_parquet(csv_path, save_name):
    df = pd.read_csv(csv_path)
    df.to_parquet(f'./Dataset/{save_name}.parquet')
    del df
    gc.collect()
    print(save_name, 'Done.')
```

```python
csv_to_parquet('./Dataset/train.csv', 'train')
csv_to_parquet('./Dataset/test.csv', 'test')
```

```
train Done.
test Done.
```

```python
train = pd.read_parquet('./Dataset/train.parquet')
test = pd.read_parquet('./Dataset/test.parquet')
data_info = pd.read_csv('./Dataset/data_info.csv')
```

# 자료형 변환

```python
train.info()
```

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 4701217 entries, 0 to 4701216
Data columns (total 23 columns):
 #   Column                 Dtype  
---  ------                 -----  
 0   id                     object 
 1   base_date              int64  
 2   day_of_week            object 
 3   base_hour              int64  
 4   lane_count             int64  
 5   road_rating            int64  
 6   road_name              object 
 7   multi_linked           int64  
 8   connect_code           int64  
 9   maximum_speed_limit    float64
 10  vehicle_restricted     float64
 11  weight_restricted      float64
 12  height_restricted      float64
 13  road_type              int64  
 14  start_node_name        object 
 15  start_latitude         float64
 16  start_longitude        float64
 17  start_turn_restricted  object 
 18  end_node_name          object 
 19  end_latitude           float64
 20  end_longitude          float64
 21  end_turn_restricted    object 
 22  target                 float64
dtypes: float64(9), int64(7), object(7)
memory usage: 825.0+ MB
```

```python
# 빠른 탐색을 위한 자료형 변환
to_int32 = ["base_date", "base_hour", "lane_count", "road_rating", "multi_linked", "connect_code", "road_type"]
to_float32 = ["vehicle_restricted", "height_restricted", "maximum_speed_limit", "weight_restricted", "target"]

for i in to_int32:
    train[i] = train[i].astype("int32")
for j in to_float32:
    train[j] = train[j].astype("float32")
```

```python
train.info()
```

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 4701217 entries, 0 to 4701216
Data columns (total 23 columns):
 #   Column                 Dtype  
---  ------                 -----  
 0   id                     object 
 1   base_date              int32  
 2   day_of_week            object 
 3   base_hour              int32  
 4   lane_count             int32  
 5   road_rating            int32  
 6   road_name              object 
 7   multi_linked           int32  
 8   connect_code           int32  
 9   maximum_speed_limit    float32
 10  vehicle_restricted     float32
 11  weight_restricted      float32
 12  height_restricted      float32
 13  road_type              int32  
 14  start_node_name        object 
 15  start_latitude         float64
 16  start_longitude        float64
 17  start_turn_restricted  object 
 18  end_node_name          object 
 19  end_latitude           float64
 20  end_longitude          float64
 21  end_turn_restricted    object 
 22  target                 float32
dtypes: float32(5), float64(4), int32(7), object(7)
memory usage: 609.7+ MB
```

- memory 사용량: 825.0+ MB -> 609.7+ MB

```python
train.columns
```

```
Index(['id', 'base_date', 'day_of_week', 'base_hour', 'lane_count',
       'road_rating', 'road_name', 'multi_linked', 'connect_code',
       'maximum_speed_limit', 'vehicle_restricted', 'weight_restricted',
       'height_restricted', 'road_type', 'start_node_name', 'start_latitude',
       'start_longitude', 'start_turn_restricted', 'end_node_name',
       'end_latitude', 'end_longitude', 'end_turn_restricted', 'target'],
      dtype='object')
```

```python
train.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>base_date</th>
      <th>day_of_week</th>
      <th>base_hour</th>
      <th>lane_count</th>
      <th>road_rating</th>
      <th>road_name</th>
      <th>multi_linked</th>
      <th>connect_code</th>
      <th>maximum_speed_limit</th>
      <th>...</th>
      <th>road_type</th>
      <th>start_node_name</th>
      <th>start_latitude</th>
      <th>start_longitude</th>
      <th>start_turn_restricted</th>
      <th>end_node_name</th>
      <th>end_latitude</th>
      <th>end_longitude</th>
      <th>end_turn_restricted</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>TRAIN_0000000</td>
      <td>20220623</td>
      <td>목</td>
      <td>17</td>
      <td>1</td>
      <td>106</td>
      <td>지방도1112호선</td>
      <td>0</td>
      <td>0</td>
      <td>60.0</td>
      <td>...</td>
      <td>3</td>
      <td>제3교래교</td>
      <td>33.427747</td>
      <td>126.662612</td>
      <td>없음</td>
      <td>제3교래교</td>
      <td>33.427749</td>
      <td>126.662335</td>
      <td>없음</td>
      <td>52.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>TRAIN_0000001</td>
      <td>20220728</td>
      <td>목</td>
      <td>21</td>
      <td>2</td>
      <td>103</td>
      <td>일반국도11호선</td>
      <td>0</td>
      <td>0</td>
      <td>60.0</td>
      <td>...</td>
      <td>0</td>
      <td>광양사거리</td>
      <td>33.500730</td>
      <td>126.529107</td>
      <td>있음</td>
      <td>KAL사거리</td>
      <td>33.504811</td>
      <td>126.526240</td>
      <td>없음</td>
      <td>30.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>TRAIN_0000002</td>
      <td>20211010</td>
      <td>일</td>
      <td>7</td>
      <td>2</td>
      <td>103</td>
      <td>일반국도16호선</td>
      <td>0</td>
      <td>0</td>
      <td>80.0</td>
      <td>...</td>
      <td>0</td>
      <td>창고천교</td>
      <td>33.279145</td>
      <td>126.368598</td>
      <td>없음</td>
      <td>상창육교</td>
      <td>33.280072</td>
      <td>126.362147</td>
      <td>없음</td>
      <td>61.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>TRAIN_0000003</td>
      <td>20220311</td>
      <td>금</td>
      <td>13</td>
      <td>2</td>
      <td>107</td>
      <td>태평로</td>
      <td>0</td>
      <td>0</td>
      <td>50.0</td>
      <td>...</td>
      <td>0</td>
      <td>남양리조트</td>
      <td>33.246081</td>
      <td>126.567204</td>
      <td>없음</td>
      <td>서현주택</td>
      <td>33.245565</td>
      <td>126.566228</td>
      <td>없음</td>
      <td>20.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>TRAIN_0000004</td>
      <td>20211005</td>
      <td>화</td>
      <td>8</td>
      <td>2</td>
      <td>103</td>
      <td>일반국도12호선</td>
      <td>0</td>
      <td>0</td>
      <td>80.0</td>
      <td>...</td>
      <td>0</td>
      <td>애월샷시</td>
      <td>33.462214</td>
      <td>126.326551</td>
      <td>없음</td>
      <td>애월입구</td>
      <td>33.462677</td>
      <td>126.330152</td>
      <td>없음</td>
      <td>38.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 23 columns</p>
</div>

## 1. EDA

```python
data_info
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }


</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>변수명</th>
      <th>변수 설명</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>id</td>
      <td>아이디</td>
    </tr>
    <tr>
      <th>1</th>
      <td>base_date</td>
      <td>날짜</td>
    </tr>
    <tr>
      <th>2</th>
      <td>day_of_week</td>
      <td>요일</td>
    </tr>
    <tr>
      <th>3</th>
      <td>base_hour</td>
      <td>시간대</td>
    </tr>
    <tr>
      <th>4</th>
      <td>road_in_use</td>
      <td>도로사용여부</td>
    </tr>
    <tr>
      <th>5</th>
      <td>lane_count</td>
      <td>차로수</td>
    </tr>
    <tr>
      <th>6</th>
      <td>road_rating</td>
      <td>도로등급</td>
    </tr>
    <tr>
      <th>7</th>
      <td>multi_linked</td>
      <td>중용구간 여부</td>
    </tr>
    <tr>
      <th>8</th>
      <td>connect_code</td>
      <td>연결로 코드</td>
    </tr>
    <tr>
      <th>9</th>
      <td>maximum_speed_limit</td>
      <td>최고속도제한</td>
    </tr>
    <tr>
      <th>10</th>
      <td>weight_restricted</td>
      <td>통과제한하중</td>
    </tr>
    <tr>
      <th>11</th>
      <td>height_restricted</td>
      <td>통과제한높이</td>
    </tr>
    <tr>
      <th>12</th>
      <td>road_type</td>
      <td>도로유형</td>
    </tr>
    <tr>
      <th>13</th>
      <td>start_latitude</td>
      <td>시작지점의 위도</td>
    </tr>
    <tr>
      <th>14</th>
      <td>start_longitude</td>
      <td>시작지점의 경도</td>
    </tr>
    <tr>
      <th>15</th>
      <td>start_turn_restricted</td>
      <td>시작 지점의 회전제한 유무</td>
    </tr>
    <tr>
      <th>16</th>
      <td>end_latitude</td>
      <td>도착지점의 위도</td>
    </tr>
    <tr>
      <th>17</th>
      <td>end_longitude</td>
      <td>도착지점의 경도</td>
    </tr>
    <tr>
      <th>18</th>
      <td>end_turn_restricted</td>
      <td>도작지점의 회전제한 유무</td>
    </tr>
    <tr>
      <th>19</th>
      <td>road_name</td>
      <td>도로명</td>
    </tr>
    <tr>
      <th>20</th>
      <td>start_node_name</td>
      <td>시작지점명</td>
    </tr>
    <tr>
      <th>21</th>
      <td>end_node_name</td>
      <td>도착지점명</td>
    </tr>
    <tr>
      <th>22</th>
      <td>vehicle_restricted</td>
      <td>통과제한차량</td>
    </tr>
    <tr>
      <th>23</th>
      <td>target</td>
      <td>평균속도(km)</td>
    </tr>
  </tbody>
</table>
</div>

```python
len(train)
```

```
4701217
```

```python
train.head().T
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }


</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>id</th>
      <td>TRAIN_0000000</td>
      <td>TRAIN_0000001</td>
      <td>TRAIN_0000002</td>
      <td>TRAIN_0000003</td>
      <td>TRAIN_0000004</td>
    </tr>
    <tr>
      <th>base_date</th>
      <td>20220623</td>
      <td>20220728</td>
      <td>20211010</td>
      <td>20220311</td>
      <td>20211005</td>
    </tr>
    <tr>
      <th>day_of_week</th>
      <td>목</td>
      <td>목</td>
      <td>일</td>
      <td>금</td>
      <td>화</td>
    </tr>
    <tr>
      <th>base_hour</th>
      <td>17</td>
      <td>21</td>
      <td>7</td>
      <td>13</td>
      <td>8</td>
    </tr>
    <tr>
      <th>lane_count</th>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>road_rating</th>
      <td>106</td>
      <td>103</td>
      <td>103</td>
      <td>107</td>
      <td>103</td>
    </tr>
    <tr>
      <th>road_name</th>
      <td>지방도1112호선</td>
      <td>일반국도11호선</td>
      <td>일반국도16호선</td>
      <td>태평로</td>
      <td>일반국도12호선</td>
    </tr>
    <tr>
      <th>multi_linked</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>connect_code</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>maximum_speed_limit</th>
      <td>60</td>
      <td>60</td>
      <td>80</td>
      <td>50</td>
      <td>80</td>
    </tr>
    <tr>
      <th>vehicle_restricted</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>weight_restricted</th>
      <td>32400</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>height_restricted</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>road_type</th>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>start_node_name</th>
      <td>제3교래교</td>
      <td>광양사거리</td>
      <td>창고천교</td>
      <td>남양리조트</td>
      <td>애월샷시</td>
    </tr>
    <tr>
      <th>start_latitude</th>
      <td>33.4277</td>
      <td>33.5007</td>
      <td>33.2791</td>
      <td>33.2461</td>
      <td>33.4622</td>
    </tr>
    <tr>
      <th>start_longitude</th>
      <td>126.663</td>
      <td>126.529</td>
      <td>126.369</td>
      <td>126.567</td>
      <td>126.327</td>
    </tr>
    <tr>
      <th>start_turn_restricted</th>
      <td>없음</td>
      <td>있음</td>
      <td>없음</td>
      <td>없음</td>
      <td>없음</td>
    </tr>
    <tr>
      <th>end_node_name</th>
      <td>제3교래교</td>
      <td>KAL사거리</td>
      <td>상창육교</td>
      <td>서현주택</td>
      <td>애월입구</td>
    </tr>
    <tr>
      <th>end_latitude</th>
      <td>33.4277</td>
      <td>33.5048</td>
      <td>33.2801</td>
      <td>33.2456</td>
      <td>33.4627</td>
    </tr>
    <tr>
      <th>end_longitude</th>
      <td>126.662</td>
      <td>126.526</td>
      <td>126.362</td>
      <td>126.566</td>
      <td>126.33</td>
    </tr>
    <tr>
      <th>end_turn_restricted</th>
      <td>없음</td>
      <td>없음</td>
      <td>없음</td>
      <td>없음</td>
      <td>없음</td>
    </tr>
    <tr>
      <th>target</th>
      <td>52</td>
      <td>30</td>
      <td>61</td>
      <td>20</td>
      <td>38</td>
    </tr>
  </tbody>
</table>
</div>

- 수치형 데이터: 위도, 경도, 평균 속도 (위도, 경도도 범주형 데이터처럼 사용)
- 나머지는 모두 범주형 데이터

## 컬럼에 관한 정보

```python
train['road_rating']
```

```
0          106
1          103
2          103
3          107
4          103
          ... 
4701212    107
4701213    107
4701214    103
4701215    103
4701216    107
Name: road_rating, Length: 4701217, dtype: int32
```

![image](https://user-images.githubusercontent.com/84713532/206893593-3c0f8994-0152-41dd-a0d4-e34b515f30a3.png)


```python
train['connect_code']
```

```
0          0
1          0
2          0
3          0
4          0
          ..
4701212    0
4701213    0
4701214    0
4701215    0
4701216    0
Name: connect_code, Length: 4701217, dtype: int32
```

```python
train['connect_code'].unique()
```

```
array([  0, 103])
```

```python
train['connect_code'].value_counts()
```



```
0      4689075
103      12142
Name: connect_code, dtype: int64
```

![image](https://user-images.githubusercontent.com/84713532/206893581-d8a554d6-4d48-4925-93d1-1f7d965f428e.png)

- 값의 불균형 발견

```python
train.groupby(['base_date'])['target'].size()
```

```
base_date
20210901    19722
20210902    18809
20210903    19880
20210904    17998
20210905    17836
            ...  
20220727     9195
20220728     7601
20220729     5138
20220730     1845
20220731     5539
Name: target, Length: 281, dtype: int64
```

- 기간은 2021년 9월 1일부터 2022년 7월 31일까지인 것을 확인

```python
train.hist(bins=50, figsize=(20, 15))
```

```
array([[<AxesSubplot:title={'center':'base_date'}>,
        <AxesSubplot:title={'center':'base_hour'}>,
        <AxesSubplot:title={'center':'lane_count'}>,
        <AxesSubplot:title={'center':'road_rating'}>],
       [<AxesSubplot:title={'center':'multi_linked'}>,
        <AxesSubplot:title={'center':'connect_code'}>,
        <AxesSubplot:title={'center':'maximum_speed_limit'}>,
        <AxesSubplot:title={'center':'vehicle_restricted'}>],
       [<AxesSubplot:title={'center':'weight_restricted'}>,
        <AxesSubplot:title={'center':'height_restricted'}>,
        <AxesSubplot:title={'center':'road_type'}>,
        <AxesSubplot:title={'center':'start_latitude'}>],
       [<AxesSubplot:title={'center':'start_longitude'}>,
        <AxesSubplot:title={'center':'end_latitude'}>,
        <AxesSubplot:title={'center':'end_longitude'}>,
        <AxesSubplot:title={'center':'target'}>]], dtype=object)
```

![image](https://user-images.githubusercontent.com/84713532/206893521-68c06ee3-5c58-4d3c-b7fd-ebff9afd8b07.png)


#### 값이 유일한 컬럼 탐색

```python
train_desc = train.describe().transpose()
train_desc[train_desc['std'] == 0].index
```

```
Index(['vehicle_restricted', 'height_restricted'], dtype='object')
```

- 'vehicle_restricted', 'height_restricted' 컬럼은 삭제하기로 함

#### 컬럼별 유니크값 확인

```python
column_names = train.columns.values.tolist()

for i in column_names:
    print(f'{i} = {train[i].nunique()}')
```

```
id = 4701217
base_date = 281
day_of_week = 7
base_hour = 24
lane_count = 3
road_rating = 3
road_name = 61
multi_linked = 2
connect_code = 2
maximum_speed_limit = 6
vehicle_restricted = 1
weight_restricted = 4
height_restricted = 1
road_type = 2
start_node_name = 487
start_latitude = 586
start_longitude = 586
start_turn_restricted = 2
end_node_name = 487
end_latitude = 586
end_longitude = 586
end_turn_restricted = 2
target = 102
```

#### 상관관계 확인

```python
train_corr = train.corr()
f, ax = plt.subplots(figsize=(13, 10))
sns.heatmap(train_corr, annot=True, fmt = '.2f', square=True)
```

```
<AxesSubplot:>
```

![image](https://user-images.githubusercontent.com/84713532/206893508-d19f7a83-4a33-4416-a735-f8749f9cb25f.png)


```python
# target 0.2이상 관계
corr = train.corr()
corr[(corr['target']>0.2)]
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>base_hour</th>
      <th>lane_count</th>
      <th>road_rating</th>
      <th>multi_linked</th>
      <th>connect_code</th>
      <th>maximum_speed_limit</th>
      <th>vehicle_restricted</th>
      <th>weight_restricted</th>
      <th>height_restricted</th>
      <th>road_type</th>
      <th>start_latitude</th>
      <th>start_longitude</th>
      <th>end_latitude</th>
      <th>end_longitude</th>
      <th>target</th>
      <th>year</th>
      <th>month</th>
      <th>day</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>maximum_speed_limit</th>
      <td>-0.036756</td>
      <td>0.384002</td>
      <td>-0.327474</td>
      <td>-0.020245</td>
      <td>-0.015190</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>0.085080</td>
      <td>NaN</td>
      <td>0.059511</td>
      <td>0.253147</td>
      <td>-0.033018</td>
      <td>0.252958</td>
      <td>-0.032907</td>
      <td>0.425715</td>
      <td>-0.017561</td>
      <td>0.001396</td>
      <td>-0.001759</td>
    </tr>
    <tr>
      <th>weight_restricted</th>
      <td>-0.003231</td>
      <td>-0.177224</td>
      <td>-0.118630</td>
      <td>-0.008790</td>
      <td>-0.020491</td>
      <td>0.085080</td>
      <td>NaN</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>0.792803</td>
      <td>-0.128291</td>
      <td>0.034926</td>
      <td>-0.128305</td>
      <td>0.034915</td>
      <td>0.294092</td>
      <td>-0.010346</td>
      <td>0.000740</td>
      <td>-0.000207</td>
    </tr>
    <tr>
      <th>road_type</th>
      <td>-0.007880</td>
      <td>-0.050715</td>
      <td>-0.125618</td>
      <td>0.042977</td>
      <td>-0.025846</td>
      <td>0.059511</td>
      <td>NaN</td>
      <td>0.792803</td>
      <td>NaN</td>
      <td>1.000000</td>
      <td>-0.043420</td>
      <td>0.033684</td>
      <td>-0.043430</td>
      <td>0.033664</td>
      <td>0.200840</td>
      <td>-0.004586</td>
      <td>0.003837</td>
      <td>-0.000432</td>
    </tr>
    <tr>
      <th>target</th>
      <td>-0.159407</td>
      <td>-0.144256</td>
      <td>-0.261693</td>
      <td>-0.008408</td>
      <td>0.048348</td>
      <td>0.425715</td>
      <td>NaN</td>
      <td>0.294092</td>
      <td>NaN</td>
      <td>0.200840</td>
      <td>0.036280</td>
      <td>-0.001168</td>
      <td>0.036139</td>
      <td>-0.001000</td>
      <td>1.000000</td>
      <td>-0.031676</td>
      <td>-0.000225</td>
      <td>-0.011605</td>
    </tr>
  </tbody>
</table>
</div>

- 'target'과 뚜렷한 상관관계를 가지고 있는 컬럼의 개수가 적음
- 상관관계가 뚜렷한 데이터부터 꼼꼼히 살펴보며 전처리를 진행해야겠음
- 파생변수 생성과 여러 인코딩 방식을 적극 활용하기로 결정

```python
test_corr = test.corr()
f, ax = plt.subplots(figsize=(13, 10))
sns.heatmap(test_corr, annot=True, fmt = '.2f', square=True)
```

```
<AxesSubplot:>
```

![image](https://user-images.githubusercontent.com/84713532/206893483-505f41c4-7b20-467d-9c98-279811a18fc9.png)


- 'connect_code', 'multi_linked' 컬럼에서 이상한 점 발견

#### 'connect_code'

```python
train.connect_code.value_counts()
```

```
0      4689075
103      12142
Name: connect_code, dtype: int64
```

```python
print(train.groupby('connect_code')['target'].mean())
print(train.groupby('connect_code')['target'].std())      
```

```
connect_code
0      42.749191
103    57.947044
Name: target, dtype: float32
connect_code
0      15.949711
103     9.075532
Name: target, dtype: float64
```

#### 'multi_linked'

```python
train.multi_linked.value_counts()
```

```
0    4698978
1       2239
Name: multi_linked, dtype: int64
```

```python
print(train.groupby('multi_linked')['target'].mean())
print(train.groupby('multi_linked')['target'].std())      
```

```
multi_linked
0    42.791370
1    36.642696
Name: target, dtype: float32
multi_linked
0    15.954885
1    13.661950
Name: target, dtype: float64
```

- 개수의 차이도 심각하고 평균값과 표준편차까지도 차이가 있는 것을 볼 수 있음.
- 컬럼 자체를 제거하기로 결정

```python
train['road_name']
```

```
0          지방도1112호선
1           일반국도11호선
2           일반국도16호선
3                태평로
4           일반국도12호선
             ...    
4701212            -
4701213            -
4701214     일반국도12호선
4701215     일반국도95호선
4701216          경찰로
Name: road_name, Length: 4701217, dtype: object
```

- 'road_name' 컬럼에서도 이상한 점 발견

```python
train['road_name'].unique()
```

```
array(['지방도1112호선', '일반국도11호선', '일반국도16호선', '태평로', '일반국도12호선', '경찰로', '-',
       '외도천교', '일반국도99호선', '중정로', '번영로', '연동로', '중산간서로', '지방도1118호선',
       '새서귀로', '지방도1115호선', '지방도1132호선', '어시천교', '지방도1120호선', '삼무로',
       '애조로', '지방도1116호선', '일반국도95호선', '동부관광도로', '동홍로', '지방도97호선', '중문로',
       '연삼로', '중앙로', '산서로', '지방도1117호선', '연북로', '남조로', '지방도1119호선', '동문로',
       '한천로', '삼봉로', '고평교', '연북2교', '관광단지로', '권학로', '시청로', '신대로', '서사로',
       '관덕로', '관광단지1로', '신산로', '관광단지2로', '신광로', '지방도1136호선', '첨단로',
       '제2거로교', '시민광장로', '임항로', '수영장길', '애원로', '삼성로', '일주동로', '호서중앙로',
       '아봉로', '호근로'], dtype=object)
```

```python
train['road_name'].value_counts()
```

```
일반국도12호선    1046092
-            569463
일반국도16호선     554510
일반국도95호선     248181
일반국도11호선     215701
             ...   
애원로            7718
아봉로            7342
남조로            6813
호서중앙로          2819
호근로             587
Name: road_name, Length: 61, dtype: int64
```

```python
road_name_null = train.groupby([train['road_name'] == '-'])['road_name'].value_counts()
road_name_null
```

```
road_name  road_name
False      일반국도12호선     1046092
           일반국도16호선      554510
           일반국도95호선      248181
           일반국도11호선      215701
           지방도1132호선     179200
                         ...   
           아봉로             7342
           남조로             6813
           호서중앙로           2819
           호근로              587
True       -             569463
Name: road_name, Length: 61, dtype: int64
```

```python
import warnings
warnings.filterwarnings('ignore')

pichart = road_name_null.plot(kind = 'pie', autopct = '%.1f%%', figsize=(16, 12))
pichart
```

```
<AxesSubplot:ylabel='road_name'>
```

![image](https://user-images.githubusercontent.com/84713532/206893470-a35827f4-f39b-4b9d-88a6-874062b7147f.png)


- 도로 이름이 아닌 '-' 값이 무시할 수 없을 정도로 있는 것을 확인(12.1%)

```python
train["base_date"] = pd.to_datetime(train["base_date"],format='%Y%m%d')
train['year']= train['base_date'].dt.year
train['month']= train['base_date'].dt.month
train['day']= train['base_date'].dt.day
```

#### 일별, 월별 평균속도 그래프

```python
figure, (ax1, ax2) = plt.subplots(nrows=2)
sns.pointplot(data=train, x="base_hour", y="target", ax=ax1)
sns.pointplot(data=train, x="month", y="target", ax=ax2)
```

```
<AxesSubplot:xlabel='month', ylabel='target'>
```

![image](https://user-images.githubusercontent.com/84713532/206893464-7ab919ea-1bc3-4c21-a083-e270e6b24770.png)


- 평균속도의 차이가 확연히 드러나고, 7월에 속도가 낮은 것을 확인할 수 있음

#### 범주형 데이터 확인

```python
figure, axes = plt.subplots(3,4)
figure.set_size_inches(20, 10)
sns.barplot(data=train, x='year', y='target',ax=axes[0][0])
sns.barplot(data=train, x='month', y='target',ax=axes[0][1])
sns.barplot(data=train, x='day', y='target',ax=axes[0][2])
sns.barplot(data=train, x='base_hour', y='target',ax=axes[0][3])
sns.barplot(data=train, x='lane_count', y='target',ax=axes[1][0])
sns.barplot(data=train, x='road_rating', y='target',ax=axes[1][1])
sns.barplot(data=train, x='multi_linked', y='target',ax=axes[1][2])
sns.barplot(data=train, x='connect_code', y='target',ax=axes[1][3])
sns.barplot(data=train, x='maximum_speed_limit', y='target',ax=axes[2][0])
sns.barplot(data=train, x='road_type', y='target',ax=axes[2][1])
sns.barplot(data=train, x='start_turn_restricted', y='target',ax=axes[2][2])
sns.barplot(data=train, x='end_turn_restricted', y='target',ax=axes[2][3])
```

```
<AxesSubplot:xlabel='end_turn_restricted', ylabel='target'>
```

![image](https://user-images.githubusercontent.com/84713532/206893454-faf7721f-c09c-4904-beca-c6f6b3e5f285.png)


```python
train['base_hour'].value_counts()
```

```
15    214541
13    214297
14    214182
12    211833
19    209870
11    208515
16    208420
17    208377
18    207500
10    206316
9     205327
20    205059
21    203585
8     201875
22    200629
7     199061
6     189418
23    184229
1     182353
5     181128
2     169322
4     165284
3     155938
0     154158
Name: base_hour, dtype: int64
```

```python
train['road_rating'].value_counts()
```

```
103    2159511
107    1582214
106     959492
Name: road_rating, dtype: int64
```

```python
train['maximum_speed_limit'].value_counts()
```

```
60.0    1665573
50.0    1103682
70.0     995077
80.0     700334
30.0     229761
40.0       6790
Name: maximum_speed_limit, dtype: int64
```

```python
# 최고 제한속도 확인
train.maximum_speed_limit.value_counts().sort_index()
```

```
30.0     229761
40.0       6790
50.0    1103682
60.0    1665573
70.0     995077
80.0     700334
Name: maximum_speed_limit, dtype: int64
```

```python
# 차로 수 확인
train.lane_count.value_counts()
```

```
2    2352092
1    1558531
3     790594
Name: lane_count, dtype: int64
```

#### 위도, 경도 확인

```python
train.plot(kind='scatter', x='start_longitude', y='start_latitude', alpha=0.3, grid=True)
```

```
<AxesSubplot:xlabel='start_longitude', ylabel='start_latitude'>
```

![image](https://user-images.githubusercontent.com/84713532/206893441-9db73d25-08fe-43af-b010-862eb149694a.png)


```python
# map
m = folium.Map(location=[33.427747, 126.662612], zoom_start=10.4, tiles="Stamen Terrain")

tooltip = "Click me!"
a=train[['start_latitude','start_longitude','start_node_name']]
a = a.drop_duplicates()
b=a['start_longitude']
c=a['start_node_name']
a=a['start_latitude']

for i,j,k in zip(a,b,c) :
    folium.Marker([i, j], popup="<i>{}</i>".format(k), tooltip=tooltip).add_to(m)
m
```

```python
gps = train[['start_longitude', 'end_longitude', 'start_latitude', 'end_latitude', 'target']]
```

```python
gps_set = [gps['start_longitude'].min(), gps['start_longitude'].max(), gps['start_latitude'].min(), gps['start_latitude'].max()] # 지도 그림의 gps 좌표
gps_set
```

```
[126.182616549771, 126.930940973848, 33.2434317486804, 33.5560801767072]
```

```python
from tqdm import tqdm
```

```python
vel_low_idx = gps.loc[gps['target']<15].index # 시내 교통 체증기준 10 km/h 미만
vel_high_idx = gps.loc[gps['target']>80].index # 고속도로 원활기준 80 km/h 초과
```

```python
f, ax = plt.subplots(figsize=(21,10))

ax.set_xlim(gps_set[0], gps_set[1])
ax.set_ylim(gps_set[2], gps_set[3])

image = plt.imread('map5.png')
ax.imshow(image, zorder=0, extent=gps_set, aspect='equal')

for i in tqdm(vel_low_idx): # 교통 체증 도로 빨강
    x_1 = gps.loc[i,'start_longitude']
    x_2 = gps.loc[i,'end_longitude'] 
    y_1 = gps.loc[i,'start_latitude']
    y_2 = gps.loc[i,'end_latitude'] 
    ax.plot([x_1, x_2], [y_1, y_2], color='red')

for i in tqdm(vel_high_idx): # 교통 원활 도로 파랑
    x_1 = gps.loc[i,'start_longitude']
    x_2 = gps.loc[i,'end_longitude'] 
    y_1 = gps.loc[i,'start_latitude']
    y_2 = gps.loc[i,'end_latitude'] 
    ax.plot([x_1, x_2], [y_1, y_2], color='blue')

plt.show()
```

```
100%|██████████| 111725/111725 [01:24<00:00, 1319.47it/s]
100%|██████████| 26239/26239 [00:16<00:00, 1558.22it/s]
```

![image](https://user-images.githubusercontent.com/84713532/206893430-5f8d4ed3-4a3c-45f8-8101-fde4b9e13c2c.png)


```python
sns.distplot(train.maximum_speed_limit)
```

```
<AxesSubplot:xlabel='maximum_speed_limit', ylabel='Density'>
```

![image](https://user-images.githubusercontent.com/84713532/206893415-f8e9bed9-1534-4ce4-a230-fb1f7cc21fba.png)


```python
sns.boxplot(x = "maximum_speed_limit", y = "target", data = train)
```

```
<AxesSubplot:xlabel='maximum_speed_limit', ylabel='target'>
```

![image](https://user-images.githubusercontent.com/84713532/206893408-878e2f80-bc2d-41c4-a09b-863746070944.png)


#### maximum_spped_limit 극단치확인 전처리과정에서 제거

```python
sns.histplot(x = train.target, hue = train.lane_count, palette=["C0", "C1", "k"])
```

```
<AxesSubplot:xlabel='target', ylabel='Count'>
```

![image](https://user-images.githubusercontent.com/84713532/206893397-96b676cb-c289-4127-921f-c1f9083cc728.png)


##### 차로 수 확인

```python
for i in train.groupby("lane_count")["target"].mean() :
    print(i)
```

```
43.57056163785
44.9157129057877
34.917783337591736
```

```python
sns.boxplot(x = train.lane_count, y = train.target)
```

```
<AxesSubplot:xlabel='lane_count', ylabel='target'>
```

![image](https://user-images.githubusercontent.com/84713532/206893391-db421963-56f0-436d-abb5-c83e9ab6776a.png)


##### 예상과는 반대로 차선이 수가 늘어날 수록 오히려 정체되는 모습을 보임. 모델학습 시 활용

## 2. Data Preprocessing

```
컬럼 삭제
'distance', 'airport_distance' 컬럼 생성
'center_start', 'center_end' 컬럼 생성
Target Encoding
결측값 제거
'season' 컬럼 추가
'work_or_rest_or_other' 컬럼 추가
'target' 이상치 제거
Label Encoding
```

## 컬럼 삭제

### 값이 유일한 컬럼 탐색

```python
train_desc = train.describe().transpose()
train_desc[train_desc['std']==0].index
```

```
Index(['vehicle_restricted', 'height_restricted'], dtype='object')
```

### 'vehicle_restricted', 'height_restricted' 삭제

```python
train = train.drop(columns=['vehicle_restricted', 'height_restricted'])
test = test.drop(columns=['vehicle_restricted', 'height_restricted'])
train.shape, test.shape
```

```
((4701217, 21), (291241, 20))
```

## 지리 데이터 (공간 파생변수)

### 두 지점 사이의 거리

```python
# https://www.kaggle.com/code/speedoheck/calculate-distance-with-geo-coordinates/notebook

from math import radians, cos, sin, asin, sqrt

def haversine(row):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1 = row['start_longitude']
    lat1 = row['start_latitude']
    lon2 = row['end_longitude']
    lat2 = row['end_latitude']

    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    km = 6367 * c
    return km

train['distance'] = train.apply(haversine, axis=1)
test['distance'] = test.apply(haversine, axis=1)
```

### 제주공항까지 거리

```python
def haversine_airport(row):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1 = 126.4913534
    lat1 = 33.5104135
    lon2 = (row['start_longitude'] + row['end_longitude']) / 2
    lat2 = (row['start_latitude'] + row['end_latitude']) / 2

    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    km = 6367 * c
    return km

train['airport_distance'] = train.apply(haversine_airport, axis=1)
test['airport_distance'] = test.apply(haversine_airport, axis=1)
```

### 제주도 권역별 구분하여 변수 추가

- 제주시 도심 : 126.4531517 ~ 126.5900257 , 33.4670429 ~
- 서귀포 도심 : 126.3972753 ~ 126.6076604 , ~ 33.2686052

```python
# 출발지점 권역
mask_jj_start = (train['start_longitude'] > 126.4531517) & (train['start_longitude']< 126.5900257) & (train['start_latitude'] > 33.4670429)
mask_jj_end = (train['end_longitude'] > 126.4531517) & (train['end_longitude']< 126.5900257) & (train['end_latitude'] > 33.4670429)

mask_sgp_start = (train['start_longitude'] > 126.3972753) & (train['start_longitude']< 126.6076604) & (train['start_latitude'] < 33.2686052)
mask_sgp_end = (train['end_longitude'] > 126.3972753) & (train['end_longitude']< 126.6076604) & (train['end_latitude'] < 33.2686052)
```

```python
train['center_start'] = 0
test['center_start'] = 0

train.loc[mask_jj_start, 'center_start'] = 1
train.loc[mask_sgp_start, 'center_start'] = 2

test.loc[mask_jj_start, 'center_start'] = 1
test.loc[mask_sgp_start, 'center_start'] = 2

train['center_end'] = 0
test['center_end'] = 0

train.loc[mask_jj_end, 'center_end'] = 1
train.loc[mask_sgp_end, 'center_end'] = 2

test.loc[mask_jj_end, 'center_end'] = 1
test.loc[mask_sgp_end, 'center_end'] = 2
```

### GPS 정보를 사용해서 road 구분

```python
train['road_code'] = train['start_latitude'].astype(str)+'_'+train['start_longitude'].astype(str)+'_'+train['end_latitude'].astype(str)+'_'+train['end_longitude'].astype(str)
train['road_code'].value_counts()
```

```
33.3058672207151_126.599081327413_33.3082357708673_126.598689775097    6477
33.3082357708673_126.598689775097_33.3058672207151_126.599081327413    6397
33.5014774884938_126.569223187609_33.4968633703578_126.58123009621     6077
33.5016270326083_126.568923085567_33.5014774884938_126.569223187609    6077
33.496710616894_126.581529061335_33.4918481088766_126.591872255149     6075
                                                                       ... 
33.2566709359707_126.52441046863_33.2541529264473_126.524330998601      744
33.26127013848_126.524428741607_33.2574097173209_126.524412034435       744
33.2574097173209_126.524412034435_33.2566709359707_126.52441046863      744
33.2574097173209_126.524412034435_33.26127013848_126.524428741607       587
33.2574006381515_126.52574476307_33.2574097173209_126.524412034435      587
Name: road_code, Length: 904, dtype: int64
```

```python
test['road_code'] = test['start_latitude'].astype(str)+'_'+test['start_longitude'].astype(str)+'_'+test['end_latitude'].astype(str)+'_'+test['end_longitude'].astype(str)
test['road_code'].value_counts()
```

```
33.508463678702_126.558231105407_33.5087115227295_126.558702856002     740
33.4937925855376_126.492189386746_33.4923347723675_126.490247073997    740
33.4666066165642_126.454021511351_33.4664333666973_126.454583167413    740
33.4923347723675_126.490247073997_33.4937925855376_126.492189386746    740
33.4658632729266_126.456384480352_33.4664333666973_126.454583167413    740
                                                                      ... 
33.3452396554215_126.850113181832_33.3446283972409_126.849278713014      7
33.4857069297096_126.604162168012_33.4886994919865_126.597620980703      7
33.4379464931581_126.73250865826_33.4383285187565_126.732031757687       7
33.4359411786532_126.736248543312_33.4379464931581_126.73250865826       7
33.4288406442461_126.750881044473_33.4359411786532_126.736248543312      7
Name: road_code, Length: 441, dtype: int64
```

### Target Encoding

```python
road_stats = train.groupby(['road_code'])[['target']].agg(['min', 'mean', 'max', 'std']).reset_index()
road_stats.columns = ['road_code', 'road_min', 'road_mean', 'road_max', 'road_std']
train = train.merge(road_stats, how='left', on='road_code')
train.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>base_date</th>
      <th>day_of_week</th>
      <th>base_hour</th>
      <th>lane_count</th>
      <th>road_rating</th>
      <th>road_name</th>
      <th>multi_linked</th>
      <th>connect_code</th>
      <th>maximum_speed_limit</th>
      <th>...</th>
      <th>lon_change</th>
      <th>distance</th>
      <th>airport_distance</th>
      <th>center_start</th>
      <th>center_end</th>
      <th>road_code</th>
      <th>road_min</th>
      <th>road_mean</th>
      <th>road_max</th>
      <th>road_std</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>TRAIN_0000000</td>
      <td>20220623</td>
      <td>목</td>
      <td>17</td>
      <td>1</td>
      <td>106</td>
      <td>지방도1112호선</td>
      <td>0</td>
      <td>0</td>
      <td>60.0</td>
      <td>...</td>
      <td>0.000277</td>
      <td>0.025694</td>
      <td>18.330548</td>
      <td>0</td>
      <td>0</td>
      <td>33.427747274683_126.662612038652_33.4277487730...</td>
      <td>16.0</td>
      <td>51.756910</td>
      <td>72.0</td>
      <td>4.587047</td>
    </tr>
    <tr>
      <th>1</th>
      <td>TRAIN_0000001</td>
      <td>20220728</td>
      <td>목</td>
      <td>21</td>
      <td>2</td>
      <td>103</td>
      <td>일반국도11호선</td>
      <td>0</td>
      <td>0</td>
      <td>60.0</td>
      <td>...</td>
      <td>0.002867</td>
      <td>0.525560</td>
      <td>3.470872</td>
      <td>1</td>
      <td>1</td>
      <td>33.5007304293026_126.529106761554_33.504811303...</td>
      <td>5.0</td>
      <td>26.400712</td>
      <td>59.0</td>
      <td>7.102290</td>
    </tr>
    <tr>
      <th>2</th>
      <td>TRAIN_0000002</td>
      <td>20211010</td>
      <td>일</td>
      <td>7</td>
      <td>2</td>
      <td>103</td>
      <td>일반국도16호선</td>
      <td>0</td>
      <td>0</td>
      <td>80.0</td>
      <td>...</td>
      <td>0.006450</td>
      <td>0.608016</td>
      <td>28.185912</td>
      <td>0</td>
      <td>0</td>
      <td>33.2791450972975_126.368597660936_33.280072104...</td>
      <td>32.0</td>
      <td>59.101720</td>
      <td>88.0</td>
      <td>12.091252</td>
    </tr>
    <tr>
      <th>3</th>
      <td>TRAIN_0000003</td>
      <td>20220311</td>
      <td>금</td>
      <td>13</td>
      <td>2</td>
      <td>107</td>
      <td>태평로</td>
      <td>0</td>
      <td>0</td>
      <td>50.0</td>
      <td>...</td>
      <td>0.000976</td>
      <td>0.107285</td>
      <td>30.222870</td>
      <td>2</td>
      <td>2</td>
      <td>33.2460808686345_126.56720431031_33.2455654004...</td>
      <td>2.0</td>
      <td>25.024923</td>
      <td>51.0</td>
      <td>7.667545</td>
    </tr>
    <tr>
      <th>4</th>
      <td>TRAIN_0000004</td>
      <td>20211005</td>
      <td>화</td>
      <td>8</td>
      <td>2</td>
      <td>103</td>
      <td>일반국도12호선</td>
      <td>0</td>
      <td>0</td>
      <td>80.0</td>
      <td>...</td>
      <td>-0.003601</td>
      <td>0.337736</td>
      <td>16.019878</td>
      <td>0</td>
      <td>0</td>
      <td>33.4622143482158_126.326551111199_33.462676772...</td>
      <td>25.0</td>
      <td>39.873670</td>
      <td>72.0</td>
      <td>6.946840</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 32 columns</p>
</div>

```python
test = test.merge(road_stats, how='left', on='road_code')
print(test['road_code'].isnull().sum())
test.head()
```

```
0
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }


</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>base_date</th>
      <th>day_of_week</th>
      <th>base_hour</th>
      <th>lane_count</th>
      <th>road_rating</th>
      <th>road_name</th>
      <th>multi_linked</th>
      <th>connect_code</th>
      <th>maximum_speed_limit</th>
      <th>...</th>
      <th>lon_change</th>
      <th>distance</th>
      <th>airport_distance</th>
      <th>center_start</th>
      <th>center_end</th>
      <th>road_code</th>
      <th>road_min</th>
      <th>road_mean</th>
      <th>road_max</th>
      <th>road_std</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>TEST_000000</td>
      <td>20220825</td>
      <td>목</td>
      <td>17</td>
      <td>3</td>
      <td>107</td>
      <td>연삼로</td>
      <td>0</td>
      <td>0</td>
      <td>70.0</td>
      <td>...</td>
      <td>-0.002538</td>
      <td>0.278752</td>
      <td>4.881938</td>
      <td>0</td>
      <td>0</td>
      <td>33.4994265233055_126.541298167922_33.500772491...</td>
      <td>9.0</td>
      <td>33.623164</td>
      <td>60.0</td>
      <td>9.997806</td>
    </tr>
    <tr>
      <th>1</th>
      <td>TEST_000001</td>
      <td>20220809</td>
      <td>화</td>
      <td>12</td>
      <td>2</td>
      <td>103</td>
      <td>일반국도12호선</td>
      <td>0</td>
      <td>0</td>
      <td>70.0</td>
      <td>...</td>
      <td>0.011164</td>
      <td>1.038287</td>
      <td>28.756351</td>
      <td>1</td>
      <td>1</td>
      <td>33.2585071159642_126.427003448638_33.258119397...</td>
      <td>21.0</td>
      <td>48.359276</td>
      <td>69.0</td>
      <td>5.381406</td>
    </tr>
    <tr>
      <th>2</th>
      <td>TEST_000002</td>
      <td>20220805</td>
      <td>금</td>
      <td>2</td>
      <td>1</td>
      <td>103</td>
      <td>일반국도16호선</td>
      <td>0</td>
      <td>0</td>
      <td>60.0</td>
      <td>...</td>
      <td>0.001820</td>
      <td>0.171335</td>
      <td>27.967410</td>
      <td>0</td>
      <td>0</td>
      <td>33.2589595714352_126.476507600171_33.259205665...</td>
      <td>35.0</td>
      <td>59.993453</td>
      <td>86.0</td>
      <td>5.444524</td>
    </tr>
    <tr>
      <th>3</th>
      <td>TEST_000003</td>
      <td>20220818</td>
      <td>목</td>
      <td>23</td>
      <td>3</td>
      <td>103</td>
      <td>일반국도11호선</td>
      <td>0</td>
      <td>0</td>
      <td>70.0</td>
      <td>...</td>
      <td>0.000180</td>
      <td>0.270917</td>
      <td>6.572143</td>
      <td>2</td>
      <td>2</td>
      <td>33.4734941166381_126.54564685499_33.4710608036...</td>
      <td>6.0</td>
      <td>33.185444</td>
      <td>55.0</td>
      <td>6.060564</td>
    </tr>
    <tr>
      <th>4</th>
      <td>TEST_000004</td>
      <td>20220810</td>
      <td>수</td>
      <td>17</td>
      <td>3</td>
      <td>106</td>
      <td>번영로</td>
      <td>0</td>
      <td>0</td>
      <td>70.0</td>
      <td>...</td>
      <td>-0.012007</td>
      <td>1.225101</td>
      <td>7.871524</td>
      <td>0</td>
      <td>0</td>
      <td>33.5014774884938_126.569223187609_33.496863370...</td>
      <td>10.0</td>
      <td>46.299654</td>
      <td>83.0</td>
      <td>6.779434</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 31 columns</p>
</div>

### 'road_name' 컬럼의 "-" (결측값) 제거

```python
drop_road_name_index = train[train["road_name"] == "-"].index
temp_train = train.iloc[drop_road_name_index]
temp_train
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }


</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>base_date</th>
      <th>day_of_week</th>
      <th>base_hour</th>
      <th>lane_count</th>
      <th>road_rating</th>
      <th>road_name</th>
      <th>multi_linked</th>
      <th>connect_code</th>
      <th>maximum_speed_limit</th>
      <th>...</th>
      <th>lon_change</th>
      <th>distance</th>
      <th>airport_distance</th>
      <th>center_start</th>
      <th>center_end</th>
      <th>road_code</th>
      <th>road_min</th>
      <th>road_mean</th>
      <th>road_max</th>
      <th>road_std</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6</th>
      <td>TRAIN_0000006</td>
      <td>20220106</td>
      <td>목</td>
      <td>0</td>
      <td>2</td>
      <td>107</td>
      <td>-</td>
      <td>0</td>
      <td>0</td>
      <td>60.0</td>
      <td>...</td>
      <td>-0.001349</td>
      <td>0.487165</td>
      <td>23.139421</td>
      <td>0</td>
      <td>0</td>
      <td>33.418411972574_126.268029025365_33.4141750121...</td>
      <td>2.0</td>
      <td>37.754143</td>
      <td>72.0</td>
      <td>6.166624</td>
    </tr>
    <tr>
      <th>14</th>
      <td>TRAIN_0000014</td>
      <td>20220203</td>
      <td>목</td>
      <td>16</td>
      <td>1</td>
      <td>107</td>
      <td>-</td>
      <td>0</td>
      <td>0</td>
      <td>60.0</td>
      <td>...</td>
      <td>0.000767</td>
      <td>0.073214</td>
      <td>24.778337</td>
      <td>0</td>
      <td>0</td>
      <td>33.3169132415404_126.624634355945_33.317065379...</td>
      <td>30.0</td>
      <td>58.279321</td>
      <td>72.0</td>
      <td>4.900624</td>
    </tr>
    <tr>
      <th>28</th>
      <td>TRAIN_0000028</td>
      <td>20220612</td>
      <td>일</td>
      <td>14</td>
      <td>2</td>
      <td>107</td>
      <td>-</td>
      <td>0</td>
      <td>0</td>
      <td>50.0</td>
      <td>...</td>
      <td>-0.000808</td>
      <td>0.076826</td>
      <td>23.654612</td>
      <td>0</td>
      <td>0</td>
      <td>33.3308220849345_126.354178885417_33.330672710...</td>
      <td>23.0</td>
      <td>58.649140</td>
      <td>72.0</td>
      <td>4.145843</td>
    </tr>
    <tr>
      <th>30</th>
      <td>TRAIN_0000030</td>
      <td>20220623</td>
      <td>목</td>
      <td>6</td>
      <td>2</td>
      <td>107</td>
      <td>-</td>
      <td>0</td>
      <td>0</td>
      <td>80.0</td>
      <td>...</td>
      <td>0.009314</td>
      <td>0.872175</td>
      <td>8.313546</td>
      <td>0</td>
      <td>0</td>
      <td>33.4722764927296_126.418442753822_33.473390153...</td>
      <td>10.0</td>
      <td>60.106969</td>
      <td>87.0</td>
      <td>5.821785</td>
    </tr>
    <tr>
      <th>31</th>
      <td>TRAIN_0000031</td>
      <td>20211028</td>
      <td>목</td>
      <td>15</td>
      <td>1</td>
      <td>107</td>
      <td>-</td>
      <td>0</td>
      <td>0</td>
      <td>60.0</td>
      <td>...</td>
      <td>-0.000308</td>
      <td>0.033301</td>
      <td>27.020508</td>
      <td>0</td>
      <td>0</td>
      <td>33.3372436401165_126.695809499868_33.337397876...</td>
      <td>17.0</td>
      <td>46.824555</td>
      <td>69.0</td>
      <td>5.048228</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>4701204</th>
      <td>TRAIN_4701204</td>
      <td>20211001</td>
      <td>금</td>
      <td>19</td>
      <td>1</td>
      <td>107</td>
      <td>-</td>
      <td>0</td>
      <td>0</td>
      <td>60.0</td>
      <td>...</td>
      <td>-0.010732</td>
      <td>1.240364</td>
      <td>26.207930</td>
      <td>0</td>
      <td>0</td>
      <td>33.3250958586337_126.665698173918_33.331742200...</td>
      <td>11.0</td>
      <td>57.356468</td>
      <td>71.0</td>
      <td>3.816154</td>
    </tr>
    <tr>
      <th>4701205</th>
      <td>TRAIN_4701205</td>
      <td>20220112</td>
      <td>수</td>
      <td>19</td>
      <td>1</td>
      <td>107</td>
      <td>-</td>
      <td>0</td>
      <td>0</td>
      <td>50.0</td>
      <td>...</td>
      <td>0.004414</td>
      <td>0.442984</td>
      <td>40.172235</td>
      <td>0</td>
      <td>0</td>
      <td>33.450215274642_126.920771364786_33.4486894847...</td>
      <td>7.0</td>
      <td>33.133922</td>
      <td>72.0</td>
      <td>12.865501</td>
    </tr>
    <tr>
      <th>4701208</th>
      <td>TRAIN_4701208</td>
      <td>20220323</td>
      <td>수</td>
      <td>19</td>
      <td>2</td>
      <td>107</td>
      <td>-</td>
      <td>0</td>
      <td>0</td>
      <td>80.0</td>
      <td>...</td>
      <td>-0.000987</td>
      <td>0.095026</td>
      <td>33.663477</td>
      <td>0</td>
      <td>0</td>
      <td>33.2873555929438_126.736525350221_33.287581677...</td>
      <td>32.0</td>
      <td>58.925765</td>
      <td>93.0</td>
      <td>4.357269</td>
    </tr>
    <tr>
      <th>4701212</th>
      <td>TRAIN_4701212</td>
      <td>20211104</td>
      <td>목</td>
      <td>16</td>
      <td>1</td>
      <td>107</td>
      <td>-</td>
      <td>0</td>
      <td>0</td>
      <td>50.0</td>
      <td>...</td>
      <td>0.004374</td>
      <td>0.426736</td>
      <td>22.277728</td>
      <td>0</td>
      <td>0</td>
      <td>33.4221445845451_126.278124511889_33.420954611...</td>
      <td>9.0</td>
      <td>27.620482</td>
      <td>53.0</td>
      <td>5.777613</td>
    </tr>
    <tr>
      <th>4701213</th>
      <td>TRAIN_4701213</td>
      <td>20220331</td>
      <td>목</td>
      <td>2</td>
      <td>2</td>
      <td>107</td>
      <td>-</td>
      <td>0</td>
      <td>0</td>
      <td>80.0</td>
      <td>...</td>
      <td>-0.000522</td>
      <td>0.048433</td>
      <td>7.481579</td>
      <td>0</td>
      <td>0</td>
      <td>33.4725049919444_126.424368218158_33.472525119...</td>
      <td>20.0</td>
      <td>68.900580</td>
      <td>88.0</td>
      <td>5.235256</td>
    </tr>
  </tbody>
</table>
<p>569463 rows × 32 columns</p>
</div>

```python
print(temp_train["road_rating"].value_counts())
print(temp_train["weight_restricted"].value_counts())
print("----------------------------------")
print(train[(train["road_rating"] == 107) & (train["weight_restricted"] == 43200.0)]["road_name"].value_counts())
print(train[(train["road_rating"] == 107) & (train["weight_restricted"] == 32400.0)]["road_name"].value_counts())
```

```
107    569463
Name: road_rating, dtype: int64
0.0        481943
43200.0     68013
32400.0     19507
Name: weight_restricted, dtype: int64
----------------------------------
-      68013
중문로    11336
Name: road_name, dtype: int64
-      19507
산서로     7940
Name: road_name, dtype: int64
```

- 'road_name'이 "-"인 값은 모두 road_rating이 107입니다.
- 'road_name'이 "-" 인 값에서 'weight_restricted'가 43200인 곳은 "중문로", 32400인 곳은 "산서로"입니다.
- 이들을 모두 대체하겠습니다.

```python
# .loc로 값 대체하기 전의 수 = 569463
print(len(train[train["road_name"] == "-"]))
train.loc[(train["road_rating"] == 107) & (train["weight_restricted"] == 32400.0) & (train["road_name"] == "-"), "road_name"] = "산서로"
train.loc[(train["road_rating"] == 107) & (train["weight_restricted"] == 43200.0) & (train["road_name"] == "-"), "road_name"] = "중문로"

test.loc[(test["road_rating"] == 107) & (test["weight_restricted"] == 32400.0) & (test["road_name"] == "-"), "road_name"] = "산서로"
test.loc[(test["road_rating"] == 107) & (test["weight_restricted"] == 43200.0) & (test["road_name"] == "-"), "road_name"] = "중문로"

# .loc로 값 대체한 이후의 수 = 481943
print(len(train[train["road_name"] == "-"]))
```

```
569463
481943
```

- 약 8만7천개의 "-" 값을 대체
- 추가적으로 "-" 값을 탐색

```python
# "-" 값 대체를 위한 탐색"
# 모든 값을 뽑으면 너무 길어지기에, 2개 값만을 출력합니다.
for i in train["start_node_name"].unique():
    if (len(train[(train["start_node_name"] == i)]["road_name"].value_counts()) != 2) :
        continue
    if "-" in train[(train["start_node_name"] == i)]["road_name"].value_counts().index:
        print("----------------", i, "-------------------")
        print(train[(train["start_node_name"] == i)]["road_name"].value_counts())
```

```
---------------- 송목교 -------------------
중문로    10390
-       5183
Name: road_name, dtype: int64
---------------- 남수교 -------------------
중문로    10360
-       5156
Name: road_name, dtype: int64
---------------- 하귀입구 -------------------
일반국도12호선    10656
-            5190
Name: road_name, dtype: int64
---------------- 양계장 -------------------
-           5330
일반국도12호선    5329
Name: road_name, dtype: int64
---------------- 난산입구 -------------------
지방도1119호선    4923
-            3113
Name: road_name, dtype: int64
---------------- 영주교 -------------------
일반국도11호선    23909
-             472
Name: road_name, dtype: int64
---------------- 서중2교 -------------------
중문로    10380
-       5204
Name: road_name, dtype: int64
---------------- 천제이교 -------------------
-      10930
산서로    10706
Name: road_name, dtype: int64
---------------- 하나로교 -------------------
중문로    10578
-       5282
Name: road_name, dtype: int64
---------------- 신하교 -------------------
중문로    10390
-       5205
Name: road_name, dtype: int64
---------------- 야영장 -------------------
관광단지1로    5570
-         5415
Name: road_name, dtype: int64
---------------- 월계교 -------------------
-      9073
산서로    8801
Name: road_name, dtype: int64
---------------- 서울이용원 -------------------
태평로    11669
-       1425
Name: road_name, dtype: int64
---------------- 김녕교차로 -------------------
일반국도12호선    5312
-           3107
Name: road_name, dtype: int64
---------------- 어도초등교 -------------------
-           7300
일반국도16호선    4942
Name: road_name, dtype: int64
---------------- 광삼교 -------------------
중문로    5341
-      5333
Name: road_name, dtype: int64
---------------- 오렌지농원 -------------------
일반국도11호선    5933
-            464
Name: road_name, dtype: int64
---------------- 우사 -------------------
-           10036
일반국도16호선     1651
Name: road_name, dtype: int64
---------------- 서귀포시산림조합 -------------------
지방도1136호선    5951
-            4200
Name: road_name, dtype: int64
---------------- 성읍삼거리 -------------------
일반국도16호선    4991
-           4857
Name: road_name, dtype: int64
```

- 위에서 나온 값들을 다음과 같이 대체

```python
print(len(train[train["road_name"] == "-"]))

train.loc[(train["start_node_name"] == "송목교") & (train["road_name"] == "-"), "road_name"] = "중문로"
train.loc[(train["start_node_name"] == "남수교") & (train["road_name"] == "-"), "road_name"] = "중문로"
train.loc[(train["start_node_name"] == "하귀입구") & (train["road_name"] == "-"), "road_name"] = "일반국도12호선"
train.loc[(train["start_node_name"] == "양계장") & (train["road_name"] == "-"), "road_name"] = "일반국도12호선"
train.loc[(train["start_node_name"] == "난산입구") & (train["road_name"] == "-"), "road_name"] = "지방도1119호선"
train.loc[(train["start_node_name"] == "영주교") & (train["road_name"] == "-"), "road_name"] = "일반국도11호선"
train.loc[(train["start_node_name"] == "서중2교") & (train["road_name"] == "-"), "road_name"] = "중문로"
train.loc[(train["start_node_name"] == "천제이교") & (train["road_name"] == "-"), "road_name"] = "산서로"
train.loc[(train["start_node_name"] == "하나로교") & (train["road_name"] == "-"), "road_name"] = "중문로"
train.loc[(train["start_node_name"] == "신하교") & (train["road_name"] == "-"), "road_name"] = "중문로"
train.loc[(train["start_node_name"] == "야영장") & (train["road_name"] == "-"), "road_name"] = "관광단지1로"
train.loc[(train["start_node_name"] == "월계교") & (train["road_name"] == "-"), "road_name"] = "산서로"
train.loc[(train["start_node_name"] == "서울이용원") & (train["road_name"] == "-"), "road_name"] = "태평로"
train.loc[(train["start_node_name"] == "김녕교차로") & (train["road_name"] == "-"), "road_name"] = "일반국도12호선"
train.loc[(train["start_node_name"] == "어도초등교") & (train["road_name"] == "-"), "road_name"] = "일반국도16호선"
train.loc[(train["start_node_name"] == "광삼교") & (train["road_name"] == "-"), "road_name"] = "중문로"
train.loc[(train["start_node_name"] == "오렌지농원") & (train["road_name"] == "-"), "road_name"] = "일반국도11호선"
train.loc[(train["start_node_name"] == "우사") & (train["road_name"] == "-"), "road_name"] = "일반국도16호선"
train.loc[(train["start_node_name"] == "서귀포시산림조합") & (train["road_name"] == "-"), "road_name"] = "지방도1136호선"
train.loc[(train["start_node_name"] == "성읍삼거리") & (train["road_name"] == "-"), "road_name"] = "일반국도16호선"

test.loc[(test["start_node_name"] == "송목교") & (test["road_name"] == "-"), "road_name"] = "중문로"
test.loc[(test["start_node_name"] == "남수교") & (test["road_name"] == "-"), "road_name"] = "중문로"
test.loc[(test["start_node_name"] == "하귀입구") & (test["road_name"] == "-"), "road_name"] = "일반국도12호선"
test.loc[(test["start_node_name"] == "양계장") & (test["road_name"] == "-"), "road_name"] = "일반국도12호선"
test.loc[(test["start_node_name"] == "난산입구") & (test["road_name"] == "-"), "road_name"] = "지방도1119호선"
test.loc[(test["start_node_name"] == "영주교") & (test["road_name"] == "-"), "road_name"] = "일반국도11호선"
test.loc[(test["start_node_name"] == "서중2교") & (test["road_name"] == "-"), "road_name"] = "중문로"
test.loc[(test["start_node_name"] == "천제이교") & (test["road_name"] == "-"), "road_name"] = "산서로"
test.loc[(test["start_node_name"] == "하나로교") & (test["road_name"] == "-"), "road_name"] = "중문로"
test.loc[(test["start_node_name"] == "신하교") & (test["road_name"] == "-"), "road_name"] = "중문로"
test.loc[(test["start_node_name"] == "야영장") & (test["road_name"] == "-"), "road_name"] = "관광단지1로"
test.loc[(test["start_node_name"] == "월계교") & (test["road_name"] == "-"), "road_name"] = "산서로"
test.loc[(test["start_node_name"] == "서울이용원") & (test["road_name"] == "-"), "road_name"] = "태평로"
test.loc[(test["start_node_name"] == "김녕교차로") & (test["road_name"] == "-"), "road_name"] = "일반국도12호선"
test.loc[(test["start_node_name"] == "어도초등교") & (test["road_name"] == "-"), "road_name"] = "일반국도16호선"
test.loc[(test["start_node_name"] == "광삼교") & (test["road_name"] == "-"), "road_name"] = "중문로"
test.loc[(test["start_node_name"] == "오렌지농원") & (test["road_name"] == "-"), "road_name"] = "일반국도11호선"
test.loc[(test["start_node_name"] == "우사") & (test["road_name"] == "-"), "road_name"] = "일반국도16호선"
test.loc[(test["start_node_name"] == "서귀포시산림조합") & (test["road_name"] == "-"), "road_name"] = "지방도1136호선"
test.loc[(test["start_node_name"] == "성읍삼거리") & (test["road_name"] == "-"), "road_name"] = "일반국도16호선"

print(len(train[train["road_name"] == "-"]))
```

```
481943
379668
```

- 약 10만개의 값이 대체
- 이어서 대체할 값을 탐색

```python
# "-" 값 대체를 위한 탐색"
for i in train["end_node_name"].unique():
    if (len(train[(train["end_node_name"] == i)]["road_name"].value_counts()) != 2) :
        continue
    if "-" in train[(train["end_node_name"] == i)]["road_name"].value_counts().index:
        print("----------------", i, "-------------------")
        print(train[(train["end_node_name"] == i)]["road_name"].value_counts())
```

```
---------------- 남수교 -------------------
중문로    10360
-       5187
Name: road_name, dtype: int64
---------------- 농협주유소 -------------------
-      8053
산서로    5089
Name: road_name, dtype: int64
---------------- 난산입구 -------------------
지방도1119호선    4978
-            2946
Name: road_name, dtype: int64
---------------- 성읍삼거리 -------------------
일반국도16호선    5030
-           4670
Name: road_name, dtype: int64
---------------- 김녕교차로 -------------------
일반국도12호선    5281
-           3266
Name: road_name, dtype: int64
---------------- 한남교차로 -------------------
중문로    5204
-      5198
Name: road_name, dtype: int64
---------------- 서울이용원 -------------------
태평로    11653
-       1417
Name: road_name, dtype: int64
---------------- 하귀입구 -------------------
일반국도12호선    10661
-            5144
Name: road_name, dtype: int64
---------------- 우사 -------------------
일반국도16호선    7677
-           4784
Name: road_name, dtype: int64
---------------- 어도초등교 -------------------
-           7053
일반국도16호선    5135
Name: road_name, dtype: int64
---------------- 월계교 -------------------
-      9598
산서로    8801
Name: road_name, dtype: int64
---------------- 양계장 -------------------
일반국도12호선    5352
-           5331
Name: road_name, dtype: int64
---------------- 하나로교 -------------------
중문로    10578
-       5306
Name: road_name, dtype: int64
---------------- 광삼교 -------------------
-      5342
중문로    5341
Name: road_name, dtype: int64
---------------- 수간교차로 -------------------
일반국도12호선    5330
-           5329
Name: road_name, dtype: int64
---------------- 난산사거리 -------------------
지방도1119호선    3113
-            2841
Name: road_name, dtype: int64
---------------- 서중2교 -------------------
중문로    10380
-       5187
Name: road_name, dtype: int64
---------------- 서귀포시산림조합 -------------------
지방도1136호선    5962
-            5471
Name: road_name, dtype: int64
---------------- 옹포사거리 -------------------
-      4338
산서로    3984
Name: road_name, dtype: int64
---------------- 진은교차로 -------------------
-      5307
중문로    5282
Name: road_name, dtype: int64
```

```python
print(len(train[train["road_name"] == "-"]))

train.loc[(train["end_node_name"] == "남수교") & (train["road_name"] == "-"), "road_name"] = "중문로"
train.loc[(train["end_node_name"] == "농협주유소") & (train["road_name"] == "-"), "road_name"] = "월계교"
train.loc[(train["end_node_name"] == "난산입구") & (train["road_name"] == "-"), "road_name"] = "지방도1119호선"
train.loc[(train["end_node_name"] == "성읍삼거리") & (train["road_name"] == "-"), "road_name"] = "일반국도16호선"
train.loc[(train["end_node_name"] == "김녕교차로") & (train["road_name"] == "-"), "road_name"] = "일반국도12호선"
train.loc[(train["end_node_name"] == "한남교차로") & (train["road_name"] == "-"), "road_name"] = "서중2교"
train.loc[(train["end_node_name"] == "서울이용원") & (train["road_name"] == "-"), "road_name"] = "태평로"
train.loc[(train["end_node_name"] == "하귀입구") & (train["road_name"] == "-"), "road_name"] = "일반국도12호선"
train.loc[(train["end_node_name"] == "어도초등교") & (train["road_name"] == "-"), "road_name"] = "일반국도16호선"
train.loc[(train["end_node_name"] == "월계교") & (train["road_name"] == "-"), "road_name"] = "산서로"
train.loc[(train["end_node_name"] == "양계장") & (train["road_name"] == "-"), "road_name"] = "일반국도12호선"
train.loc[(train["end_node_name"] == "하나로교") & (train["road_name"] == "-"), "road_name"] = "중문로"
train.loc[(train["end_node_name"] == "광삼교") & (train["road_name"] == "-"), "road_name"] = "중문로"
train.loc[(train["end_node_name"] == "수간교차로") & (train["road_name"] == "-"), "road_name"] = "양계장"
train.loc[(train["end_node_name"] == "난산사거리") & (train["road_name"] == "-"), "road_name"] = "난산입구"
train.loc[(train["end_node_name"] == "서중2교") & (train["road_name"] == "-"), "road_name"] = "중문로"
train.loc[(train["end_node_name"] == "서귀포시산림조합") & (train["road_name"] == "-"), "road_name"] = "지방도1136호선"
train.loc[(train["end_node_name"] == "옹포사거리") & (train["road_name"] == "-"), "road_name"] = "월계교"
train.loc[(train["end_node_name"] == "진은교차로") & (train["road_name"] == "-"), "road_name"] = "하나로교"

test.loc[(test["end_node_name"] == "남수교") & (test["road_name"] == "-"), "road_name"] = "중문로"
test.loc[(test["end_node_name"] == "농협주유소") & (test["road_name"] == "-"), "road_name"] = "월계교"
test.loc[(test["end_node_name"] == "난산입구") & (test["road_name"] == "-"), "road_name"] = "지방도1119호선"
test.loc[(test["end_node_name"] == "성읍삼거리") & (test["road_name"] == "-"), "road_name"] = "일반국도16호선"
test.loc[(test["end_node_name"] == "김녕교차로") & (test["road_name"] == "-"), "road_name"] = "일반국도12호선"
test.loc[(test["end_node_name"] == "한남교차로") & (test["road_name"] == "-"), "road_name"] = "서중2교"
test.loc[(test["end_node_name"] == "서울이용원") & (test["road_name"] == "-"), "road_name"] = "태평로"
test.loc[(test["end_node_name"] == "하귀입구") & (test["road_name"] == "-"), "road_name"] = "일반국도12호선"
test.loc[(test["end_node_name"] == "어도초등교") & (test["road_name"] == "-"), "road_name"] = "일반국도16호선"
test.loc[(test["end_node_name"] == "월계교") & (test["road_name"] == "-"), "road_name"] = "산서로"
test.loc[(test["end_node_name"] == "양계장") & (test["road_name"] == "-"), "road_name"] = "일반국도12호선"
test.loc[(test["end_node_name"] == "하나로교") & (test["road_name"] == "-"), "road_name"] = "중문로"
test.loc[(test["end_node_name"] == "광삼교") & (test["road_name"] == "-"), "road_name"] = "중문로"
test.loc[(test["end_node_name"] == "수간교차로") & (test["road_name"] == "-"), "road_name"] = "양계장"
test.loc[(test["end_node_name"] == "난산사거리") & (test["road_name"] == "-"), "road_name"] = "난산입구"
test.loc[(test["end_node_name"] == "서중2교") & (test["road_name"] == "-"), "road_name"] = "중문로"
test.loc[(test["end_node_name"] == "서귀포시산림조합") & (test["road_name"] == "-"), "road_name"] = "지방도1136호선"
test.loc[(test["end_node_name"] == "옹포사거리") & (test["road_name"] == "-"), "road_name"] = "월계교"
test.loc[(test["end_node_name"] == "진은교차로") & (test["road_name"] == "-"), "road_name"] = "하나로교"

print(len(train[train["road_name"] == "-"]))
```

```
379668
282684
```

- 이번에는 약 9만 6천개의 데이터가 대체

```python
# 소숫점 문제상 출력된 값을 그대로 사용한다면 값을 대체할 수 없는 문제가 있습니다.
# 이를 해결하기 위해서 소숫점의 자릿수를 제한하겠습니다.
print(train["start_latitude"].nunique(),train["start_longitude"].nunique(), train["end_latitude"].nunique(), train["end_longitude"].nunique())

# 7번째자리에서 반올림 할 경우 train에서의 고윳값 갯수가 변하지 않습니다
train[["start_latitude", "start_longitude", "end_latitude", "end_longitude"]] = train[["start_latitude", "start_longitude", "end_latitude", "end_longitude"]].apply(lambda x: round(x, 6))
test[["start_latitude", "start_longitude", "end_latitude", "end_longitude"]] = test[["start_latitude", "start_longitude", "end_latitude", "end_longitude"]].apply(lambda x: round(x, 6))

print(train["start_latitude"].nunique(),train["start_longitude"].nunique(), train["end_latitude"].nunique(), train["end_longitude"].nunique())
```

```
586 586 586 586
586 586 586 586
```

- 고윳값의 갯수가 변하지 않은 것을 확인할 수 있습니다.
- 이어서 위도, 경도를 바탕으로 대체할 수 있는 값을 찾아보겠습니다

```python
for i in train["start_latitude"].unique():
    if (len(train[(train["start_latitude"] == i)]["road_name"].value_counts()) != 2) :
        continue
    if "-" in train[(train["start_latitude"] == i)]["road_name"].value_counts().index:
        print("----------------", i, "-------------------")
        print(train[(train["start_latitude"] == i)]["road_name"].value_counts())
```

```
---------------- 33.409416 -------------------
-      3321
월계교    3184
Name: road_name, dtype: int64
---------------- 33.402546 -------------------
-            2953
지방도1119호선    2946
Name: road_name, dtype: int64
---------------- 33.471164 -------------------
-           5334
일반국도12호선    5331
Name: road_name, dtype: int64
---------------- 33.411255 -------------------
-      7382
월계교    4338
Name: road_name, dtype: int64
---------------- 33.405319 -------------------
산서로    4821
-      4159
Name: road_name, dtype: int64
---------------- 33.322018 -------------------
서중2교    5198
-       2396
Name: road_name, dtype: int64
---------------- 33.325096 -------------------
중문로    5187
-      5187
Name: road_name, dtype: int64
---------------- 33.408431 -------------------
-      8441
산서로    4777
Name: road_name, dtype: int64
---------------- 33.284189 -------------------
중문로    5306
-      5288
Name: road_name, dtype: int64
---------------- 33.47339 -------------------
-      5344
양계장    5329
Name: road_name, dtype: int64
```

```python
print(len(train[train["road_name"] == "-"]))

train.loc[(train["start_latitude"] == 33.409416) & (train["road_name"] == "-"), "road_name"] = "월계교"
train.loc[(train["start_latitude"] == 33.402546) & (train["road_name"] == "-"), "road_name"] = "지방도1119호선"
train.loc[(train["start_latitude"] == 33.471164) & (train["road_name"] == "-"), "road_name"] = "일반국도12호선"
train.loc[(train["start_latitude"] == 33.411255) & (train["road_name"] == "-"), "road_name"] = "월계교"
train.loc[(train["start_latitude"] == 33.405319) & (train["road_name"] == "-"), "road_name"] = "산서로"
train.loc[(train["start_latitude"] == 33.322018) & (train["road_name"] == "-"), "road_name"] = "서중2교"
train.loc[(train["start_latitude"] == 33.325096) & (train["road_name"] == "-"), "road_name"] = "중문로"
train.loc[(train["start_latitude"] == 33.408431) & (train["road_name"] == "-"), "road_name"] = "산서로"
train.loc[(train["start_latitude"] == 33.284189) & (train["road_name"] == "-"), "road_name"] = "중문로"
train.loc[(train["start_latitude"] == 33.47339) & (train["road_name"] == "-"), "road_name"] = "양계장"

test.loc[(test["start_latitude"] == 33.409416) & (test["road_name"] == "-"), "road_name"] = "월계교"
test.loc[(test["start_latitude"] == 33.402546) & (test["road_name"] == "-"), "road_name"] = "지방도1119호선"
test.loc[(test["start_latitude"] == 33.471164) & (test["road_name"] == "-"), "road_name"] = "일반국도12호선"
test.loc[(test["start_latitude"] == 33.411255) & (test["road_name"] == "-"), "road_name"] = "월계교"
test.loc[(test["start_latitude"] == 33.405319) & (test["road_name"] == "-"), "road_name"] = "산서로"
test.loc[(test["start_latitude"] == 33.322018) & (test["road_name"] == "-"), "road_name"] = "서중2교"
test.loc[(test["start_latitude"] == 33.325096) & (test["road_name"] == "-"), "road_name"] = "중문로"
test.loc[(test["start_latitude"] == 33.408431) & (test["road_name"] == "-"), "road_name"] = "산서로"
test.loc[(test["start_latitude"] == 33.284189) & (test["road_name"] == "-"), "road_name"] = "중문로"
test.loc[(test["start_latitude"] == 33.47339) & (test["road_name"] == "-"), "road_name"] = "양계장"

print(len(train[train["road_name"] == "-"]))
```

```
282684
232879
```

- 약 5만개의 데이터가 대체
- 이어서 나머지 위도, 경도에 대해서 대체할 값 탐색

```python
# "-" 값 대체를 위한 탐색"
for i in train["end_latitude"].unique():
    if (len(train[(train["end_latitude"] == i)]["road_name"].value_counts()) != 2) :
        continue
    if "-" in train[(train["end_latitude"] == i)]["road_name"].value_counts().index:
        print("----------------", i, "-------------------")
        print(train[(train["end_latitude"] == i)]["road_name"].value_counts())
```

```
---------------- 33.47339 -------------------
-           5338
일반국도12호선    5334
Name: road_name, dtype: int64
---------------- 33.358358 -------------------
-           4784
일반국도16호선    2251
Name: road_name, dtype: int64
---------------- 33.412573 -------------------
-      4389
월계교    4199
Name: road_name, dtype: int64
---------------- 33.244882 -------------------
-      5528
산서로    5415
Name: road_name, dtype: int64
---------------- 33.322018 -------------------
중문로    5187
-      2493
Name: road_name, dtype: int64
```

```python
train.loc[(train["end_latitude"] == 33.47339) & (train["road_name"] == "-"), "road_name"] = "일반국도12호선"
train.loc[(train["end_latitude"] == 33.358358) & (train["road_name"] == "-"), "road_name"] = "일반국도16호선"
train.loc[(train["end_latitude"] == 33.412573) & (train["road_name"] == "-"), "road_name"] = "월계교"
train.loc[(train["end_latitude"] == 33.244882) & (train["road_name"] == "-"), "road_name"] = "산서로"
train.loc[(train["end_latitude"] == 33.322018) & (train["road_name"] == "-"), "road_name"] = "중문로"

test.loc[(test["end_latitude"] == 33.47339) & (test["road_name"] == "-"), "road_name"] = "일반국도12호선"
test.loc[(test["end_latitude"] == 33.358358) & (test["road_name"] == "-"), "road_name"] = "일반국도16호선"
test.loc[(test["end_latitude"] == 33.412573) & (test["road_name"] == "-"), "road_name"] = "월계교"
test.loc[(test["end_latitude"] == 33.244882) & (test["road_name"] == "-"), "road_name"] = "산서로"
test.loc[(test["end_latitude"] == 33.322018) & (test["road_name"] == "-"), "road_name"] = "중문로"
```

```python
# "-" 값 대체를 위한 탐색"
for i in train["start_longitude"].unique():
    if (len(train[(train["start_longitude"] == i)]["road_name"].value_counts()) != 2) :
        continue
    if "-" in train[(train["start_longitude"] == i)]["road_name"].value_counts().index:
        print("----------------", i, "-------------------")
        print(train[(train["start_longitude"] == i)]["road_name"].value_counts())
```

```
---------------- 126.259693 -------------------
월계교    4389
-      4223
Name: road_name, dtype: int64
```

```python
train.loc[(train["start_longitude"] == 126.259693) & (train["road_name"] == "-"), "road_name"] = "월계교"

test.loc[(test["start_longitude"] == 126.259693) & (test["road_name"] == "-"), "road_name"] = "월계교"
```

```python
# "-" 값 대체를 위한 탐색"
for i in train["end_longitude"].unique():
    if (len(train[(train["end_longitude"] == i)]["road_name"].value_counts()) != 2) :
        continue
    if "-" in train[(train["end_longitude"] == i)]["road_name"].value_counts().index:
        print("----------------", i, "-------------------")
        print(train[(train["end_longitude"] == i)]["road_name"].value_counts())
```

```
---------------- 126.261797 -------------------
-      4438
월계교    4223
Name: road_name, dtype: int64
```

```python
train.loc[(train["end_longitude"] == 126.261797) & (train["road_name"] == "-"), "road_name"] = "월계교"

test.loc[(test["end_longitude"] == 126.261797) & (test["road_name"] == "-"), "road_name"] = "월계교"
```

```python
print(len(train[train["road_name"] == "-"]))
```

```
201686
```

- "-" 값이 569463에서 201686까지 감소
- 이번에는 두 개 이상의 항목에 대해 탐색

```python
# 추가 탐색 - 종료지점을 중심으로
temp_train = train.groupby(["end_longitude", "end_latitude", "lane_count"])[["road_name"]].sum()
temp_train

temp_train1 = temp_train.agg({"road_name": pd.Series.mode})
temp_train1

long_lat = []

for i in range(len(temp_train1)):
    if "-" in temp_train1["road_name"].iloc[i][0]:
        #print(temp_train1.index[i])
        long_lat.append(temp_train1.index[i])

for i in range(len(long_lat)):
    if len(train[(train["end_longitude"] == long_lat[i][0]) & (train["end_latitude"] == long_lat[i][1])]["road_name"].value_counts()) > 1:
        print(train[(train["end_longitude"] == long_lat[i][0]) & (train["end_latitude"] == long_lat[i][1]) & (train["lane_count"] == long_lat[i][2])]["road_name"].value_counts())
        print(long_lat[i][0], long_lat[i][1], long_lat[i][2])
        print("------------------------------")
```

```python
train.loc[(train["end_longitude"] == 126.414236) & (train["end_latitude"] == 33.255215) & (train["lane_count"] == 2) & (train["road_name"] == "-"), "road_name"] = "중문로"
train.loc[(train["end_longitude"] == 126.456384) & (train["end_latitude"] == 33.465863) & (train["lane_count"] == 2) & (train["road_name"] == "-"), "road_name"] = "애조로"

test.loc[(test["end_longitude"] == 126.414236) & (test["end_latitude"] == 33.255215) & (test["lane_count"] == 2) & (test["road_name"] == "-"), "road_name"] = "중문로"
test.loc[(test["end_longitude"] == 126.456384) & (test["end_latitude"] == 33.465863) & (test["lane_count"] == 2) & (test["road_name"] == "-"), "road_name"] = "애조로"
```

```python
# 추가 탐색 - 시작지점을 중심으로
temp_train = train.groupby(["start_longitude", "start_latitude", "lane_count"])[["road_name"]].sum()
temp_train

temp_train1 = temp_train.agg({"road_name": pd.Series.mode})
temp_train1

long_lat = []

for i in range(len(temp_train1)):
    if "-" in temp_train1["road_name"].iloc[i][0]:
        #print(temp_train1.index[i])
        long_lat.append(temp_train1.index[i])

for i in range(len(long_lat)):
    if len(train[(train["start_longitude"] == long_lat[i][0]) & (train["start_latitude"] == long_lat[i][1])]["road_name"].value_counts()) > 1:
        print(train[(train["start_longitude"] == long_lat[i][0]) & (train["start_latitude"] == long_lat[i][1]) & (train["lane_count"] == long_lat[i][2])]["road_name"].value_counts())
        print(long_lat[i][0], long_lat[i][1], long_lat[i][2])
        print("------------------------------")
```

```python
train.loc[(train["start_longitude"] == 126.262739) & (train["start_latitude"] == 33.415854) & (train["lane_count"] == 2) & (train["road_name"] == "-"), "road_name"] = "월계교"
train.loc[(train["start_longitude"] == 126.413687) & (train["start_latitude"] == 33.255431) & (train["lane_count"] == 2) & (train["road_name"] == "-"), "road_name"] = "중문로"
train.loc[(train["start_longitude"] == 126.454583) & (train["start_latitude"] == 33.466433) & (train["lane_count"] == 2) & (train["road_name"] == "-"), "road_name"] = "애조로"
train.loc[(train["start_longitude"] == 126.456384) & (train["start_latitude"] == 33.465863) & (train["lane_count"] == 2) & (train["road_name"] == "-"), "road_name"] = "애조로"

test.loc[(test["start_longitude"] == 126.262739) & (test["start_latitude"] == 33.415854) & (test["lane_count"] == 2) & (test["road_name"] == "-"), "road_name"] = "월계교"
test.loc[(test["start_longitude"] == 126.413687) & (test["start_latitude"] == 33.255431) & (test["lane_count"] == 2) & (test["road_name"] == "-"), "road_name"] = "중문로"
test.loc[(test["start_longitude"] == 126.454583) & (test["start_latitude"] == 33.466433) & (test["lane_count"] == 2) & (test["road_name"] == "-"), "road_name"] = "애조로"
test.loc[(test["start_longitude"] == 126.456384) & (test["start_latitude"] == 33.465863) & (test["lane_count"] == 2) & (test["road_name"] == "-"), "road_name"] = "애조로"
```

```python
print(len(train[train["road_name"] == "-"]))
```

```
167420
```

- 최종적으로 16.7만개의 "-" 값이 남았습니다

### season 컬럼 추가

- 우리는 8월의 속도(target)를 예측하여야 합니다.
- train에는 8월 데이터가 주어지지 않습니다.
- 평균적인 속도는 인접한 달과 가장 속도가 비슷할 것으로 예상됩니다.
- 8월과 인접한 달인 7월과 9월을 표기하겠습니다.

```python
# 예측해야하는 8월을 기준으로 전 후 한 달을 컬럼으로 지정하겠습니다
def add_season(x):
    if x == 1:
        season = "not_7_to_9"
    elif x == 2:
        season= "not_7_to_9"
    elif x == 3:
        season= "not_7_to_9"
    elif x == 4:
        season= "not_7_to_9"
    elif x == 5:
        season= "not_7_to_9"
    elif x == 6:
        season= "not_7_to_9"
    elif x == 7:
        season= "seven_to_nine"
    elif x == 8:
        season= "seven_to_nine"
    elif x == 9:
        season= "seven_to_nine"
    elif x == 10:
        season= "not_7_to_9"
    elif x == 11:
        season= "not_7_to_9"
    elif x == 12:
        season= "not_7_to_9"
    else:
        season = "not_7_to_9"
    return season
```

```python
# 슬라이싱을 위해 base_date의 dtype을 string으로 변경합니다
train["base_date"] = train["base_date"].astype(str)
test["base_date"] = test["base_date"].astype(str)

# 임시로 month 컬럼을 생성합니다
train["month"] = train["base_date"].str[4:6].astype("int32")
test["month"] = test["base_date"].str[4:6].astype("int32")

# 7~9월일경우와 아닌경우 구분하기 위한 컬럼 season 생성
train["season"] = train["month"].apply(add_season)
test["season"] = test["month"].apply(add_season)

# base_date를 다시 int형으로 되돌립니다.
train["base_date"] = train["base_date"].astype("int32")
test["base_date"] = test["base_date"].astype("int32")
```

```python
train.drop("month", axis = 1, inplace = True)
test.drop("month", axis = 1, inplace = True)
```

### 근무시간을 기준으로 나누기 (8 ~ 20시), (21시 ~ 7시)

```python
# 주말에는 근무를 하지 않는 곳이 많지만 어딘가에 가는 것도 휴식시간이 아닌 경우가 있기 때문에 일괄적인 시간을 기준으로 값을 나눴습니다.
def set_binned_time(x):
    if 8 <= x <= 20:
        time = "worktime"
    elif x >= 21:
        time = "resttime"
    elif x <=7:
        time = "resttime"
    else: # 0 ~ 24 이외에 다른 값이 적용된 경우
        time = None
    return time
```

```python
train["work_or_rest_or_other"] = train["base_hour"].apply(set_binned_time)
test["work_or_rest_or_other"] = test["base_hour"].apply(set_binned_time)
```

```python
train.work_or_rest_or_other.value_counts()
```

```
worktime    2716112
resttime    1985105
Name: work_or_rest_or_other, dtype: int64
```

- 한 가지 값이 아닌 각각의 값으로 대체가 되었습니다.

### target 속도 100km/h 이상 이상치 제거

- 속도 EDA부분을 봤을 때 최고 제한속도가 80km/h이었지만 이상치 중에서도 서로 비슷한 부분이 존재하였습니다.
- 따라서 이상치는 그대로 두고 차이가 심하게나는 극단치(100km/h 이상)만을 제거하겠습니다.

```python
len(train)
```

```
4701217
```

```python
train = train[train.target<100]
```

```python
len(train)
```

```
4701212
```

- 5건의 데이터가 감소하였습니다.

### 라벨인코더

```python
# 범주형 데이터에 라벨인코더 사용
str_col = ["day_of_week", "road_name", "start_node_name", "end_node_name",
           "start_turn_restricted", "end_turn_restricted", "weight_restricted", "road_rating",
           "road_type", "season", "work_or_rest_or_other"]

for i in str_col:
    le = LabelEncoder()
    le=le.fit(train[i])
    train[i]=le.transform(train[i])

    for label in np.unique(test[i]):
        if label not in le.classes_: 
            le.classes_ = np.append(le.classes_, label)
    test[i]=le.transform(test[i])
```

```python
train.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>base_date</th>
      <th>day_of_week</th>
      <th>base_hour</th>
      <th>lane_count</th>
      <th>road_rating</th>
      <th>road_name</th>
      <th>multi_linked</th>
      <th>connect_code</th>
      <th>maximum_speed_limit</th>
      <th>...</th>
      <th>airport_distance</th>
      <th>center_start</th>
      <th>center_end</th>
      <th>road_code</th>
      <th>road_min</th>
      <th>road_mean</th>
      <th>road_max</th>
      <th>road_std</th>
      <th>season</th>
      <th>work_or_rest_or_other</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>TRAIN_0000000</td>
      <td>20220623</td>
      <td>1</td>
      <td>17</td>
      <td>1</td>
      <td>1</td>
      <td>50</td>
      <td>0</td>
      <td>0</td>
      <td>60.0</td>
      <td>...</td>
      <td>18.330548</td>
      <td>0</td>
      <td>0</td>
      <td>33.427747274683_126.662612038652_33.4277487730...</td>
      <td>16.0</td>
      <td>51.756910</td>
      <td>72.0</td>
      <td>4.587047</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>TRAIN_0000001</td>
      <td>20220728</td>
      <td>1</td>
      <td>21</td>
      <td>2</td>
      <td>0</td>
      <td>38</td>
      <td>0</td>
      <td>0</td>
      <td>60.0</td>
      <td>...</td>
      <td>3.470872</td>
      <td>1</td>
      <td>1</td>
      <td>33.5007304293026_126.529106761554_33.504811303...</td>
      <td>5.0</td>
      <td>26.400712</td>
      <td>59.0</td>
      <td>7.102290</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>TRAIN_0000002</td>
      <td>20211010</td>
      <td>4</td>
      <td>7</td>
      <td>2</td>
      <td>0</td>
      <td>40</td>
      <td>0</td>
      <td>0</td>
      <td>80.0</td>
      <td>...</td>
      <td>28.185912</td>
      <td>0</td>
      <td>0</td>
      <td>33.2791450972975_126.368597660936_33.280072104...</td>
      <td>32.0</td>
      <td>59.101720</td>
      <td>88.0</td>
      <td>12.091252</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>TRAIN_0000003</td>
      <td>20220311</td>
      <td>0</td>
      <td>13</td>
      <td>2</td>
      <td>2</td>
      <td>61</td>
      <td>0</td>
      <td>0</td>
      <td>50.0</td>
      <td>...</td>
      <td>30.222870</td>
      <td>2</td>
      <td>2</td>
      <td>33.2460808686345_126.56720431031_33.2455654004...</td>
      <td>2.0</td>
      <td>25.024923</td>
      <td>51.0</td>
      <td>7.667545</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>TRAIN_0000004</td>
      <td>20211005</td>
      <td>6</td>
      <td>8</td>
      <td>2</td>
      <td>0</td>
      <td>39</td>
      <td>0</td>
      <td>0</td>
      <td>80.0</td>
      <td>...</td>
      <td>16.019878</td>
      <td>0</td>
      <td>0</td>
      <td>33.4622143482158_126.326551111199_33.462676772...</td>
      <td>25.0</td>
      <td>39.873670</td>
      <td>72.0</td>
      <td>6.946840</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 34 columns</p>
</div>

### 데이터 자료형 변형 2

```python
train.info()
```

```
<class 'pandas.core.frame.DataFrame'>
Int64Index: 4701212 entries, 0 to 4701216
Data columns (total 34 columns):
 #   Column                 Dtype  
---  ------                 -----  
 0   id                     object 
 1   base_date              int32  
 2   day_of_week            int32  
 3   base_hour              int64  
 4   lane_count             int64  
 5   road_rating            int64  
 6   road_name              int32  
 7   multi_linked           int64  
 8   connect_code           int64  
 9   maximum_speed_limit    float64
 10  weight_restricted      int64  
 11  road_type              int64  
 12  start_node_name        int32  
 13  start_latitude         float64
 14  start_longitude        float64
 15  start_turn_restricted  int32  
 16  end_node_name          int32  
 17  end_latitude           float64
 18  end_longitude          float64
 19  end_turn_restricted    int32  
 20  target                 float64
 21  lat_change             float64
 22  lon_change             float64
 23  distance               float64
 24  airport_distance       float64
 25  center_start           int64  
 26  center_end             int64  
 27  road_code              object 
 28  road_min               float64
 29  road_mean              float64
 30  road_max               float64
 31  road_std               float64
 32  season                 int32  
 33  work_or_rest_or_other  int32  
dtypes: float64(14), int32(9), int64(9), object(2)
memory usage: 1.1+ GB
```

- 라벨인코딩을 하면서 몇몇 컬럼의 자료형이 변형되었습니다. 해당 컬럼을 resize 하겠습니다

```python
Y_train = train['target']
X_train = train.drop(["id", "multi_linked", "connect_code", "target"], axis = 1)
X_test = test.drop(["id", "multi_linked", "connect_code"], axis = 1)

# 데이터 램 사용량을 감소시키기 위해 더이상 필요하지 않은 데이터는 제거합니다
del train
del test
gc.collect()

## 상관관계 확인

### 확인을 위해 타겟 컬럼 추가

X_train['tar'] = Y_train

X_train.corr()

train_corr = X_train.corr()
f, ax = plt.subplots(figsize=(16, 12))
sns.heatmap(train_corr, annot=True, fmt = '.2f', square=True)

### 예측에 방해되는 컬럼 삭제

X_train.drop(['tar', 'road_code'], axis=1, inplace=True)

X_test.drop(['road_code'], axis=1, inplace=True)to_int8 = ["day_of_week","weight_restricted", "base_hour", "lane_count", "road_rating", 
           "road_name","road_type", "start_turn_restricted", "end_turn_restricted", 
           "maximum_speed_limit", "season", "work_or_rest_or_other"]
to_int16 = ["start_node_name", "end_node_name"]
to_int32 = ["base_date"]

for i in to_int8:
    train[i] = train[i].astype("int8")
for j in to_int16:
    train[j] = train[j].astype("int16")
for k in to_int32:
    train[k] = train[k].astype("int32")
```

```python
train.info()
```

```
<class 'pandas.core.frame.DataFrame'>
Int64Index: 4701212 entries, 0 to 4701216
Data columns (total 34 columns):
 #   Column                 Dtype  
---  ------                 -----  
 0   id                     object 
 1   base_date              int32  
 2   day_of_week            int8   
 3   base_hour              int8   
 4   lane_count             int8   
 5   road_rating            int8   
 6   road_name              int8   
 7   multi_linked           int64  
 8   connect_code           int64  
 9   maximum_speed_limit    int8   
 10  weight_restricted      int8   
 11  road_type              int8   
 12  start_node_name        int16  
 13  start_latitude         float64
 14  start_longitude        float64
 15  start_turn_restricted  int8   
 16  end_node_name          int16  
 17  end_latitude           float64
 18  end_longitude          float64
 19  end_turn_restricted    int8   
 20  target                 float64
 21  lat_change             float64
 22  lon_change             float64
 23  distance               float64
 24  airport_distance       float64
 25  center_start           int64  
 26  center_end             int64  
 27  road_code              object 
 28  road_min               float64
 29  road_mean              float64
 30  road_max               float64
 31  road_std               float64
 32  season                 int8   
 33  work_or_rest_or_other  int8   
dtypes: float64(13), int16(2), int32(1), int64(4), int8(12), object(2)
memory usage: 807.0+ MB
```

#### 자료형이 감소되었고, 메모리량이 크게 줄은 것을 볼 수 있습니다.

```python
Y_train = train['target']
X_train = train.drop(["id", "multi_linked", "connect_code", "target"], axis = 1)
X_test = test.drop(["id", "multi_linked", "connect_code"], axis = 1)
```

```python
# 데이터 램 사용량을 감소시키기 위해 더이상 필요하지 않은 데이터는 제거합니다
del train
del test
gc.collect()
```

```
706
```

## 전처리가 끝난 데이터의 상관관계 확인

### 확인을 위해 타겟 컬럼 추가

```python
X_train['tar'] = Y_train
```

```python
X_train.corr()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>base_date</th>
      <th>day_of_week</th>
      <th>base_hour</th>
      <th>lane_count</th>
      <th>road_rating</th>
      <th>road_name</th>
      <th>maximum_speed_limit</th>
      <th>weight_restricted</th>
      <th>road_type</th>
      <th>start_node_name</th>
      <th>...</th>
      <th>airport_distance</th>
      <th>center_start</th>
      <th>center_end</th>
      <th>road_min</th>
      <th>road_mean</th>
      <th>road_max</th>
      <th>road_std</th>
      <th>season</th>
      <th>work_or_rest_or_other</th>
      <th>tar</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>base_date</th>
      <td>1.000000</td>
      <td>0.036289</td>
      <td>-0.008645</td>
      <td>0.011463</td>
      <td>0.021064</td>
      <td>-0.015595</td>
      <td>-0.018713</td>
      <td>-0.011119</td>
      <td>-0.004599</td>
      <td>0.002393</td>
      <td>...</td>
      <td>-0.005890</td>
      <td>0.041208</td>
      <td>0.041606</td>
      <td>-0.023060</td>
      <td>-0.030693</td>
      <td>-0.025759</td>
      <td>0.010305</td>
      <td>-0.269214</td>
      <td>-0.010182</td>
      <td>-0.033996</td>
    </tr>
    <tr>
      <th>day_of_week</th>
      <td>0.036289</td>
      <td>1.000000</td>
      <td>0.004889</td>
      <td>0.001426</td>
      <td>-0.002440</td>
      <td>0.001154</td>
      <td>0.001651</td>
      <td>0.000206</td>
      <td>0.000429</td>
      <td>0.000105</td>
      <td>...</td>
      <td>-0.001535</td>
      <td>-0.000474</td>
      <td>-0.000473</td>
      <td>0.001162</td>
      <td>0.001664</td>
      <td>0.001719</td>
      <td>0.000825</td>
      <td>-0.015661</td>
      <td>0.005468</td>
      <td>0.006397</td>
    </tr>
    <tr>
      <th>base_hour</th>
      <td>-0.008645</td>
      <td>0.004889</td>
      <td>1.000000</td>
      <td>-0.029195</td>
      <td>0.034073</td>
      <td>-0.024981</td>
      <td>-0.036757</td>
      <td>-0.004205</td>
      <td>-0.007881</td>
      <td>0.004756</td>
      <td>...</td>
      <td>0.028379</td>
      <td>0.006347</td>
      <td>0.006322</td>
      <td>-0.025417</td>
      <td>-0.024616</td>
      <td>-0.033581</td>
      <td>-0.016198</td>
      <td>0.000833</td>
      <td>0.363051</td>
      <td>-0.159402</td>
    </tr>
    <tr>
      <th>lane_count</th>
      <td>0.011463</td>
      <td>0.001426</td>
      <td>-0.029195</td>
      <td>1.000000</td>
      <td>-0.088097</td>
      <td>-0.105303</td>
      <td>0.384002</td>
      <td>-0.158700</td>
      <td>-0.050715</td>
      <td>-0.061336</td>
      <td>...</td>
      <td>-0.333835</td>
      <td>0.244378</td>
      <td>0.244660</td>
      <td>-0.190792</td>
      <td>-0.160188</td>
      <td>-0.028263</td>
      <td>0.388710</td>
      <td>0.038643</td>
      <td>-0.039087</td>
      <td>-0.144255</td>
    </tr>
    <tr>
      <th>road_rating</th>
      <td>0.021064</td>
      <td>-0.002440</td>
      <td>0.034073</td>
      <td>-0.088097</td>
      <td>1.000000</td>
      <td>-0.318527</td>
      <td>-0.351835</td>
      <td>-0.125360</td>
      <td>-0.147249</td>
      <td>-0.012973</td>
      <td>...</td>
      <td>0.139860</td>
      <td>0.313372</td>
      <td>0.314694</td>
      <td>-0.272933</td>
      <td>-0.344537</td>
      <td>-0.265808</td>
      <td>-0.083340</td>
      <td>0.017420</td>
      <td>0.037959</td>
      <td>-0.310354</td>
    </tr>
    <tr>
      <th>road_name</th>
      <td>-0.015595</td>
      <td>0.001154</td>
      <td>-0.024981</td>
      <td>-0.105303</td>
      <td>-0.318527</td>
      <td>1.000000</td>
      <td>0.165001</td>
      <td>0.110760</td>
      <td>0.134940</td>
      <td>0.062366</td>
      <td>...</td>
      <td>0.016637</td>
      <td>-0.165926</td>
      <td>-0.164566</td>
      <td>0.278053</td>
      <td>0.241767</td>
      <td>0.158177</td>
      <td>-0.063775</td>
      <td>-0.020662</td>
      <td>-0.027557</td>
      <td>0.217723</td>
    </tr>
    <tr>
      <th>maximum_speed_limit</th>
      <td>-0.018713</td>
      <td>0.001651</td>
      <td>-0.036757</td>
      <td>0.384002</td>
      <td>-0.351835</td>
      <td>0.165001</td>
      <td>1.000000</td>
      <td>0.102248</td>
      <td>0.059511</td>
      <td>-0.031339</td>
      <td>...</td>
      <td>-0.233143</td>
      <td>-0.288147</td>
      <td>-0.289071</td>
      <td>0.284929</td>
      <td>0.472733</td>
      <td>0.502248</td>
      <td>0.198350</td>
      <td>-0.021150</td>
      <td>-0.044174</td>
      <td>0.425720</td>
    </tr>
    <tr>
      <th>weight_restricted</th>
      <td>-0.011119</td>
      <td>0.000206</td>
      <td>-0.004205</td>
      <td>-0.158700</td>
      <td>-0.125360</td>
      <td>0.110760</td>
      <td>0.102248</td>
      <td>1.000000</td>
      <td>0.758760</td>
      <td>0.092265</td>
      <td>...</td>
      <td>0.101060</td>
      <td>-0.150612</td>
      <td>-0.151004</td>
      <td>0.267590</td>
      <td>0.330382</td>
      <td>0.242250</td>
      <td>-0.225760</td>
      <td>-0.017357</td>
      <td>-0.004312</td>
      <td>0.297525</td>
    </tr>
    <tr>
      <th>road_type</th>
      <td>-0.004599</td>
      <td>0.000429</td>
      <td>-0.007881</td>
      <td>-0.050715</td>
      <td>-0.147249</td>
      <td>0.134940</td>
      <td>0.059511</td>
      <td>0.758760</td>
      <td>1.000000</td>
      <td>0.052607</td>
      <td>...</td>
      <td>-0.012011</td>
      <td>-0.056679</td>
      <td>-0.057253</td>
      <td>0.220557</td>
      <td>0.223022</td>
      <td>0.127485</td>
      <td>-0.136686</td>
      <td>0.005911</td>
      <td>-0.007306</td>
      <td>0.200842</td>
    </tr>
    <tr>
      <th>start_node_name</th>
      <td>0.002393</td>
      <td>0.000105</td>
      <td>0.004756</td>
      <td>-0.061336</td>
      <td>-0.012973</td>
      <td>0.062366</td>
      <td>-0.031339</td>
      <td>0.092265</td>
      <td>0.052607</td>
      <td>1.000000</td>
      <td>...</td>
      <td>0.040150</td>
      <td>-0.017202</td>
      <td>-0.019782</td>
      <td>-0.000653</td>
      <td>0.032211</td>
      <td>0.023749</td>
      <td>-0.015654</td>
      <td>0.009526</td>
      <td>0.005293</td>
      <td>0.029007</td>
    </tr>
    <tr>
      <th>start_latitude</th>
      <td>-0.016819</td>
      <td>0.001201</td>
      <td>-0.021601</td>
      <td>0.182674</td>
      <td>-0.239376</td>
      <td>0.041744</td>
      <td>0.253147</td>
      <td>-0.122722</td>
      <td>-0.043421</td>
      <td>-0.034024</td>
      <td>...</td>
      <td>-0.718825</td>
      <td>-0.531331</td>
      <td>-0.530671</td>
      <td>0.050985</td>
      <td>0.040287</td>
      <td>0.077875</td>
      <td>0.193520</td>
      <td>-0.025959</td>
      <td>-0.025108</td>
      <td>0.036285</td>
    </tr>
    <tr>
      <th>start_longitude</th>
      <td>-0.004954</td>
      <td>0.000656</td>
      <td>-0.011480</td>
      <td>-0.094807</td>
      <td>-0.026750</td>
      <td>0.167376</td>
      <td>-0.033018</td>
      <td>0.035122</td>
      <td>0.033683</td>
      <td>0.059345</td>
      <td>...</td>
      <td>0.258119</td>
      <td>-0.043973</td>
      <td>-0.043549</td>
      <td>0.067001</td>
      <td>-0.001296</td>
      <td>0.100479</td>
      <td>-0.008058</td>
      <td>-0.007763</td>
      <td>-0.013676</td>
      <td>-0.001164</td>
    </tr>
    <tr>
      <th>start_turn_restricted</th>
      <td>0.010650</td>
      <td>0.000879</td>
      <td>-0.012476</td>
      <td>0.292852</td>
      <td>-0.042450</td>
      <td>-0.037161</td>
      <td>0.094124</td>
      <td>-0.113353</td>
      <td>-0.103950</td>
      <td>0.010010</td>
      <td>...</td>
      <td>-0.358621</td>
      <td>0.051013</td>
      <td>0.062162</td>
      <td>-0.129606</td>
      <td>-0.173363</td>
      <td>-0.065905</td>
      <td>0.212830</td>
      <td>0.023035</td>
      <td>-0.015187</td>
      <td>-0.156121</td>
    </tr>
    <tr>
      <th>end_node_name</th>
      <td>0.003071</td>
      <td>0.000212</td>
      <td>0.005336</td>
      <td>-0.071248</td>
      <td>-0.012834</td>
      <td>0.046132</td>
      <td>-0.051494</td>
      <td>0.093010</td>
      <td>0.053671</td>
      <td>0.310048</td>
      <td>...</td>
      <td>0.045705</td>
      <td>-0.020176</td>
      <td>-0.018015</td>
      <td>0.011931</td>
      <td>0.044162</td>
      <td>0.009725</td>
      <td>-0.045193</td>
      <td>0.009087</td>
      <td>0.005962</td>
      <td>0.039770</td>
    </tr>
    <tr>
      <th>end_latitude</th>
      <td>-0.016787</td>
      <td>0.001214</td>
      <td>-0.021599</td>
      <td>0.182330</td>
      <td>-0.239391</td>
      <td>0.042024</td>
      <td>0.252958</td>
      <td>-0.122735</td>
      <td>-0.043431</td>
      <td>-0.035139</td>
      <td>...</td>
      <td>-0.718883</td>
      <td>-0.531304</td>
      <td>-0.530672</td>
      <td>0.052240</td>
      <td>0.040131</td>
      <td>0.078685</td>
      <td>0.194742</td>
      <td>-0.025964</td>
      <td>-0.025080</td>
      <td>0.036144</td>
    </tr>
    <tr>
      <th>end_longitude</th>
      <td>-0.004972</td>
      <td>0.000664</td>
      <td>-0.011490</td>
      <td>-0.094732</td>
      <td>-0.026748</td>
      <td>0.167097</td>
      <td>-0.032907</td>
      <td>0.035113</td>
      <td>0.033663</td>
      <td>0.058852</td>
      <td>...</td>
      <td>0.258069</td>
      <td>-0.043901</td>
      <td>-0.043743</td>
      <td>0.066236</td>
      <td>-0.001110</td>
      <td>0.101249</td>
      <td>-0.007092</td>
      <td>-0.007755</td>
      <td>-0.013704</td>
      <td>-0.000996</td>
    </tr>
    <tr>
      <th>end_turn_restricted</th>
      <td>0.010630</td>
      <td>0.000949</td>
      <td>-0.012589</td>
      <td>0.312854</td>
      <td>-0.042849</td>
      <td>-0.036726</td>
      <td>0.104737</td>
      <td>-0.113210</td>
      <td>-0.103781</td>
      <td>0.004197</td>
      <td>...</td>
      <td>-0.358610</td>
      <td>0.059147</td>
      <td>0.053905</td>
      <td>-0.118479</td>
      <td>-0.180797</td>
      <td>-0.060201</td>
      <td>0.233035</td>
      <td>0.023179</td>
      <td>-0.015190</td>
      <td>-0.162816</td>
    </tr>
    <tr>
      <th>lat_change</th>
      <td>-0.000785</td>
      <td>-0.000313</td>
      <td>-0.000075</td>
      <td>0.008422</td>
      <td>-0.000022</td>
      <td>-0.006541</td>
      <td>0.004866</td>
      <td>0.000119</td>
      <td>0.000166</td>
      <td>0.026314</td>
      <td>...</td>
      <td>0.000265</td>
      <td>-0.001468</td>
      <td>-0.000805</td>
      <td>-0.029602</td>
      <td>0.003766</td>
      <td>-0.019022</td>
      <td>-0.028591</td>
      <td>0.000089</td>
      <td>-0.000718</td>
      <td>0.003393</td>
    </tr>
    <tr>
      <th>lon_change</th>
      <td>0.000454</td>
      <td>-0.000190</td>
      <td>0.000231</td>
      <td>-0.002098</td>
      <td>-0.000111</td>
      <td>0.007435</td>
      <td>-0.002902</td>
      <td>0.000320</td>
      <td>0.000592</td>
      <td>0.012619</td>
      <td>...</td>
      <td>0.001842</td>
      <td>-0.001917</td>
      <td>0.004834</td>
      <td>0.019510</td>
      <td>-0.004726</td>
      <td>-0.019260</td>
      <td>-0.024457</td>
      <td>-0.000224</td>
      <td>0.000686</td>
      <td>-0.004252</td>
    </tr>
    <tr>
      <th>distance</th>
      <td>-0.014383</td>
      <td>0.000344</td>
      <td>0.003693</td>
      <td>-0.167181</td>
      <td>-0.005455</td>
      <td>0.104033</td>
      <td>0.052221</td>
      <td>-0.266097</td>
      <td>-0.317429</td>
      <td>-0.020303</td>
      <td>...</td>
      <td>0.082840</td>
      <td>-0.245167</td>
      <td>-0.244935</td>
      <td>0.265338</td>
      <td>0.200917</td>
      <td>0.156047</td>
      <td>-0.145724</td>
      <td>-0.031204</td>
      <td>0.007616</td>
      <td>0.180935</td>
    </tr>
    <tr>
      <th>airport_distance</th>
      <td>-0.005890</td>
      <td>-0.001535</td>
      <td>0.028379</td>
      <td>-0.333835</td>
      <td>0.139860</td>
      <td>0.016637</td>
      <td>-0.233143</td>
      <td>0.101060</td>
      <td>-0.012011</td>
      <td>0.040150</td>
      <td>...</td>
      <td>1.000000</td>
      <td>0.157809</td>
      <td>0.156847</td>
      <td>0.070417</td>
      <td>0.085211</td>
      <td>0.047291</td>
      <td>-0.255495</td>
      <td>-0.028384</td>
      <td>0.034434</td>
      <td>0.076734</td>
    </tr>
    <tr>
      <th>center_start</th>
      <td>0.041208</td>
      <td>-0.000474</td>
      <td>0.006347</td>
      <td>0.244378</td>
      <td>0.313372</td>
      <td>-0.165926</td>
      <td>-0.288147</td>
      <td>-0.150612</td>
      <td>-0.056679</td>
      <td>-0.017202</td>
      <td>...</td>
      <td>0.157809</td>
      <td>1.000000</td>
      <td>0.991306</td>
      <td>-0.401235</td>
      <td>-0.508221</td>
      <td>-0.439657</td>
      <td>0.139291</td>
      <td>0.087232</td>
      <td>0.001729</td>
      <td>-0.457674</td>
    </tr>
    <tr>
      <th>center_end</th>
      <td>0.041606</td>
      <td>-0.000473</td>
      <td>0.006322</td>
      <td>0.244660</td>
      <td>0.314694</td>
      <td>-0.164566</td>
      <td>-0.289071</td>
      <td>-0.151004</td>
      <td>-0.057253</td>
      <td>-0.019782</td>
      <td>...</td>
      <td>0.156847</td>
      <td>0.991306</td>
      <td>1.000000</td>
      <td>-0.401718</td>
      <td>-0.508520</td>
      <td>-0.440852</td>
      <td>0.143325</td>
      <td>0.086792</td>
      <td>0.001681</td>
      <td>-0.457944</td>
    </tr>
    <tr>
      <th>road_min</th>
      <td>-0.023060</td>
      <td>0.001162</td>
      <td>-0.025417</td>
      <td>-0.190792</td>
      <td>-0.272933</td>
      <td>0.278053</td>
      <td>0.284929</td>
      <td>0.267590</td>
      <td>0.220557</td>
      <td>-0.000653</td>
      <td>...</td>
      <td>0.070417</td>
      <td>-0.401235</td>
      <td>-0.401718</td>
      <td>1.000000</td>
      <td>0.701707</td>
      <td>0.539085</td>
      <td>-0.338897</td>
      <td>-0.034924</td>
      <td>-0.032085</td>
      <td>0.631920</td>
    </tr>
    <tr>
      <th>road_mean</th>
      <td>-0.030693</td>
      <td>0.001664</td>
      <td>-0.024616</td>
      <td>-0.160188</td>
      <td>-0.344537</td>
      <td>0.241767</td>
      <td>0.472733</td>
      <td>0.330382</td>
      <td>0.223022</td>
      <td>0.032211</td>
      <td>...</td>
      <td>0.085211</td>
      <td>-0.508221</td>
      <td>-0.508520</td>
      <td>0.701707</td>
      <td>1.000000</td>
      <td>0.776071</td>
      <td>-0.334308</td>
      <td>-0.051998</td>
      <td>-0.028813</td>
      <td>0.900547</td>
    </tr>
    <tr>
      <th>road_max</th>
      <td>-0.025759</td>
      <td>0.001719</td>
      <td>-0.033581</td>
      <td>-0.028263</td>
      <td>-0.265808</td>
      <td>0.158177</td>
      <td>0.502248</td>
      <td>0.242250</td>
      <td>0.127485</td>
      <td>0.023749</td>
      <td>...</td>
      <td>0.047291</td>
      <td>-0.439657</td>
      <td>-0.440852</td>
      <td>0.539085</td>
      <td>0.776071</td>
      <td>1.000000</td>
      <td>0.022116</td>
      <td>-0.040112</td>
      <td>-0.040875</td>
      <td>0.698876</td>
    </tr>
    <tr>
      <th>road_std</th>
      <td>0.010305</td>
      <td>0.000825</td>
      <td>-0.016198</td>
      <td>0.388710</td>
      <td>-0.083340</td>
      <td>-0.063775</td>
      <td>0.198350</td>
      <td>-0.225760</td>
      <td>-0.136686</td>
      <td>-0.015654</td>
      <td>...</td>
      <td>-0.255495</td>
      <td>0.139291</td>
      <td>0.143325</td>
      <td>-0.338897</td>
      <td>-0.334308</td>
      <td>0.022116</td>
      <td>1.000000</td>
      <td>0.033017</td>
      <td>-0.019181</td>
      <td>-0.301065</td>
    </tr>
    <tr>
      <th>season</th>
      <td>-0.269214</td>
      <td>-0.015661</td>
      <td>0.000833</td>
      <td>0.038643</td>
      <td>0.017420</td>
      <td>-0.020662</td>
      <td>-0.021150</td>
      <td>-0.017357</td>
      <td>0.005911</td>
      <td>0.009526</td>
      <td>...</td>
      <td>-0.028384</td>
      <td>0.087232</td>
      <td>0.086792</td>
      <td>-0.034924</td>
      <td>-0.051998</td>
      <td>-0.040112</td>
      <td>0.033017</td>
      <td>1.000000</td>
      <td>0.001951</td>
      <td>-0.044089</td>
    </tr>
    <tr>
      <th>work_or_rest_or_other</th>
      <td>-0.010182</td>
      <td>0.005468</td>
      <td>0.363051</td>
      <td>-0.039087</td>
      <td>0.037959</td>
      <td>-0.027557</td>
      <td>-0.044174</td>
      <td>-0.004312</td>
      <td>-0.007306</td>
      <td>0.005293</td>
      <td>...</td>
      <td>0.034434</td>
      <td>0.001729</td>
      <td>0.001681</td>
      <td>-0.032085</td>
      <td>-0.028813</td>
      <td>-0.040875</td>
      <td>-0.019181</td>
      <td>0.001951</td>
      <td>1.000000</td>
      <td>-0.215662</td>
    </tr>
    <tr>
      <th>tar</th>
      <td>-0.033996</td>
      <td>0.006397</td>
      <td>-0.159402</td>
      <td>-0.144255</td>
      <td>-0.310354</td>
      <td>0.217723</td>
      <td>0.425720</td>
      <td>0.297525</td>
      <td>0.200842</td>
      <td>0.029007</td>
      <td>...</td>
      <td>0.076734</td>
      <td>-0.457674</td>
      <td>-0.457944</td>
      <td>0.631920</td>
      <td>0.900547</td>
      <td>0.698876</td>
      <td>-0.301065</td>
      <td>-0.044089</td>
      <td>-0.215662</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>30 rows × 30 columns</p>
</div>

```python
train_corr = X_train.corr()
f, ax = plt.subplots(figsize=(16, 12))
sns.heatmap(train_corr, annot=True, fmt = '.2f', square=True)
```

```
<AxesSubplot:>
```

![image](https://user-images.githubusercontent.com/84713532/206893235-ddf024a2-3319-4fd7-b5eb-6d465b4d4e2f.png)


### 예측에 방해되는 컬럼 삭제

```python
X_train.drop(['tar', 'road_code'], axis=1, inplace=True)
```

```python
X_test.drop(['road_code'], axis=1, inplace=True)
```

## 3. Modeling

### LinearRegression

### DecisionTreeRegressor

### Ridge

### Kfold + RandomForestRegressor

### Kford + ExtraTreesRegressor

#### 대회 규칙인 MAE으로 오차 계산

### LinearRegression

```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()

scores = cross_val_score(lin_reg, X_train, y_train, scoring = 'neg_mean_squared_error', cv = 5)
lin_reg_rmse = np.sqrt((-scores).mean())
lin_reg_rmse
```

```
12.58878925952928
```

### DecisionTreeRegressor

```python
from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()

tree_scores = cross_val_score(tree_reg, X_train, y_train, scoring="neg_mean_squared_error", cv=10, n_jobs=-1)

tree_rmse = np.sqrt(-tree_scores).mean()
tree_rmse
```

```
5.762942764000505
```

### Ridge

```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet

ridge = Ridge()
lasso = Lasso()
elastic = ElasticNet()
from sklearn.model_selection import GridSearchCV
```

```python
# 평가지표 계산 함수
from sklearn.metrics import make_scorer
# log 값 변환 시 언더플로우 영향으로 log() 가 아닌 log1p() 를 이용하여 RMSLE 계산
def rmsle(y, pred, convertExp=True):
    if convertExp:
        y = np.expm1(y)
        pred = np.expm1(pred)

    log_y = np.log1p(y)
    log_pred = np.log1p(pred)
    squared_error = (log_y - log_pred) ** 2
    rmsle = np.sqrt(np.mean(squared_error))
    return rmsle

rmsle_scorer = make_scorer(rmsle, greater_is_better=False)
```

```python
# 릿지모델
ridge = Ridge()
ridge_params = {'alpha' : [0.1, 1, 2, 3, 4, 10, 30, 100, 200, 300, 400, 800, 900, 1000]} # 14개
gridsearch_ridge = GridSearchCV(ridge, ridge_params, scoring=rmsle_scorer, cv=5, n_jobs=-1) # 14 * 5
```

```python
%time gridsearch_ridge.fit(X_train, y_train)
```

```
CPU times: total: 6.38 s
Wall time: 1min 20s





GridSearchCV(cv=5, estimator=Ridge(), n_jobs=-1,
             param_grid={'alpha': [0.1, 1, 2, 3, 4, 10, 30, 100, 200, 300, 400,
                                   800, 900, 1000]},
             scoring=make_scorer(rmsle, greater_is_better=False))
```

```python
gridsearch_ridge.best_params_
```

```
{'alpha': 0.1}
```

```python
cvres = gridsearch_ridge.cv_results_

for mean_score, params in zip(cvres['mean_test_score'], cvres['params']):
    print(-mean_score, params) # rmsle와 그 때의 하이퍼 파라미터
```

```
12.588786599362003 {'alpha': 0.1}
12.588786692168853 {'alpha': 1}
12.588787246469451 {'alpha': 2}
12.588788185941212 {'alpha': 3}
12.588789435047687 {'alpha': 4}
12.58880067305844 {'alpha': 10}
12.588843843630762 {'alpha': 30}
12.588922989673593 {'alpha': 100}
12.58896273188661 {'alpha': 200}
12.588980369917937 {'alpha': 300}
12.58899040688326 {'alpha': 400}
12.589008071609262 {'alpha': 800}
12.589010468154935 {'alpha': 900}
12.589012543748403 {'alpha': 1000}
```

### Kfold + RandomForestRegressor, ExtraTreesRegressor

```python
from sklearn.model_selection import StratifiedKFold as kfold
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error

# 연속적인 값인 만큼 stratify kfold 를 사용할 수 없지만 나누는 것은 kfold와 동일합니다.
kf = kfold(n_splits = 3, shuffle = True, random_state = 42)
split = kf.split(X_train, Y_train)

# 평균 mae를 확인하기 위한 리스트
mae_list1 = []
mae_list2 = []

# 폴드별 예측값 저장을 위한 리스트
test_pred_list1 = []
test_pred_list2 = []

for train, test in split:
    x_train, x_val, y_train, y_val = X_train.iloc[train], X_train.iloc[test], Y_train.iloc[train], Y_train.iloc[test]    

    rf = RandomForestRegressor(n_estimators=40, min_samples_leaf=10,
                                min_samples_split=10, random_state=2022)
    et = ExtraTreesRegressor(n_estimators = 40, min_samples_split=10, min_samples_leaf = 10, random_state = 2022)

    rf.fit(x_train, y_train)
    et.fit(x_train, y_train)

    pred1 = rf.predict(x_val)
    pred2 = et.predict(x_val)

    result1 = mean_absolute_error(pred1,y_val)
    result2 = mean_absolute_error(pred2,y_val)

    mae_list1.append(result1)
    mae_list2.append(result2)

    print(f'RandomForestRegressor mae : {result1:.4f}', end='\n\n')
    print(f'ExtraTreeRegressor mae : {result2:.4f}', end='\n\n')

    test_pred_list1.append(rf.predict(X_test))
    test_pred_list2.append(et.predict(X_test))

print(f'mean mae {np.mean(mae_list1):.4f}')
print(f'mean mae {np.mean(mae_list2):.4f}')
```

```
RandomForestRegressor mae : 2.9155

ExtraTreeRegressor mae : 2.9633

RandomForestRegressor mae : 2.9162

ExtraTreeRegressor mae : 2.9632

RandomForestRegressor mae : 2.9165

ExtraTreeRegressor mae : 2.9623

mean mae 2.9161
mean mae 2.9629
```

### 파일 저장

```python
sample_submission = pd.read_csv('./Dataset/sample_submission.csv')
```

```python
sample_submission
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }


</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>TEST_000000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>TEST_000001</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>TEST_000002</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>TEST_000003</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>TEST_000004</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>291236</th>
      <td>TEST_291236</td>
      <td>0</td>
    </tr>
    <tr>
      <th>291237</th>
      <td>TEST_291237</td>
      <td>0</td>
    </tr>
    <tr>
      <th>291238</th>
      <td>TEST_291238</td>
      <td>0</td>
    </tr>
    <tr>
      <th>291239</th>
      <td>TEST_291239</td>
      <td>0</td>
    </tr>
    <tr>
      <th>291240</th>
      <td>TEST_291240</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>291241 rows × 2 columns</p>
</div>

```python
sample_submission['target'] = pred_last
sample_submission.to_csv("./All_Process.csv", index = False)

sample_submission
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }


</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>TEST_000000</td>
      <td>25.651475</td>
    </tr>
    <tr>
      <th>1</th>
      <td>TEST_000001</td>
      <td>42.761723</td>
    </tr>
    <tr>
      <th>2</th>
      <td>TEST_000002</td>
      <td>66.101029</td>
    </tr>
    <tr>
      <th>3</th>
      <td>TEST_000003</td>
      <td>38.324813</td>
    </tr>
    <tr>
      <th>4</th>
      <td>TEST_000004</td>
      <td>42.730050</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>291236</th>
      <td>TEST_291236</td>
      <td>47.148742</td>
    </tr>
    <tr>
      <th>291237</th>
      <td>TEST_291237</td>
      <td>51.391279</td>
    </tr>
    <tr>
      <th>291238</th>
      <td>TEST_291238</td>
      <td>23.297811</td>
    </tr>
    <tr>
      <th>291239</th>
      <td>TEST_291239</td>
      <td>22.632192</td>
    </tr>
    <tr>
      <th>291240</th>
      <td>TEST_291240</td>
      <td>47.696575</td>
    </tr>
  </tbody>
</table>
<p>291241 rows × 2 columns</p>
</div>

### 제출 결과

- 751개팀 중 120위

![image](https://user-images.githubusercontent.com/84713532/206893687-0b97965a-b913-40e8-96ee-8c7b12568aea.png)
