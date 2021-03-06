# firstDeepLearningPrj
DNN with FDS data from Caggle

Project 기간 : 2019년 1학기

1. 텐서플로, 넘파이, 판다스 import
2. 필요한 함수 선언
    1. split_train_test(data, test_ratio)
        데이터셋을 train, test 데이터로 test_ratio에 따라 나누는 함수
    2. MinMaxScaler
        데이터셋을 표준화 시키는 함수
3. 데이터셋 읽어오기
4. MinMaxScaler 함수를 통해 데이터셋 표준화
5. split_train_test 함수로 70%의 train_set, 30%의 test_set 나누기
6. train, test 데이터셋 개수 출력 및 x변수, y변수 나누기
7. learning rate 설정
learning_rate를 0.01로 설정했을 때 cost가 가장 낮아짐을 확인
8. layer 설계
activation function으로 sigmoid를 사용함
overfitting을 막기 위해 dropout을 사용함
9. cost, optimizer 설정
cost는 tf 내장함수를 사용하였음(수식을 직접 입력할 경우 cost가 처음부터 nan이 나오는 문제발생)
optimizer은 AdamOptimizer를 사용하였음
10 predicted, accuracy 정의
11. 모델 train 및 Accuracy 검사
2000번 반복하며 model을 train 시키고 100번 마다 step과 cost를 출력
Accuracy를 test_set을 데이터로 출력, accuracy run을 할 때에도 dropout을 활용하였음
참고
위 모델의 학습시간이 너무 오래 걸려서 구글 colab에서 gpu가속을 통해 model을 학습시켰습니다.
colab의 경우 '3.데이터 읽어오기' 부분 코드가 다릅니다.

#**한계**
FDS 데이터셋 자체가 이상거래 비율이 현저히 낮은데 이를 구분하지 않고 한번에 러닝시킨 후 나온 결과이다. 정확도는 매우 높게 나오지만 이는 사실 이상거래가 아닌 정상거래에 기반한 정확도라 생각해도 될 정도로 정상거래 데이터 비율이 높다. 나중에 기회가 되면 모델을 세부화 시켜서 러닝시켜봐야겠다.
