# Model-Agnostic Meta Learning for Fast Adaptive of Deep Networks(ICML 2017)

- 딥러닝을 통한 시계열 이상탐지를 서비스화하기 위한 두 가지 조건
    1. 좋은 예측을 위해 너무 많은 데이터가 필요하지 않음
    2. 다양한 시계열을 소수의 모델로 관리 가능해야함

    -> 필요한 방법 :  Few shot learning + Transfer learning
    - 해당 논문에서 제안하는 방법 Learn to Learn 모델

## Few-shot learning
    - k-shot learning : 각 클래스별, K개(1~5shot) 데이터로 효과적인 분류가 가능하도록 하는 것
    - one shot learning : 클래스 별 하나의 데이터로 분류 모델 수행
    - zero shot learning : 타겟 클래스에 대한 데이터가 전혀 없는 상태에서 모델 학습

## Meta Learning: Learning to learn
- Aim to train a model than can rapidly adapt to a new task with a few examples
    - 적은 양의 데이터로부터, 새로운 task 빠르게 학습할 수 있는 Meta Learner를 학습
    - Few shot learning 학습을 잘 하기 위한 모델 학습

## Transfer Learning(pre-training) 접근법과 차이점?
- Pre-Training은 일반적으로 재사용을 목적으로 사용, 상위 layer의 파라미터는 고정하고 세세한 피쳐를 뽑아줄거라고 기대하는 하위 layer만 업데이트한다.
- manifold가 유사한 이미지에 대해 이미 학습이 선행되어 다른 데이터나 클래스 분류 과제로 이관이 가능할것 이라고 믿는다.
- 학습하는 방법을 학습하는 컨셉의 메타러닝과는 다르게 transfer learning은 선행학습이라 볼 수 있다.

## Abstract
- 모델에 무관한 meta learning을 가능하게하는 알고리즘을 제안한다.
- Meta Learning 목표:
    - 적은 샘플 트레이닝 데이터로도 새로운 과제를 잘 수행할 수 있도록 모델 학습
    - 메타 러닝을 통해 모델의 parameter는 적은 데이터만으로도 효율적으로 업데이트 가능

## Introduction
- MAML 모델은 Gradient Descent Procedure 로 학습되는 어떤 문제에도 적용 가능
- 궁극적인 목표는 적은 데이터만으로 사람이 할 수 있는 일을 모델이 할 수 있도록 모델을 학습하는 것이다.
    - Initial Parameter가 몇 번의 업데이트만으로 maximal performance를 발휘하도록 학습
    - 기존의 연구들과 달리, 특정 모델 구조에 한정적이거나 parameter 를 늘리거나 하지 않는다.

- 궁극적인 학습의 목적은 여러과제에 범용적으로 적합하게 internal representation을 설정해 놓는 것이다.
- 최적의 포지션에 모델이 있다면 적은 데이터와 업데이트만으로도 좋은 결과를 기대할 수 있다.
- parameter 관점에서 loss function의 sensitivity를 극대화하는 과정

## Meta learning problem set-up
- 각 task가 하나의 원소가 되고, {task 집합}이 train set/test set 이 된다.
- key idea
    - 모든 task를 관통하는 distribution 'P(t)'의 internal feature를 찾는다.
        - internal feature 중에서도 more transferable than others한 것을 찾는다.
        - 하나의 task에 over fitting 되지 않고, 각 gradient step 마다 여러 {sample task} loss를 줄인다.
        ![image](https://user-images.githubusercontent.com/93111772/183550929-8f767326-0fc1-4a09-80bd-fe41f065b8be.png)

## A Model-Agnostic Meta Learning Algorithm
- Task의 변화에 민감하게 반응하는 model parameter를 찾는다.
- 새로운 task가 p(T)와 같은 Space에 존재한다면 적은 update로도 좋은 성능 발휘 가능
- meta learner 'f'를 찾는 것이 중요
- 새로운 i번째 task가 주어졌을때, 모델은 우리가 일반적으로 알고있는 역전파의 형태로 학습

## 구체적인 학습 목표
[샘플 Task 집합] 한 번 update된 loss의 sum을 최소화한다.
![image](https://user-images.githubusercontent.com/93111772/183551392-611db051-4c22-4c5e-8313-588eea035211.png)
![image](https://user-images.githubusercontent.com/93111772/183551425-41da939e-b340-4a9d-986b-3ae782527d50.png)

- MAML 알고리즘 학습 과정
    - {샘플 Task 집합}의 사이즈를 3이라 가정할 때
    - 전체 train 집합에서 T(1), T(2), T(3)를 샘플링한다.
    - 각 T(i)에 대해 한 번 또는 여러번 gradient update를 진행한 θ(i) 를 얻는다.
    - θ(1),θ(2),θ(3)로 생긴 loss의 sum 이 우리의 최종 loss
    - 최종loss에 대해 back propagation을 하여 θ를 업데이트한다.
    - 이 과정을 반복

# 주의 논문의 loss가 multi-task learning(MTL)과 같다고 생각하면 안된다.
![image](https://user-images.githubusercontent.com/93111772/183551893-15d5fc58-559d-466e-a92a-4f70f101877e.png)

- Multi task learning loss
    - 일반적으로 각 Task가 각자의 model을 갖고, shared layer에서 parameter를 공유하는 구조. MTL은 여러 task들을 동시에 학습하는 것이 각 task의 성능 향상에 도움을 줄 수 있다는 믿음이 될 수 있다. 
    - MTL의 장점은 regularization, knowledge의 공유되어 학습될 수 있다.
    - "주어진 모든 일을 잘하도록 학습" MTL의 컨셉과 "주어진 어떤 일에든 빠르게 적응하도록 학습"되는 MAML의 컨셉과는 다른점이 있다.

- Species of MAML
    - 해당 논문에서는 3가지 종류의 Task를 다룬다. (regression, classificationm, reinforcement learning) 컨셉은 모두 동일하고 loss만 task에 적합한 형태로 바뀌기 때문에 해당 리뷰에는 Regression만 다룬다.

- Supervised Regression
    - Goal
        - similar statistical properties로 여러 함수들에 학습 되어 있을때
        - 새로운 함수의 #K data points만으로 다음 값을 잘 예측한다. 

- Loss
![image](https://user-images.githubusercontent.com/93111772/183552639-0c8f1d13-b800-4dc1-b82a-9996191c9557.png)
[setting: t 값으로 t+1 값 예측하기]

- Key IDEA:
    - 우리는 메타러버 'f'를 잘 만들었다.
    - 새로운 함수의 k개의 데이터 포인트를 알고 있다.
    - k data points로 f를 gradient process한다.
    - f는 굉장히 loss sensitive한 상태이므로 k개로 충분히 좋은 모델로 진화한다.

- Experimental Evaluation
    - Regression
        - P(t) = continuous sine wave
        - amplitude(높이) varies[0.1 ~5.0]
        - phase(넓이) varies[0,π]
        - 'f': 2 hidden layers of size 40 with ReLU
        - K-shot Learning(5개, 10개, 20개)

        ![image](https://user-images.githubusercontent.com/93111772/183553462-298efcdf-e73e-4d1a-9631-94fb0c17afe5.png)

    - 왼쪽 두개는 MAML한 것, 오른쪽은 fine tuning 한것
    - 정리하면 적은 데이터로, update로 타겟 함수의 phase&amplitude를 잡아 낼 수 있다.
    - 또한 Wave 절반의 데이터만으로 나머지 wave의 형태를 캐치한다.
    - meta learner 'f'rk sine wave의 feature를 알고 있기 때문이다.

- Discussion and Future work
- 논문에서 소개한 meta-learning접근법은 모델이 굉장히 간단하고 미리 세팅해둬야 할 parameter가 없다.
- gradient based 모든 task에 적용 가능하기 때문에 아주 활용성이 높을 것 같다.
- 수 많은 시계열을 단 몇 개의 모델로 관리할 수 있다면 매우 편할것 같다.
- 모델에서 언급한대로 'f'는 similar statistical properties를 공유한 task에 대해서만 범용적인 적용이 가능하다.
- 해당 방법을 실제로 적용해본다면 주기성이 동일한 시계열끼리 묶고 각 군집마다 통용될 수 있는 'f'를 설계한 후 주기 길이만큼 K points로 매일 임시 모델을 업데이트하고 값을 계산하는 방식을 적용할 수 있지 않을까 싶다.

