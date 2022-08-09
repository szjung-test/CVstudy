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

