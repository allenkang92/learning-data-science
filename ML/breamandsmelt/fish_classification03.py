"""
도미와 빙어 분류 프로그램 - 모듈화 및 기능 개선 버전

이 프로그램은 도미와 빙어를 분류하는 머신러닝 모델을 구현하며,
이전 버전들과 달리 코드를 모듈화하여 재사용성과 가독성을 높였습니다.
또한 scikit-learn의 train_test_split 함수를 사용하여 더 체계적인 데이터 분할을 구현했습니다.

주요 기능:
1. 데이터 로드 및 전처리
2. 모델 학습 및 평가
3. 새로운 데이터 예측
4. 데이터 시각화

작성일: 2024-12-26
"""

# 필요한 라이브러리 임포트
import numpy as np  # 수치 계산용 라이브러리(n차원 배열, 난수, 선형대수 등 지원)
from sklearn.model_selection import train_test_split  # 데이터셋을 학습/테스트 세트로 나누기 위한 함수
from sklearn.neighbors import KNeighborsClassifier    # K-최근접 이웃(KNN) 알고리즘 구현 클래스
import matplotlib.pyplot as plt                       # 데이터 시각화를 위한 2D 플롯 라이브러리

def load_data():
    """
    도미와 빙어의 길이, 무게 데이터를 로드하고 numpy 배열로 변환합니다.
    
    Returns:
        tuple: (fish_data, fish_target)
            - fish_data: 물고기의 길이와 무게 데이터 (n_samples, 2)
            - fish_target: 물고기의 종류 레이블 (n_samples,), 도미=1, 빙어=0
    """
    # 도미(35개)와 빙어(14개)의 길이 데이터
    fish_length = [
        25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0,
        31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0,
        35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0,
        9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0
    ]
    # 도미(35개)와 빙어(14개)의 무게 데이터
    fish_weight = [
        242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0,
        500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0,
        700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0,
        6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9
    ]
    # NumPy의 column_stack 함수로 길이와 무게를 열 단위로 묶어 2차원 배열 생성
    fish_data = np.column_stack((fish_length, fish_weight))
    # NumPy의 concatenate 함수로 도미(1)와 빙어(0) 레이블을 연결하여 타깃 배열 생성
    fish_target = np.concatenate((np.ones(35), np.zeros(14)))
    return fish_data, fish_target

def preprocess_data(fish_data, fish_target):
    """
    데이터를 학습용과 테스트용으로 분할합니다.
    
    Args:
        fish_data (np.ndarray): 물고기의 길이와 무게 데이터
        fish_target (np.ndarray): 물고기의 종류 레이블
    
    Returns:
        tuple: (train_input, test_input, train_target, test_target)
            - train_input: 학습용 입력 데이터
            - test_input: 테스트용 입력 데이터
            - train_target: 학습용 타깃 데이터
            - test_target: 테스트용 타깃 데이터
    """
    # train_test_split 함수로 데이터를 학습용과 테스트용으로 분할
    # stratify=fish_target: 클래스 비율을 유지하면서 분할
    # random_state=42: 실험 재현성을 위해 난수 시드 고정
    train_input, test_input, train_target, test_target = train_test_split(
        fish_data, fish_target, stratify=fish_target, random_state=42
    )
    return train_input, test_input, train_target, test_target

def train_model(train_input, train_target):
    """
    K-최근접 이웃 분류기를 학습시킵니다.
    
    Args:
        train_input (np.ndarray): 학습용 입력 데이터
        train_target (np.ndarray): 학습용 타깃 데이터
    
    Returns:
        KNeighborsClassifier: 학습된 KNN 모델
    """
    kn = KNeighborsClassifier()  # 기본 하이퍼파라미터로 KNN 모델 초기화
    kn.fit(train_input, train_target)  # 모델 학습
    return kn

def evaluate_model(kn, test_input, test_target):
    """
    학습된 모델의 성능을 평가합니다.
    
    Args:
        kn (KNeighborsClassifier): 학습된 KNN 모델
        test_input (np.ndarray): 테스트용 입력 데이터
        test_target (np.ndarray): 테스트용 타깃 데이터
    
    Returns:
        float: 모델의 정확도 (0~1)
    """
    score = kn.score(test_input, test_target)  # 테스트 데이터로 모델 정확도 계산
    print("모델 정확도:", score)  # 정확도 출력 (1.0이면 100% 정확)
    return score

def predict_new_data(kn, new_data):
    """
    새로운 데이터에 대해 예측을 수행합니다.
    
    Args:
        kn (KNeighborsClassifier): 학습된 KNN 모델
        new_data (list): 예측할 새로운 데이터 [길이, 무게]
    
    Returns:
        np.ndarray: 예측 결과 (0: 빙어, 1: 도미)
    """
    prediction = kn.predict([new_data])  # 새로운 데이터 예측
    print("예측 결과:", "도미" if prediction[0] == 1 else "빙어")  # 예측 결과를 사람이 읽기 쉽게 출력
    return prediction

def visualize_data(train_input, train_target, test_input, test_target, new_data=None):
    """
    데이터의 분포를 시각화합니다.
    
    Args:
        train_input (np.ndarray): 학습용 입력 데이터
        train_target (np.ndarray): 학습용 타깃 데이터
        test_input (np.ndarray): 테스트용 입력 데이터
        test_target (np.ndarray): 테스트용 타깃 데이터
        new_data (list, optional): 예측할 새로운 데이터 [길이, 무게]
    """
    plt.scatter(train_input[:, 0], train_input[:, 1], label='Train')  # 학습 데이터 (점)
    plt.scatter(test_input[:, 0], test_input[:, 1], marker='^', label='Test')  # 테스트 데이터 (삼각형)
    if new_data:  # 새로운 데이터가 있으면 빨간색 원으로 표시
        plt.scatter(new_data[0], new_data[1], marker='o', color='red', label='New')
    plt.xlabel('length (cm)')  # x축: 물고기 길이
    plt.ylabel('weight (g)')  # y축: 물고기 무게
    plt.legend()  # 범례 표시
    plt.show()    # 그래프 출력

if __name__ == "__main__":
    # 메인 실행 코드
    # 1. 데이터 준비
    fish_data, fish_target = load_data()
    train_input, test_input, train_target, test_target = preprocess_data(fish_data, fish_target)

    # 2. 모델 학습 및 평가
    kn = train_model(train_input, train_target)
    evaluate_model(kn, test_input, test_target)
    
    # 3. 새로운 물고기 예측 (길이 25cm, 무게 150g)
    new_data = [25, 150]
    predict_new_data(kn, new_data)
    
    # 4. 데이터 분포 시각화
    visualize_data(train_input, train_target, test_input, test_target, new_data)