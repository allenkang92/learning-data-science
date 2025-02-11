# DNA 염기서열의 기본 문자를 클래스 변수로 가지는 클래스 정의
class Myclass:
    base = ["A", "C", "G", "T"]  # 클래스 변수로 DNA 염기서열 리스트 정의
                                 # A: 아데닌, C: 시토신, G: 구아닌, T: 티민

# Myclass의 인스턴스 생성
obj = Myclass()

# 생성된 객체의 클래스 변수 'base' 출력
# 클래스 변수는 모든 인스턴스가 공유하는 변수
print(obj.base)  # 출력: ['A', 'C', 'G', 'T']