# 가장 기본적인 형태의 클래스 정의
# 'pass' 키워드를 사용하여 아무 기능도 없는 빈 클래스 생성
class Myclass:
    pass

# Myclass의 인스턴스(객체) 생성
# 클래스명() 형식으로 호출하여 새로운 객체를 생성하고 'obj' 변수에 할당
obj = Myclass()

# obj 객체의 타입을 출력
# type() 함수를 사용하여 객체가 어떤 클래스의 인스턴스인지 확인
# 출력 결과: <class '__main__.Myclass'>
print(type(obj))