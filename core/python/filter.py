# 정수와 문자가 섞인 리스트 생성
data = [1, 2, 3, 4, 5, 6, 'a', 'b', 'c']

# 필터링 조건을 정의하는 함수
def condition(item):
    return isinstance(item, int)  # item이 정수형인지 확인하여 True/False 반환

# filter() 함수를 사용하여 data 리스트에서 정수만 필터링
# filter() 함수는 iterator 객체를 반환
result = filter(condition, data)

# iterator 객체에서 __next__() 메서드를 사용하여 필터링된 요소를 하나씩 추출
print(result)                # filter 객체 출력
print(result.__next__())    # 첫 번째 정수 1 출력
print(result.__next__())    # 두 번째 정수 2 출력
print(result.__next__())    # 세 번째 정수 3 출력
print(result.__next__())    # 네 번째 정수 4 출력
print(result.__next__())    # 다섯 번째 정수 5 출력
print(result.__next__())    # 여섯 번째 정수 6 출력
print(result.__next__())    # StopIteration 예외 발생 - 더 이상 정수가 없음
print(result.__next__())    # StopIteration 예외 발생
print(result.__next__())    # StopIteration 예외 발생
print(result.__next__())    # StopIteration 예외 발생