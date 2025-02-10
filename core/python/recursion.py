# 재귀(recursion)를 사용한 중첩 리스트의 모든 숫자 합계 계산

# 테스트를 위한 중첩 리스트 생성
# [1, [2, [3, 4], 5], 6, [7, 7]] 구조:
# - 1 (첫 번째 요소)
# - [2, [3, 4], 5] (두 번째 요소는 중첩 리스트)
# - 6 (세 번째 요소)
# - [7, 7] (네 번째 요소는 리스트)
mylist = [1, [2, [3, 4], 5], 6, [7, 7]]

def adding_machine(L):
    num = 0  # 합계를 저장할 변수 초기화
    for item in L:  # 리스트의 각 요소를 순회
        if not isinstance(item, list):  # item이 리스트가 아닌 경우
            num = num + item  # 숫자를 직접 더함
        else:  # item이 리스트인 경우
            num = num + adding_machine(item)  # 재귀적으로 하위 리스트의 합을 계산
    return num  # 최종 합계 반환

# 함수 실행 및 결과 출력
print(adding_machine(mylist))

# 출력값: 35
# 계산 과정:
# 1 + (2 + (3 + 4) + 5) + 6 + (7 + 7)
# = 1 + (2 + 7 + 5) + 6 + 14
# = 1 + 14 + 6 + 14
# = 35