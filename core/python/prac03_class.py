class MyClass:
    def get_length(self, seq):  # 인스턴스 메서드 정의 (self는 인스턴스 자신을 가리키는 매개변수)
        return len(seq)         # seq 문자열의 길이를 반환
    
obj = MyClass()                 # MyClass의 인스턴스 생성
seq = "ACGTACGT"               # DNA 염기서열 문자열
print(obj.get_length(seq))     # 인스턴스 메서드 호출하여 seq의 길이 출력
                               # 출력값: 8