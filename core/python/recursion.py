# recursion

mylist = [1, [2, [3, 4], 5], 6, [7, 7]]


def adding_machine(L):
    num = 0
    for item in L:
        if not isinstance(item, list):
            num = num + item
        else:
            num = num + adding_machine(item)
    return num 

print(adding_machine(mylist))



# 출력값: 35