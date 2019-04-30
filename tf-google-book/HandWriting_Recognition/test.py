import random


def foo(n):
    x = 6
    y = 1000000
    count = 0
    for i in range(n):
        cur = y
        while cur > 0:
            i = random.randint(0, x - 1)
            if i > cur % x:
                break
            elif i < cur % x:
                count += 1
                break
            cur = cur // x
    return count/n


print(foo(123132113123123))
