import sys


def total(n):
    s = 0
    for i in range(n):
        s += i
    return s


if __name__ == "__main__":
    print(total(int(sys.argv[1])))
