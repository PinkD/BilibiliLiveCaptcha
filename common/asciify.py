"""
ascii print a image
"""


def int_to_char(v):
    if v > 200:
        return " "
    if v > 150:
        return "o"
    if v > 100:
        return "G"
    if v > 50:
        return "M"
    return "#"


def print_img(img):
    h = img.shape[0]
    w = img.shape[1]
    print(f"{h}x{w}")
    for i in range(w):
        print("-", end="")
    print()
    for i in range(h):
        for j in range(w):
            print(f"{int_to_char(img[i][j])}", end="")
        print("|")
    for i in range(w):
        print("-", end="")
    print()
