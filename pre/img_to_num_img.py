"""
split image matrix to number matrix
"""

from collections import defaultdict

import numpy as np

OP_ADD = 1
OP_SUB = 2


def img_to_num(img):
    h = img.shape[0]
    w = img.shape[1]

    mark_area = defaultdict(bool)
    number_map = []

    for j in range(w):
        for i in range(h):
            if img[i][j] < 200 and not mark_area[(i, j)]:
                tmp_map = {}
                find_and_mark_next(i, j, img, tmp_map, mark_area)
                if len(tmp_map) > 20:
                    number_map.append(tmp_map)
    tmp = []
    for i in number_map:
        tmp.append(remove_offset(i, h, w))

    if len(tmp) < 3:
        # number not enough
        # print("number not enough")
        return None

    count = 0
    for i in tmp:
        if len(i) < 15:
            if len(tmp) == 3:
                # number not enough
                # print("number not enough")
                return None
            count += 1
            if count > 1:
                # for i in tmp:
                #     print_img(i)
                # more than one op
                # print("more than one op")
                return None
    return tmp


def remove_offset(number_map, h, w):
    min_h = h
    max_h = 0
    min_w = w
    max_w = 0
    for (i, j) in number_map.keys():
        if i < min_h:
            min_h = i
        if i > max_h:
            max_h = i

        if j < min_w:
            min_w = j
        if j > max_w:
            max_w = j
    result = np.ones([max_h - min_h + 1, max_w - min_w + 1])
    result *= 255
    for i, j in number_map.keys():
        result[i - min_h][j - min_w] = number_map[(i, j)]
    return result


def find_and_mark_next(i, j, img, result, mark_area):
    h = img.shape[0]
    w = img.shape[1]
    if i < 0 or j < 0 or i >= h or j >= w or img[i][j] > 200 or mark_area[(i, j)]:
        # end
        return
    result[(i, j)] = img[i][j]
    mark_area[(i, j)] = True
    # left
    find_and_mark_next(i - 1, j, img, result, mark_area)
    # up
    find_and_mark_next(i, j - 1, img, result, mark_area)
    # right
    find_and_mark_next(i + 1, j, img, result, mark_area)
    # down
    find_and_mark_next(i, j + 1, img, result, mark_area)
    # left up
    find_and_mark_next(i - 1, j - 1, img, result, mark_area)
    # right up
    find_and_mark_next(i + 1, j - 1, img, result, mark_area)
    # left down
    find_and_mark_next(i - 1, j + 1, img, result, mark_area)
    # right down
    find_and_mark_next(i + 1, j + 1, img, result, mark_area)
    if i == 0 or i == h - 1:
        for offset in range(10):
            find_and_mark_next(i, j - offset, img, result, mark_area)
            find_and_mark_next(i, j + offset, img, result, mark_area)
