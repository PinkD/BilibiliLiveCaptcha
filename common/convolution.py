import numpy as np

sharp_kernel = np.array(
    [[2, 2, 2],
     [2, 14, 2],
     [2, 2, 2]]
)


def generate_dst(srcImg):
    m = srcImg.shape[0]
    n = srcImg.shape[1]

    dstImg = np.zeros((m - sharp_kernel.shape[0] + 1, n - sharp_kernel.shape[0] + 1), dtype=np.uint8)
    # dstImg = np.zeros((m, n))
    return dstImg


def conv(src, dst, kernel, k_size):
    for i in range(dst.shape[0]):
        for j in range(dst.shape[1]):
            value = _con_each(src[i:i + k_size, j:j + k_size], kernel)
            dst[i, j] = value


def _con_each(src, kernel):
    pixel_count = kernel.size
    pixel_sum = 0
    _src = src.flatten()
    _kernel = kernel.flatten()

    for i in range(pixel_count):
        pixel_sum += _src[i] * _kernel[i]

    value = pixel_sum / pixel_count

    value = value if value > 0 else 0

    value = value if value < 255 else 255

    # value = sigmoid(pixel_sum)

    return value


def conv_2d(src, kernel, k_size):
    dst = generate_dst(src)
    conv(src, dst, kernel, k_size)
    return dst


def conv_2d_default(src):
    return conv_2d(src, sharp_kernel, 3)
