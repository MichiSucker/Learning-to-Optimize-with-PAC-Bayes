import torch
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import numpy as np
from numpy import zeros


# define a matrix that represents the application of filter to an
# image of size M x N with M rows and N columns using reflecting
# boundary conditions.
def make_filter2d(height: int, width: int, filter_to_apply: torch.Tensor) -> csr_matrix:

    s = np.shape(filter_to_apply)[0]  # filter size: s x s
    k = int((s - 1)/2)     # filter center (k,k)

    row = np.zeros(s*s*height*width)
    col = np.zeros(s*s*height*width)
    val = np.zeros(s*s*height*width)
    ctr = 0
    for y in range(0, height):
        for x in range(0, width):

            mat_row_idx = x*height + y

            for j in range(0, s):
                for i in range(0, s):
                    ii = x + (i - k)
                    jj = y + (j - k)
                    if ii < 0:
                        ii = -ii - 1
                    if jj < 0:
                        jj = -jj - 1
                    if ii >= width:
                        ii = 2*width - 1 - ii
                    if jj >= height:
                        jj = 2*height - 1 - jj

                    mat_col_idx = ii*height + jj
                    row[ctr] = mat_row_idx
                    col[ctr] = mat_col_idx
                    val[ctr] = filter_to_apply[j, i]
                    ctr = ctr + 1

    A = csr_matrix((val, (row, col)), shape=(height * width, height * width))

    return A


def make_derivatives2d(height: int, width: int):

    # y-derivatives
    row = zeros(2*height*width)
    col = zeros(2*height*width)
    val = zeros(2*height*width)
    ctr = 1
    for x in range(0, width):
        for y in range(0, height - 1):
            row[ctr] = x*height + y
            col[ctr] = x*height + y
            val[ctr] = -1.0
            ctr = ctr + 1

            row[ctr] = x*height + y
            col[ctr] = x*height + y + 1
            val[ctr] = 1.0
            ctr = ctr + 1

    Ky = csr_matrix((val, (row, col)), shape=(height*width, height*width))

    # x-derivatives
    row = zeros(2*height*width)
    col = zeros(2*height*width)
    val = zeros(2*height*width)
    ctr = 1
    for y in range(0, height):
        for x in range(0, width - 1):
            row[ctr] = x*height + y
            col[ctr] = x*height + y
            val[ctr] = -1.0
            ctr = ctr + 1

            row[ctr] = x*height + y
            col[ctr] = (x + 1)*height + y
            val[ctr] = 1.0
            ctr = ctr + 1

    Kx = csr_matrix((val, (row, col)), shape=(height*width, height*width))

    return sp.vstack([Kx, Ky])
