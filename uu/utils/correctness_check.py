
import numpy as np
def point_wise_compare_2d(m, n, out, out_ref) -> int:
    count = 0
    for i in range(0, m):
        for j in range(0, n):
            if out[i,j] != out_ref[i, j]:
                if count <= 20:
                    print("out {:>25}, out_ref {:>25}, diff {:>25}".format(
                        out[i,j], out_ref[i,j], (out[i,j]-out_ref[i,j])
                    ))
                count += 1

    print (" precentile {} / {} = {}".format(count, m*n, (count/m/n)))
    return count


def point_wise_compare_3d(I, J, K, out, out_ref) -> int:
    count = 0
    for i in range(0, I):
        for j in range(0, J):
            for k in range(0, K):
                if out[i,j,k] != out_ref[i, j, k]:
                    if count <= 20:
                        print("out {:>25}, out_ref {:>25}, diff {:>25}".format(
                            out[i,j,k], out_ref[i,j,k], (out[i,j,k]-out_ref[i,j,k])
                        ))
                    count += 1

    print (" precentile {} / {} = {}".format(count, I*J*K, (count/I/J/K)))
    return count

def point_wise_compare_4d(I, J, K, L, out, out_ref) -> int:
    count = 0
    for i in range(0, I):
        for j in range(0, J):
            for k in range(0, K):
                for l in range(0, L):
                    if out[i, j, k, l] != out_ref[i, j, k, l]:
                        if count <= 20:
                            print("{}{}{}{}, out {:>25}, out_ref {:>25}, diff {:>25}".format(
                                i,j,k,l, out[i,j,k,l], out_ref[i,j,k,l], (out[i,j,k,l]-out_ref[i,j,k,l])
                            ))
                        count += 1

    print (" precentile {} / {} = {}".format(count, I*J*K*L, (count/I/J/K/L)))
    return count

def check_equal(first, second, verbose):
    print("-----------------------------------------\n")
    if verbose:
        print()
    for i, (x, y) in enumerate(zip(first, second)):
        x = x.cpu().detach().numpy()
        y = y.cpu().detach().numpy()
        if verbose:
            print("x = {}".format(x.flatten()))
            print("y = {}".format(y.flatten()))
            print('-' * 80)
        
        np.testing.assert_allclose(x, y, err_msg="Index: {}".format(i))
        #
        #np.testing.assert_equal(x, y, err_msg="Index: {}".format(i))