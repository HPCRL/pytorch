
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
