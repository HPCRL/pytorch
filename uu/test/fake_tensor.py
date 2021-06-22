from uu.utils import ftensor as ft
if __name__ == '__main__':
    # test assert fail
    # t0 = ft.FakeTensor()
    
    # 1d fake tensor
    t1 = ft.FakeTensor([1])
    print (t1.size())

    # 1d fake tensor
    t2 = ft.FakeTensor([7700, 8800000000000])
    print (t2.size())
