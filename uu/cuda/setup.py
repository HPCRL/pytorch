from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='maxpool_2d_bkw_cuda',
    ext_modules=[
        CUDAExtension('maxpool_2d_bkw_cuda', [
            'maxpool_2d_bkw.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
