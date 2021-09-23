from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='lltm_cpp',
    ext_modules=[
        CppExtension('lltm_cpp', ['lltm.cpp']),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })


setup(
    name='maxpool_2d_bkw_cpp',
    ext_modules=[
        CppExtension('maxpool_2d_bkw_cpp', ['maxpool_2d_bkw.cpp']),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
