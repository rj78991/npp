"""
Build/install:
    setup build
    [sudo] setup install
Use:
    import npp
"""
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

setup(
      name='npp',
      cmdclass={'build_ext': build_ext, },
      ext_modules=[
        Extension(
            'npp',
            ['src/npp.cpp'],
            language='c++',
            include_dirs=[
                '.',
                '/usr/local/lib/python2.7/dist-packages/numpy/core/include'
                ],
            library_dirs=['/usr/lib'],
            libraries=['tbbmalloc', 'tbb'],
        ),
      ],
)
