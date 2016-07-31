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
                'c:/TBB_4.1/include',
                'c:/Python27/include',
                'c:/Python27/Lib/site-packages/numpy/core/include',
                ],
            library_dirs=['c:/TBB_4.1/lib/intel64/vc9'],
            libraries=['tbb'],
        ),
      ],
)
