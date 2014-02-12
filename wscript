APPNAME = 'arpaca'
VERSION = '0.1.0'

top = '.'
out = 'build'

def options(opt):
  opt.load('compiler_cxx unittest_gtest')

def configure(cnf):
  cnf.check_tool('compiler_cxx unittest_gtest')
  # For debugging / tests
  # cnf.env.append_unique('CXXFLAGS', ['-g', '-W', '-Wall', '-std=c++11'])
  cnf.env.append_unique('CXXFLAGS', ['-W', '-Wall', '-DNDEBUG', '-O3', '-std=c++11'])
  # cnf.env.append_unique('INCLUDES', ['/path/to/eigen'])
  # cnf.env.append_unique('LIBPATH', ['/path/to/arpack'])

  cnf.check_cxx(header_name='Eigen/Dense')
  cnf.check_cxx(lib = 'arpack')
  cnf.check_cxx(lib = 'lapack', mandatory = False)
  cnf.check_cxx(lib = 'f77blas', mandatory = False)
  cnf.check_cxx(lib = 'gfortran', mandatory = False)
  cnf.check_cxx(lib = 'atlas', mandatory = False)

def build(bld):
  bld.program(target = 'arpaca_test',
              features = 'gtest',
              source = 'arpaca_test.cpp',
              uselib = 'ARPACK LAPACK F77BLAS ATLAS GFORTRAN')

  bld.program(target = 'arpaca_really_simple_example',
              source = 'really_simple_example.cpp',
              uselib = 'ARPACK LAPACK F77BLAS ATLAS GFORTRAN',
              install_path = None)

  bld.program(target = 'arpaca_performance_test',
              source = 'performance_main.cpp',
              uselib = 'ARPACK LAPACK F77BLAS ATLAS GFORTRAN',
              install_path = None)

  bld.program(target = 'arpaca_performance_plot',
              source = 'performance_plot.cpp',
              uselib = 'ARPACK LAPACK F77BLAS ATLAS GFORTRAN',
              install_path = None)

  bld.install_files('${PREFIX}/include', 'arpaca.hpp')
