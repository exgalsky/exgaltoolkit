py_mod = import('python')
py = py_mod.find_installation('python3')

py.install_sources(
  ['__init__.py'],
  subdir: 'exgaltoolkit'
)

# Install core API modules
py.install_sources(
  ['cosmology/__init__.py', 'cosmology/parameters.py', 'cosmology/growth.py'],
  subdir: 'exgaltoolkit/cosmology'
)

py.install_sources(
  ['grid/__init__.py', 'grid/operations.py', 'grid/lpt.py'],
  subdir: 'exgaltoolkit/grid'
)

py.install_sources(
  ['initial_conditions/__init__.py', 'initial_conditions/generator.py', 'initial_conditions/writer.py'],
  subdir: 'exgaltoolkit/initial_conditions'
)

# Use subdirs for utility modules
subdir('util')
subdir('mathutil')
subdir('data')

# Legacy modules (restored for compatibility)
subdir('lpt')
subdir('mockgen')
