py_mod = import('python')
py = py_mod.find_installation('python3')

py.install_sources(
  ['__init__.py'],
  subdir: 'exgaltoolkit'
)

# Install simplified API modules
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

# Install complex architecture modules (to be deprecated later)
py.install_sources(
  ['core/__init__.py', 'core/exceptions.py', 'core/config.py', 'core/data_models.py'],
  subdir: 'exgaltoolkit/core'
)

py.install_sources(
  ['services/__init__.py', 'services/cosmology_service.py', 'services/grid_service.py', 'services/lpt_service.py', 'services/output_service.py'],
  subdir: 'exgaltoolkit/services'
)

py.install_sources(
  ['workflow/__init__.py', 'workflow/workflow_engine.py', 'workflow/simulation_steps.py'],
  subdir: 'exgaltoolkit/workflow'
)

py.install_sources(
  ['api/__init__.py', 'api/factory.py', 'api/simulation.py'],
  subdir: 'exgaltoolkit/api'
)

py.install_sources(
  ['legacy/__init__.py', 'legacy/sky.py', 'legacy/cube.py', 'legacy/cosmo.py', 'legacy/ics.py'],
  subdir: 'exgaltoolkit/legacy'
)

# Use subdirs for existing modules that have proper meson.build files
subdir('util')
subdir('mathutil')
subdir('mockgen')
subdir('lpt')
subdir('data')
