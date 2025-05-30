py_mod = import('python')
py = py_mod.find_installation('python3')

py.install_sources(
  ['__init__.py'],
  subdir: 'exgaltoolkit'
)

# Install new architecture modules directly instead of using subdirs
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
