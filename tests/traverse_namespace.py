
import pkgutil
from importlib import import_module
import inspect

def get_submodules(package_name):
     """Gets a list of submodules for the given package."""
 
     package = __import__(package_name)
     submodules = []
 
     for _, module_name, is_pkg in pkgutil.walk_packages(package.__path__, package.__name__ + '.'):
         if not is_pkg:  # Filter out packages (only include modules)
             submodules.append(module_name)
 
     return submodules

submod_list = get_submodules('exgaltoolkit')

for submod in submod_list:
    print(f"Checking exgaltoolkit submodule: {submod}")
    submodule = import_module(submod)

    for name, obj in inspect.getmembers(submodule):
        if inspect.isclass(obj):
            print(f"Found class: {name}")
        if inspect.isfunction(obj):
            print(f"Found function: {name}")