
import pkgutil
from importlib import import_module
import inspect

excluded = ["mpi"]

def get_submodules(package_name):
     """Gets a list of submodules for the given package."""
 
     package = __import__(package_name)
     submodules = []
 
     for _, module_name, is_pkg in pkgutil.walk_packages(package.__path__, package.__name__ + '.'):
         if not is_pkg:  # Filter out packages (only include modules)
              for excluded_package in excluded:
                   if excluded_package not in module_name:
                        submodules.append(module_name)
 
     return submodules

submod_list = get_submodules('exgaltoolkit')

def walk_members(obj,indent="  "):
    """Recursively get members of an object, including members of nested objects."""

    members = inspect.getmembers(obj)

    for name, value in members:
        if (inspect.isclass(value) or inspect.isfunction(value)) and name != "__base__" and name != "__class__":
            typestr=" (function)"
            if inspect.isclass(value): typestr=" (class)"
            print(indent+name+typestr)
            walk_members(value,indent=indent+"  ")


for submod in submod_list:
    print(f"\n{submod} (submodule)")
    submodule = import_module(submod)

    walk_members(submodule)
