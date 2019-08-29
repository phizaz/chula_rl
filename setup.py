from distutils.core import setup

import pkgutil
import os

# list packages
base_name = "chula_rl"
base_path = os.path.join(os.path.dirname(__file__), base_name)
packages = [base_name]


def recursive_add(base_name, path):
    for finder, name, ispkg in pkgutil.iter_modules(path):
        if ispkg:
            new_base = "{}.{}".format(base_name, name)
            new_path = [os.path.join(path[0], name)]
            packages.append(new_base)
            recursive_add(new_base, new_path)


recursive_add(base_name, [base_path])
for p in packages:
    print(p)

# setup script
setup(
    name="chula_rl",
    install_requires=[
        'numpy', 'pandas', 'gym', 'numba', 'matplotlib', 'seaborn'
    ],
    version="0.1",
    packages=packages,
)
