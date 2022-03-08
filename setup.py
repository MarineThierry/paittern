from setuptools import setup, find_packages

with open("requirements.txt") as f:
    content = f.readlines()
    requirements = [x.strip() for x in content]

setup(name="streamlitapp",
      version="1.0",
      description="streamlit app",
      packages=find_packages(),
      include_package_data=True,  # includes in package files from MANIFEST.in
      install_requires=requirements)


# from setuptools import find_packages
# from setuptools import setup

# with open('requirements.txt') as f:
#     content = f.readlines()
# requirements = [x.strip() for x in content if 'git+' not in x]

# setup(name='paittern',
#       version="1.0",
#       description="Project Description",
#       packages=find_packages(),
#       install_requires=requirements,
#       test_suite='tests',
#       # include_package_data: to install data from MANIFEST.in
#       include_package_data=True,
#       scripts=['scripts/paittern-run'],
#       zip_safe=False)
