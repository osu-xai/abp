from setuptools import find_packages
from setuptools import setup

setup(name='abp',
      version='0.1',
      description='Adaptation Based Programming Library in python',
      author='Magesh Kumar, Anurag Koul',
      author_email='muralim@oregonstate.edu, koula@oregonstate.edu',
      include_package_data=True,
      package_data={'': ['tasks']},
      packages=find_packages()
      )
