from setuptools import setup, find_packages

setup(name='toposeg',
      version='0.1',
      packages=find_packages(exclude=["scripts"]),
      description='Topology-preserving Segmentation',
      url='',
      author='*****',
      author_email='*****',
      license='MIT',
      zip_safe=False,
      entry_points={'console_scripts': [
            'train3dunet=toposeg.train:main',
            'predict3dunet=toposeg.predict:main']
            }
      )