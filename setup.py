from setuptools import setup, find_packages

setup(name='topohpm',
      version='0.1',
      packages=find_packages(exclude=["scripts"]),
      description='Topology-preserving Segmentation',
      url='',
      author='wlsdzyzl',
      author_email='wlsdzyzl',
      license='MIT',
      zip_safe=False,
      entry_points={'console_scripts': [
            'train3dunet=topohpm.train:main',
            'predict3dunet=topohpm.predict:main']
            }
      )