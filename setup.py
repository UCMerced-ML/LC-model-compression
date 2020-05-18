from setuptools import setup
setup(name='lc',
      version='0.1',
      packages=['lc'],
      author='Yerlan Idelbayev;Miguel Á. Carreira-Perpiñán',
      author_email='yidelbayev [at] ucmerced.edu; mcarreira-perpinan [at] ucmerced.edu',
      license='LICENSE',
      description="""This is the code for LC-model-compression software.""",
      install_requires=[
            "numpy >= 1.16.2",
            "scipy >= 1.2.1",
            "torch >= 1.0.1",
            "torchvision >= 0.2.2",
            "scikit-learn >= 0.20.2"
        ],
      long_description=open('README.md').read(),
      )