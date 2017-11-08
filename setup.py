from setuptools import setup

setup(name="text_similarity_tools",
      version="2.0.0",
      description="Tools for finding the similarity between two texts",
      url="https://github.com/mitmedialab/text_phrase_matching/",
      author="jakory",
      author_email="jakory@media.mit.edu",
      license="MIT",
      classifiers=[
           "Development Status :: 3 - Alpha",
           "Environment :: Console",
           "Intended Audience :: Science/Research",
           "License :: OSI Approved :: MIT License",
           "Natural Language :: English",
           "Topic :: Text Processing"
           ],
      keywords="text similarity phrase-matching",
      packages=["text_similarity_tools"],
      install_requires=[
            "fuzzywuzzy>=0.15.1",
            "nltk>=3.0.4",
            "sklearn"
          ])
