# Phrase Matching

Given one or more text files to match against, find and count matching phrases
in one or more other text files. Print the results and optionally save the
results to a file. Perform either a case-sensitive or case-insensitive match.

For exact phrase matches, the default number of words in a phrase is 3.

For similar but not exact phrase matches, the default number of words in a
phrase is 2.

## Setup and dependencies

### Python

The following libraries are required (versions used for developing and testing
listed in parantheses). You can `port install` or `pip install` or whatever
depending on your preference for python library management.

- swalign (v0.3.4)
- fuzzywuzzy (may print a warning suggesting you optionally install
  python-Levenshtein, v0.12.0) (v.0.15.1)
- nltk (3.0.4\_0)

### Version notes

This program was developed and tested with:

- Python 2.7.6
- Mac OS X 10.10.5

## Usage

`match_phrases.py [-h] [-s, --stopwords STOPWORDS] -m, --match
                        MATCH_FILES [-o, --outfile OUTFILE]
                        [-c, --case-sensitive]
                        infiles [infiles ...]`

Given one or more text files to match against, find the number of matching
phrases in each of a provided list of text files, and print the results.

positional arguments:
- infiles: One or more text files to process.

optional arguments:
- `-h, --help`: show this help message and exit
- `-s, --stopwords STOPWORDS`: Text file containing custom stopwords, one per
  line
- `-m, --match MATCH_FILES`: Text file to match against. Can specify this
  argument more than once if you want to match against multiple files.
- `-o, --outfile OUTFILE`: A file to write the results to (otherwise printed to
  stdout).
- `-c, --case-sensitive`: Do case-sensitive phrase matching. By default, the
  phrase matching is case-insensitive.

## Bugs and issues

Please report all bugs and issues on the [text_phrase_matching github issues
page](https://github.com/mitmedialab/text_phrase_matching/issues).
