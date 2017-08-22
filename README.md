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

### Matching details

There are two phrase matching scores you can get: exact matches and similar
matches. As the name implies, exact matches match words exactly; you have to
have the same words in the same order in both texts for it to count as a match.
Similar matches use fuzzy string matching algorithms to find and count similar
phrases -- so they might be worded a little differently in one text than in the
other.

#### Exact matching

Exact matching uses an ngram-based method to find all exactly matching phrases
of length N.

By default, we use N=3 because a smaller N may not retains enough information
to be considered actual phrase matching, while a larger N may encompass more
information than would constitute a single phrase. But you can pass in whatever
N you like as a command-line argument.

The matching works like this:

1. Find all ngrams in each text.
2. Remove duplicate ngrams from one text.
3. Count how many ngrams are the same in both texts.
4. Return that number as the match score.

This matching does produce duplicate matches for any phrases longer than length N; however, as a result it generates higher match scores for texts that have both more matching phrases and longer matching phrases. You can think of the match scoring like this:

- Score 1 for each matching phrase of length N.
- Score 2 for each matching phrase of length N+1.
- Score 3 for each matching phrase of length N+2.
- etc.

The total match score is the sum of all these.

#### Similar matching

TODO

## Bugs and issues

Please report all bugs and issues on the [text_phrase_matching github issues
page](https://github.com/mitmedialab/text_phrase_matching/issues).
