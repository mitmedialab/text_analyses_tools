#! /opt/local/bin/python2.7
"""
The above line is for Mac OSX. If you are running on linux, you may need:
/usr/bin/env python

Jacqueline Kory Westlund, Hae Won Park, Ara Adhikari
August 2017
The MIT License (MIT)

Copyright (c) 2017 Personal Robots Group

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import os # For getting basenames and file extensions.
import string # To use the punctuation list.
import argparse # For getting command line args.
import fuzzywuzzy.fuzz # For fuzzy string matching.
import nltk.corpus # To get list of stopwords.
from nltk.util import ngrams # To get ngrams from texts.
from nltk.stem.wordnet import WordNetLemmatizer # To lemmatize words.


def get_ngrams_matches(text1, text2, num_words=3):
    """ Find matching ngrams (i.e., phrases of N words) between two texts. We
    use a default of N=3 because a smaller N (e.g. N=2) often retains too
    little information to be considered actual phrase matching, while larger N
    may encompass more information than would constitute a single phrase.
    However, an appropriate N can be selected by the user.
    """
    # We count up matching ngrams from text1 and text2 to find how many ngrams
    # matched between the two texts. If there are some ngrams that are present
    # multiple times in the texts, they will be counted multiple times... so we
    # first remove duplicate ngrams from the first text before counting
    # matches.
    #
    # Just matching ngrams of length N won't account for any longer phrases that
    # match. E.g., if there is a phrase of length N+1 that matches, it will
    # appear as two adjacent ngrams that both match.
    #
    # This also doesn't account for ordering of phrases. As is, this will match
    # ngrams from anywhere in one text with ngrams from anywhere in the other.
    # If that's all that's needed, we're done. If more alignment and ordering is
    # necessary in the phrase matching, then there's more to do... TODO
    ngrams1 = set(ngrams(text1.split(), num_words))
    ngrams2 = list(ngrams(text2.split(), num_words))

    # Count occurrences of each ngram from text1 in text2.
    matches = {}
    for ng1 in ngrams1:
        if ng1 in ngrams2:
            if ng1 in matches:
                matches[ng1] += 1
            else:
                matches[ng1] = 1
    return matches


def get_fuzzy_matches(text1, text2, num_words=4, threshold=80):
    """ Find similar ngrams between two texts. Since we are using fuzzy string
    matching rather than exact matching, we probably want to use a larger N so
    that we have more words to consider as part of a phrase -- that way, they
    are a word off, or use a different word in the middle of a similar phrase,
    they will still match. We use a default of N=4, but the user can set this
    to whatever they feel is appropriate.

    The threshold value for fuzzy string matching is arbitrarily set to 80
    (this fuzzy match score can range from 0-100) and should be adjusted by the
    user for their particular dataset. A higher value indicates more similar
    strings.
    """

    # Lemmatize words first to make this fuzzier (i.e., the word will match
    # other tenses or cases of the same word).
    lemmatizer = WordNetLemmatizer()
    ngrams1 = set(ngrams(
        [lemmatizer.lemmatize(word) for word in text1.split()],
        num_words))
    ngrams2 = list(ngrams(
        [lemmatizer.lemmatize(word) for word in text2.split()],
        num_words))


    # Count fuzzy-matched occurrences of each ngram from text1 in text2. Save
    # a nicely formatted list of the ngrams that matched for printing out later.
    matches = 0
    matches_to_print = []

    for ng1 in ngrams1:
        for ng2 in ngrams2:
            ratio = fuzzywuzzy.fuzz.token_sort_ratio(ng1, ng2)
            if ratio > threshold:
                matches += 1
                matches_to_print.append("\t{}\t{} ~~ {}".format(
                    ratio, ng1, ng2))
    return matches, matches_to_print


def get_overall_similarity(text1, text2):
    """ Get the overall similarity between the two texts. """
    print "Computing overall similarity between texts..."

    # Collect results into a list to return.
    results = []
    # Get each text's length, the difference in lengths, and the length ratio.
    len1 = len(text1.split())
    len2 = len(text2.split())
    print "\tLength Text 1: {}\n\tLength Text 2: {}".format(len1, len2)
    print "\tLength difference: {}\n\tLength ratio: {}".format(
        len1 - len2,
        float(len1)/len2)
    results.append(len1)
    results.append(len2)
    results.append(len1 - len2)
    results.append(float(len1)/len2)

    # Compare the number of unique words in each story.
    uniq1 = len(set(text1.split()))
    uniq2 = len(set(text2.split()))
    print "\tUnique words Text 1: {}\n\tUnique words Text 2: {}".format(
        uniq1,
        uniq2)
    print "\tUnique words difference: {}\n\tUnique words ratio: {}".format(
        uniq1 - uniq2,
        float(uniq1)/uniq2)
    results.append(uniq1)
    results.append(uniq2)
    results.append(uniq1 - uniq2)
    results.append(float(uniq1)/uniq2)

    # The fuzzywuzzy library has four different string comparison ratios that it
    # can calculate, so we get all of them.
    simple = fuzzywuzzy.fuzz.ratio(text1, text2)
    partial = fuzzywuzzy.fuzz.partial_ratio(text1, text2)
    token_sort = fuzzywuzzy.fuzz.token_sort_ratio(text1, text2)
    token_set = fuzzywuzzy.fuzz.token_set_ratio(text1, text2)
    print "\tSimple fuzzy ratio: {}".format(simple)
    print "\tPartial fuzzy ratio: {}".format(partial)
    print "\tToken sort fuzzy ratio: {}".format(token_sort)
    print "\tToken set fuzzy ratio: {}".format(token_set)
    results.append(simple)
    results.append(partial)
    results.append(token_sort)
    results.append(token_set)

    # Return the list of match results.
    return results


def match_texts(text1, text2, num_words, fuzzy_n, fuzzy_threshold):
    """ Find matching phrases of at least the specified number of words in the
    two provided strings. Compute some other text similarity scores, including
    several fuzzy string matching ratios.
    """
    results = {}
    # Get overall similarity scores for the two texts.
    results["overall"] = get_overall_similarity(text1, text2)

    # Use ngram matching to find exact matching phrases of length N. This will
    # produce duplicate matches for any phrases longer than length N, but still
    # generates the general results we're looking for, i.e., higher match scores
    # for texts that have more phrases, and more longer phrases, in common.
    # You can think of it like this:
    #   - Score 1 for each matching phrase of length N.
    #   - Score 2 for each matching phrase of length N+1.
    #   - Score 3 for each matching phrase of length N+2.
    #   - etc.
    # The sum of these scores is the total match score. You get a higher score
    # for more total matches, as well as for longer matching phrases.
    print "Looking for exact phrase matches with N={}...".format(num_words)
    exact_matches = get_ngrams_matches(text1, text2, num_words=num_words)
    # Print the results of the matching.
    results["exact"] = len(exact_matches)
    if len(exact_matches) < 1:
        print "\tNo exact matches found."
    else:
        print "\tFound {} exact matches:".format(len(exact_matches))
        print "\t--------------------------------------------------------"
        print "\tMatches\tNgram"
        print "\t--------------------------------------------------------"
        for match in exact_matches:
            print "\t{}\t{}".format(exact_matches[match], match)

    # Next, find similar matches using fuzzy string matching. This is also based
    # on ngrams. It uses a larger N and fuzzy string matching to compare the
    # ngrams to each other, so ngrams that are not exactly the same can still
    # match (e.g., if they are a word off, or if the words are in a different
    # order).
    print "Looking for similar phrase matches with N={}...".format(fuzzy_n)
    print "Similar phrases must have fuzzy match scores above " + \
        "{} to be counted!".format(fuzzy_threshold)
    num_similar_matches, similar_matches = get_fuzzy_matches(
        text1,
        text2,
        num_words=fuzzy_n,
        threshold=fuzzy_threshold)
    # Print the results of the matching.
    results["similar"] = num_similar_matches
    if num_similar_matches < 1:
        print "\tNo similar matches found."
    else:
        print "\tFound {} similar matches:".format(num_similar_matches)
        print "\t--------------------------------------------------------"
        print "\tScore\tText 1 ~~ Text 2"
        print "\t--------------------------------------------------------"
        for match in similar_matches:
            print match
    # Return the dictionary of match results.
    return results


def get_text(text_file, case_sensitive, stopwords):
    """ Read in the text of the provided file, remove all punctuation, remove
    extraneous whitespace, and make it all lowercase if the case-sensitive flag
    is not set. Return a string containing the processed text.
    """
    # Open the file for reading.
    with open(text_file, "r") as fil:
        # Read the file contents.
        contents = fil.read()

    # Tokenize (naively, just split on whitespace) and remove stopwords.
    # Remove extra whitespace via the split, and rejoin words with a space as
    # the delimiter back into a single string.
    contents = " ".join([word for word in \
            contents.translate(None, string.punctuation).split() \
            if word.lower() not in stopwords])

    # If we should be case-insensitive, make all words lowercase.
    if not case_sensitive:
        contents = contents.lower()

    return contents


if __name__ == "__main__":
    """ Main function. Get args from the command line, find matching phrases in
    the provided text files, print out the results.
    """
    # Args are:
    # A text file containing a custom list of stopwords (one per line).
    # One or more text files to match against.
    # A list of text files to match.
    # Optionally, a file to write the results to (otherwise printed to stdout).
    PARSER = argparse.ArgumentParser("""Given one or more text files to match
                                     against, find the number of matching
                                     phrases in each of a provided list of text
                                     files, and print the results.""")
    PARSER.add_argument("-s, --stopwords", type=str, dest="stopwords",
                        help="""Text file containing custom stopwords, one per
                        line""")
    PARSER.add_argument("-m, --match", required=True, action="append",
                        dest="match_files", help="""Text file to match against.
                        Can specify this argument more than once if you want to
                        match against multiple files.""")
    PARSER.add_argument("-o, --outfile", dest="outfile", help="""A file to
                        write the results to (otherwise printed to stdout).""")
    PARSER.add_argument("-c, --case-sensitive", dest="case_sensitive",
                        action="store_true", default=False, help="""Do
                        case-sensitive phrase matching. By default, the phrase
                        matching is case-insensitive.""")
    PARSER.add_argument("infiles", type=str, nargs="+", help="""One or more
                        text files to process.""")
    PARSER.add_argument("-n, --n", dest="n", default=3, help="""How many words
                        to match when matching phrases.""")
    PARSER.add_argument("-f, --fuzzy-n", dest="fuzzy_n", default=4, help="""How
                        many words to match when fuzzy matching phrases.""")
    PARSER.add_argument("-t, --fuzzy_threshold", dest="fuzzy_threshold",
                        default=80, help="""The threshold over which fuzzy
                        string matches must be to be counted as a match (higher
                        is more similar, max 100).""")

    ARGS = PARSER.parse_args()

    # Open stopword file and get the list of custom stopwords. We remove any
    # punctuation and change to lowercase unless the case-sensitive flag is set.
    STOPWORDS = nltk.corpus.stopwords.words("english")
    if ARGS.stopwords:
        with open(ARGS.stopwords, "r") as custom_words:
            print "Reading custom stopword list..."
            CUSTOM_STOPWORDS = custom_words.readlines()
            CUSTOM_STOPWORDS = [w.strip().translate(None, string.punctuation)
                                for w in CUSTOM_STOPWORDS]
            if not ARGS.case_sensitive:
                CUSTOM_STOPWORDS = [w.lower() for w in CUSTOM_STOPWORDS]
            print "Custom stopwords: {}\nAdding to stopword list...".format(CUSTOM_STOPWORDS)
            STOPWORDS += CUSTOM_STOPWORDS

    # Read in the text files to match against.
    print "Going to match against the following files:"
    MATCH = {}
    for mf in ARGS.match_files:
        print "\t{}".format(os.path.basename(mf))
        MATCH[os.path.basename(mf)] = get_text(mf, ARGS.case_sensitive, STOPWORDS)


    # For each text file to match, find the phrase matching score for each of
    # the text files to match against.
    RESULTS = []
    for infile in ARGS.infiles:
        # Open text file and read in text.
        filename = os.path.splitext(os.path.basename(infile))[0]
        text = get_text(infile, ARGS.case_sensitive, STOPWORDS)

        # Do phrase matching.
        for matchme in MATCH:
            print "\nComparing \"{}\" (text 1) to \"{}\" (text 2)...".format(
                os.path.basename(infile), matchme)
            match_results = match_texts(text, MATCH[matchme], ARGS.n,
                                        ARGS.fuzzy_n, ARGS.fuzzy_threshold)
            match_results["file1"] = os.path.basename(infile)
            match_results["file2"] = matchme
            RESULTS.append(match_results)

    # If there is an output file set, write the tab-delimited results to it.
    if ARGS.outfile:
        with open(ARGS.outfile, "w") as outf:
            # Print a header.
            outf.write("file1\tfile2\tlength1\tlength2\tlength_diff\t" + \
                       "length_ratio\tunique_words1\tunique_words2\t" + \
                       "unique_diff\tunique_ratio\tfuzzy_simple_ratio\t" + \
                        "fuzzy_partial_ratio\tfuzzy_token_sort_ratio\t" + \
                        "fuzzy_token_set_ratio\tnum_exact_matches\t" + \
                        "num_similar_matches")
            # Print all the results.
            for result in RESULTS:
                outf.write(result["file1"] + "\t" + result["file2"] + "\t" \
                    + "\t".join(map(str, result["overall"])) \
                    + "\t{}\t{}".format(result["exact"], result["similar"]))

    # If there is no output file specified, just print out the tab-delimited
    # results.
    else:
        # Print a header.
        print "\nfile1\tfile2\tlength1\tlength2\tlength_diff\tlength_ratio\t" + \
            "unique_words1\tunique_words2\tunique_diff\tunique_ratio\t" + \
            "fuzzy_simple_ratio\tfuzzy_partial_ratio\tfuzzy_token_sort_ratio\t" + \
            "fuzzy_token_set_ratio\tnum_exact_matches\tnum_similar_matches"
        # Print all the results.
        for result in RESULTS:
            print result["file1"] + "\t" + result["file2"] + "\t" \
                + "\t".join(map(str, result["overall"])) \
                + "\t{}\t{}".format(result["exact"], result["similar"])
