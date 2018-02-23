#! /opt/local/bin/python2.7
"""
The above line is for Mac OSX. If you are running on linux, you may need:
/usr/bin/env python

Jacqueline Kory Westlund
November 2017
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

import os  # For getting basenames and file extensions.
import argparse  # For getting command line args.
import text_similarity_tools  # For matching text files.


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
                        help="""Text file containing custom stopwords to be APPENDED to default stopwords, one per
                        line""")
    PARSER.add_argument("-S, --stopwords-force", type=str, dest="stopwords_force",
                        help="""Text file containing custom stopwords to force REPLACE default stopwords, one per
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
    # punctuation and change to lowercase unless the case-sensitive flag is
    # set. Append flag is set or unset based on the argument option.
    if ARGS.stopwords:
        text_similarity_tools.set_stopwords(ARGS.stopwords,
                                            ARGS.case_sensitive, True)
    if ARGS.stopwords_force:
        text_similarity_tools.set_stopwords(ARGS.stopwords_force,
                                            ARGS.case_sensitive, False)

    # Read in the text files to match against.
    print "Going to match against the following files:"
    MATCH = {}
    for mf in ARGS.match_files:
        print "\t{}".format(os.path.basename(mf))
        MATCH[os.path.basename(mf)] = text_similarity_tools.get_text(
            mf,
            ARGS.case_sensitive)

    # For each text file to match, find the phrase matching score for each of
    # the text files to match against, along with several other text similarity
    # metrics.
    RESULTS = []
    for infile in ARGS.infiles:
        # Open text file and read in text.
        filename = os.path.splitext(os.path.basename(infile))[0]
        text = text_similarity_tools.get_text(infile, ARGS.case_sensitive)

        # Do text matching.
        for matchme in MATCH:
            print "\nComparing \"{}\" (text 1) to \"{}\" (text 2)...".format(
                os.path.basename(infile), matchme)
            match_results = text_similarity_tools.match_texts(
                text,
                MATCH[matchme],
                ARGS.n,
                ARGS.fuzzy_n,
                ARGS.fuzzy_threshold)
            match_results["file1"] = os.path.basename(infile)
            match_results["file2"] = matchme
            RESULTS.append(match_results)

    # If there is an output file set, write the tab-delimited results to it.
    if ARGS.outfile:
        with open(ARGS.outfile, "w") as outf:
            # Print a header.
            outf.write("file1\tfile2\tlength1\tlength2\tlength_diff\t" + \
                       "length_ratio\tunique_words1\tunique_words2\t" + \
                       "unique_diff\tunique_ratio\tunique_overlap\tfuzzy_simple_ratio\t" + \
                        "fuzzy_partial_ratio\tfuzzy_token_sort_ratio\t" + \
                        "fuzzy_token_set_ratio\tcosine_similarity\t" + \
                        "num_exact_matches\tnum_similar_matches\n")
            # Print all the results.
            for result in RESULTS:
                outf.write(result["file1"] + "\t" + result["file2"] + "\t" \
                    + "\t".join(map(str, result["overall"])) \
                    + "\t{}\t{}".format(result["exact"], result["similar"])
                           + "\n")

    # If there is no output file specified, just print out the tab-delimited
    # results.
    else:
        # Print a header.
        print "\nfile1\tfile2\tlength1\tlength2\tlength_diff\tlength_ratio\t" + \
            "unique_words1\tunique_words2\tunique_diff\tunique_ratio\tunique_overlap\t" + \
            "fuzzy_simple_ratio\tfuzzy_partial_ratio\tfuzzy_token_sort_ratio\t" + \
            "fuzzy_token_set_ratio\tcosine_similarity\tnum_exact_matches\t" + \
            "num_similar_matches"
        # Print all the results.
        for result in RESULTS:
            print result["file1"] + "\t" + result["file2"] + "\t" \
                + "\t".join(map(str, result["overall"])) \
                + "\t{}\t{}".format(result["exact"], result["similar"])
