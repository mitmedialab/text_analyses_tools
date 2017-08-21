#! /opt/local/bin/python2.7
#
# The above line is for Mac OSX. If you are running on linux, you may need:
#!/usr/bin/env python
#
# Jacqueline Kory Westlund, Hae Won Park, Ara Adhikari
# August 2017
#
# The MIT License (MIT)
#
# Copyright (c) 2017 Personal Robots Group
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import sys
import os
import getopt
import string
import numpy
import re
from difflib import SequenceMatcher
import swalign
import fuzzywuzzy.fuzz
import nltk.corpus # To get list of stopwords.
from nltk.stem.wordnet import WordNetLemmatizer # We lemmatize all words.
import argparse # For getting command line args.



'''
for the swalign stuff, had to make some changes to the file to fix syntax
like changing xrange to range and including print things in ()
'''

phrase_matching_file=open('matches.txt',"w")


def aligned_query_filtering(original,query):
    y=query.split()
    lengths=[]
    for a in y:
        x=len(a)
        lengths.append(x)
    for i in range(len(y)):
        if y[i] not in original:

            if y[i][0] is '-':
                regex=re.compile('^-+')
                y[i]=regex.sub('',y[i])
                y[i]=y[i].rjust(lengths[i])
            if y[i][-1] is '-':
                regex=re.compile('-+$')
                y[i]=regex.sub('',y[i])
                y[i]=y[i].ljust(lengths[i])

            regex=re.compile('-*')
            y[i]=regex.sub('', y[i])
            y[i]=y[i].ljust(lengths[i])
    return " ".join(y)

def aligned_ref_filtering(original, ref):
    y=ref.split()
    lengths=[]
    for a in y:
        x=len(a)
        lengths.append(x)
    for i in range(len(y)):
        if y[i] not in original:

            if y[i][0] is '-':
                regex=re.compile('^-+')
                y[i]=regex.sub('',y[i])
                y[i]=y[i].rjust(lengths[i])
            if y[i][-1] is '-':
                regex=re.compile('-+$')
                y[i]=regex.sub('',y[i])
                y[i]=y[i].ljust(lengths[i])

            regex=re.compile('-*')
            y[i]=regex.sub('', y[i])
            y[i]=y[i].ljust(lengths[i])
    return " ".join(y)



'''
number of errors (deletion, insertion, substitution)
----
wer code below from https://martin-thoma.com/word-error-rate-calculation/
'''
def wer(r, h): #r and h are lists
    # initialization
    d = numpy.zeros((len(r)+1)*(len(h)+1), dtype=numpy.uint8)
    d = d.reshape((len(r)+1, len(h)+1))
    for i in range(len(r)+1):
        for j in range(len(h)+1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i
    # computation
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitution = d[i-1][j-1] + 1
                insertion    = d[i][j-1] + 1
                deletion     = d[i-1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

    return d[len(r)][len(h)]

def print_alignment(str1,str2, out, width=100):
    l1=len(str1)
    l2=len(str2)

    i = 0
    while i < min(l1,l2):
        line1 = str1[i:min(i+width,l1)]
        line2 = str2[i:min(i+width,l2)]

        i += width

        if min(i,l1-1,l2-1) is not i:
            break

        while str1[i] is not ' ' or str2[i] is not ' ':
            line1 += str1[i]
            line2 += str2[i]
            i += 1
            if min(i,l1-1,l2-1) is not i:
                break

        out.write(line1+'\n')
        out.write(line2+'\n\n')

#print to screen instead of cmd line?
def print_alignment2(str1,str2, width=100):
    l1=len(str1)
    l2=len(str2)

    i = 0
    while i < min(l1,l2):
        line1 = str1[i:min(i+width,l1)]
        line2 = str2[i:min(i+width,l2)]

        i += width

        if min(i,l1-1,l2-1) is not i:
            break

        while str1[i] is not ' ' or str2[i] is not ' ':
            line1 += str1[i]
            line2 += str2[i]
            i += 1
            if min(i,l1-1,l2-1) is not i:
                break

        print(line1+'\n')
        print(line2+'\n\n')

def print_alignment_withSymbol(str1,str2, symbol, out, width=100):

    reg = [' ','-']

    l1=len(str1)
    l2=len(str2)
    l3=len(symbol)

    i = 0
    while i < min(l1,l2):
        line1 = str1[i:min(i+width,l1)]
        line2 = symbol[i:min(i+width,l3)]
        line3 = str2[i:min(i+width,l2)]

        i += width

        if min(i,l1-1,l2-1) is not i:
            break

        while str1[i] not in reg or str2[i] not in reg:
            line1 += str1[i]
            line2 += symbol[i]
            line3 += str2[i]
            i += 1
            if min(i,l1-1,l2-1) is not i:
                break

        out.write(line1+'\n')
        out.write(line2+'\n')
        out.write(line3+'\n\n')

def print_alignment_withSymbol2(str1,str2, symbol, width=100):

    reg = [' ','-']

    l1=len(str1)
    l2=len(str2)
    l3=len(symbol)

    i = 0
    while i < min(l1,l2):
        line1 = str1[i:min(i+width,l1)]
        line2 = symbol[i:min(i+width,l3)]
        line3 = str2[i:min(i+width,l2)]

        i += width

        if min(i,l1-1,l2-1) is not i:
            break

        while str1[i] not in reg or str2[i] not in reg:
            line1 += str1[i]
            line2 += symbol[i]
            line3 += str2[i]
            i += 1
            if min(i,l1-1,l2-1) is not i:
                break

        print(line1+'\n')
        print(line2+'\n')
        print(line3+'\n\n')

'----------------------------------------------------------------------'

def trim_query_lines(dir, query_lines, align_parser):
    # create result and directory
    if not os.path.exists(dir):
        os.makedirs(dir)

    for i in range(len(query_lines)-1):
        alignment = align_parser.align(query_lines[i],query_lines[i+1])
        alignment.dump()


#http://stackoverflow.com/questions/18715688/find-common-substring-between-two-strings
def longestSubstringFinder(str1, str2):
    answer = ""

    if len(str1) == len(str2):
        if str1==str2:
            return str1
        else:
            longer=str1
            shorter=str2
    elif (len(str1) == 0 or len(str2) == 0):
        return ""
    elif len(str1)>len(str2):
        longer=str1
        shorter=str2
    else:
        longer=str2
        shorter=str1

    matrix = numpy.zeros((len(shorter), len(longer)))

    for i in range(len(shorter)):
        for j in range(len(longer)):
            if shorter[i]== longer[j]:
                matrix[i][j]=1

    longest=0

    start=[-1,-1]
    end=[-1,-1]
    for i in range(len(shorter)-1, -1, -1):
        for j in range(len(longer)):
            count=0
            begin = [i,j]
            while matrix[i][j]==1:

                finish=[i,j]
                count=count+1
                if j==len(longer)-1 or i==len(shorter)-1:
                    break
                else:
                    j=j+1
                    i=i+1

            i = i-count
            if count>longest:
                longest=count
                start=begin
                end=finish
                break

    answer=shorter[int(start[0]): int(end[0])+1]
    return answer

def substringsFinder(str1,str2,len_min=2):
    substrings = []
    #rsplit(' ',1)[0] to help filter out cut off words
    #example: s taken out of stuck b/c it matched 'his head s' for 'his head so' and 'his head stuck'
    x = longestSubstringFinder(str1+' ', str2+' ').rsplit(' ',1)[0]
    x=re.sub('^..? ','',x).strip() #filtering up to two length word from the beginning of the phrase, hoping more cut off words are handeled with this


    #this line seems like it might be an issue later with the len_min(maybe??)
    while len(x) >= len_min:
        substrings.append(x)
        #print(x)

        str1 = re.sub(x, ' ', str1)
        str2 = re.sub(x, ' ', str2)
        #print(str1)
        #print(str2)
        x = longestSubstringFinder(str1, str2).rsplit(' ',1)[0]

    output=[]
    for s in substrings:
        ss=re.sub('  +','',s).strip()
        #print(ss)
        if ss!='' and len(ss)>=len_min:
            output.append(ss)
    output2=[]
    for i in range(len(output)):
        tokenized=output[i].split()
        #print(tokenized)
        length=len(tokenized)
        if length>=len_min:
            output2.append(output[i])
    return(output2)


# split based on spaces in the robot story alignment string
def exact_phrase_matching(text1, text2, min_len=3):
    """ Find exact phrases matches between two texts, using a default minimum
    phrase length of 3 (i.e., phrases must be 3 or more words). """

    #alignment
    match = 3
    mismatch = -1
    scoring = swalign.NucleotideScoringMatrix(match, mismatch)
    sw = swalign.LocalAlignment(scoring, -1.5, -.4)  # you can also choose gap penalties, etc...
    # can play around with values of match, mismatch, and the gaps parameters in localalignment
    alignment = sw.align(child_story, robot_story)
    # x=alignment.dump()
    x = alignment.match()  # x[0] is robot x[1] is child
    # print(x[0])
    robot_align = aligned_ref_filtering(robot_story, x[0])
    child_align = aligned_query_filtering(child_story, x[1])



    # split based on robot align
    # if you want to change the splitting to be based on child align, then replace 'robot' with 'child' and vice versa on this part of the code
    # until you get to matches
    robot_align_split = re.split("   +", robot_align)
    robot_split_index = [0]
    prev_index = 0
    for phrase in robot_align_split:
        index = robot_align[prev_index:].find(phrase)
        prev_index += len(phrase) + index
        robot_split_index.append(prev_index)
    child_align_split = []

    index_tracker = robot_split_index
    start_ind = index_tracker[0]
    end_ind = index_tracker[1]
    # trying to handle split between sentences (or was it words? i forgot...) with the end_ind_space
    end_ind_space = 0
    if len(robot_split_index) > 2:
        for index in robot_split_index[:-2]:
            end_ind_space = child_align.index(" ", end_ind - 1)
            phrase = child_align[start_ind:end_ind_space]
            child_align_split.append(phrase)
            index_tracker.pop(0)
            start_ind = end_ind_space
            end_ind = index_tracker[1]
        child_align_split.append(child_align[end_ind_space:])
    else:
        #else it's one huge string
        child_align_split.append(child_align)


    #getting the matches
    substring_matches=[]

    for i in range(len(robot_align_split)):

        str1=child_align_split[i]
        str2=robot_align_split[i]

        x=substringsFinder(str1,str2,min_len)
        #print(x)
        if len(x)!=0:
            phrase_matching_file.write('\nstr1: ')
            phrase_matching_file.write(str1)
            phrase_matching_file.write('\nstr2: ')
            phrase_matching_file.write(str2)
            phrase_matching_file.write('\n')
            phrase_matching_file.write('exact match: ')
            x_string=''
            for xx in x:
                x_string+=xx+', '

            phrase_matching_file.write(x_string.rstrip(', '))
            phrase_matching_file.write('\n')
        substring_matches+=x
    phrase_matching_file.write('\n')
    phrase_matching_file.write('----------\n')
    if len(substring_matches) ==0:
        print('No exact match')
        phrase_matching_file.write('No exact match\n')
    else:
        print('Exact matches:')
        phrase_matching_file.write('Exact matches: \n')
    return substring_matches


#split based on spaces in the child story alignment string
def similar_phrase_matching(child_story,robot_story, min_match_count=1):
    # TODO is there a minimum length of phrase to match?

    # alignment
    match = 3
    mismatch = -1
    scoring = swalign.NucleotideScoringMatrix(match, mismatch)
    sw = swalign.LocalAlignment(scoring, -1.5, -.4)  # you can also choose gap penalties, etc...
    # can play around with values of match, mismatch, and the gaps parameters in localalignment

    alignment = sw.align(child_story, robot_story)
    # x=alignment.dump()
    x = alignment.match()  # x[0] is robot x[1] is child

    robot_align = aligned_ref_filtering(robot_story, x[0])
    child_align = aligned_query_filtering(child_story, x[1])


    # split based on child align
    # if you want to change the splitting to be based on robot align, then replace 'child' with 'robot' and vice versa on this part of the code
    # until you get to matches
    child_align_split = re.split("   +", child_align)
    child_split_index = [0]
    prev_index = 0
    for phrase in child_align_split:
        index = child_align[prev_index:].find(phrase)
        prev_index += len(phrase) + index
        child_split_index.append(prev_index)
    robot_align_split = []
    index_tracker = child_split_index
    start_ind = index_tracker[0]
    end_ind = index_tracker[1]
    # trying to handle split between sentences (or was it words? i forgot...) with the end_ind_space
    end_ind_space=0
    if len(child_split_index)>2:
        for index in child_split_index[:-2]:
            end_ind_space=robot_align.index(" ",end_ind-1)
            phrase=robot_align[start_ind:end_ind_space]
            robot_align_split.append(phrase)
            index_tracker.pop(0)
            start_ind=end_ind_space
            end_ind=index_tracker[1]
        robot_align_split.append(robot_align[end_ind_space:])

    else:
        robot_align_split.append(robot_align)


    #geting the matches
    fuzzy_matches = []

    for i in range(len(robot_align_split)):
        str1 = child_align_split[i]
        str2 = robot_align_split[i]

        str1_split=re.split(" ", str1) #child words
        str2_split=re.split(" ", str2) #robot words
        str1_split_filtered=[]
        str1_split_filtered_noverbchange=[]
        str2_split_filtered=[]
        str2_split_filtered_noverbchange = []

        # make some changes to verbs
        change_stem=WordNetLemmatizer()
        for word in str1_split:
            if word!='':
                word_stem =change_stem.lemmatize(word,'v')
                str1_split_filtered_noverbchange.append(word)
                str1_split_filtered.append(word_stem)
        for word in str2_split:
            if word!='':
                word_stem = change_stem.lemmatize(word,'v')
                str2_split_filtered_noverbchange.append(word)
                str2_split_filtered.append(word_stem)

        str1_filtered_2 = " ".join(str1_split_filtered)
        str2_filtered_2 = " ".join(str2_split_filtered)

        str1_filtered_2noverbchange=" ".join(str1_split_filtered_noverbchange)
        str2_filtered_2noverbchange=" ".join(str2_split_filtered_noverbchange)


        # using both fuzzywuzzy and the number of word matches in phrases for filtering
        fuzzy = fuzzywuzzy.fuzz.token_sort_ratio(str1_filtered_2,str2_filtered_2)

        #the number can be increased/decreased for more/less filtering of similarity of phrases
        if fuzzy > 45 :
            match_count=0
            len2=len(str2_split_filtered)
            len1=len(str1_split_filtered)

            for word in str2_split_filtered:
                if len(str1_split_filtered)!= 0:
                    if word in str1_split_filtered:
                        str1_split_filtered.remove(word)
                        match_count+=1

            # increase/decrease number for more/less filtering
            #currently need atleast 2 words matching, so 2 words is min length too
            if match_count>min_match_count:

                fuzzy_matches.append((str2_filtered_2, re.sub('  +', ' ', str1_filtered_2), len2, len1, match_count, fuzzy))
                similar1=str2_filtered_2
                similar2=re.sub('  +', ' ', str1_filtered_2)

                phrase_matching_file.write('\n')
                phrase_matching_file.write(str1_filtered_2noverbchange)
                phrase_matching_file.write('\n')
                phrase_matching_file.write(str2_filtered_2noverbchange)
                phrase_matching_file.write('\n')
                phrase_matching_file.write('similar match: ')
                phrase_matching_file.write(similar1+' || '+similar2)
                phrase_matching_file.write('\n')

    phrase_matching_file.write('\n----------\n')
    if len(fuzzy_matches)==0:
        print('No similar phrases found~')
        phrase_matching_file.write('No similar phrases found~\n')
    else:
        print('Similar phrases:')
        phrase_matching_file.write('Similar phrases:\n')
    return fuzzy_matches


def text_alignments(argv,query_dir,ref_dir):
    result_dir = 'result/'

    # punctuation
    regex = re.compile('[%s]' % re.escape(string.punctuation))

    match = 3
    mismatch = -1
    scoring = swalign.NucleotideScoringMatrix(match, mismatch)
    sw = swalign.LocalAlignment(scoring,-1.5,-.4)   # you can also choose gap penalties, etc...
                                                    # can play around with values of match, mismatch, and the gaps parameters in localalignment

    try:
        opts, args = getopt.getopt(argv,"hq:r:",["queryDir=","refDir="])
    except getopt.GetoptError:
        print ('text_alignment.py -q <query_dir> -r <ref_dir>')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print ('text_alignment.py -q <query_dir> -r <ref_dir>')
            sys.exit()
        elif opt in ("-q", "--queryDir"):
            query_dir = arg
        elif opt in ("-r", "--refDir"):
            ref_dir = arg

    print ('Query Directory is ', query_dir)
    print ('Reference Directory is ', ref_dir)

    # create result and directory
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    #
    for root, subdirs, files in os.walk(query_dir):

        for filename in files:

            if filename.endswith('txt'):
                query_file_path = os.path.join(root, filename)
                print (query_file_path)
                ref_file_path = os.path.join(ref_dir, filename.replace('transcript','groundtruth'))
                print (ref_file_path)
                rst_file_path = os.path.join(result_dir, filename.replace('transcript','result'))

                #query_lines=[]
                query_line=''
                ref_line=''
                with open(query_file_path, 'r') as query_file, open(ref_file_path, 'r') as ref_file, open(rst_file_path, 'w+') as rst_file:

                    for line in query_file:
                        line = re.split(';', line)
                        line = regex.sub('', line[0]).rstrip().lower()
                        query_line += line + ' '
                        #query_lines.append(line)

                    #trim_query_lines('query_trimmed', query_lines, sw)


                    for line in ref_file:
                        line = regex.sub('', line).rstrip().lower()
                        ref_line += line + ' '


                    #print query_line
                    #print ref_line

                    alignment=sw.align(query_line,ref_line)
                    x=alignment.match() #x[0] corresponding with ref_line, x[1] corresponding with query_line

                    print('****************************************************************')

                    #print_alignment_withSymbol(x[0],x[1],x[2],sys.stdout)
                    #print_alignment(x[0],x[1],sys.stdout)

                    # changed a and aa to be ref_line_filtered and ref_line_filtered_split
                    # changed b and bb to be query_line_filtered and query_line_filtered_split
                    # changed aligned_act_filtering name to aligned_ref_filtering
                    # changed aligned_rec_filtering name to aligned_query_filtering
                    ref_line_filtered=aligned_ref_filtering(ref_line, x[0])
                    query_line_filtered=aligned_query_filtering(query_line, x[1])
                    print_alignment(ref_line_filtered,query_line_filtered,rst_file)
                    query_line_filtered_split=query_line_filtered.split()
                    ref_line_filtered_split=ref_line_filtered.split()
                    errors=wer(query_line_filtered_split,ref_line_filtered_split)
                    rst_file.write('\n\n\ntotal words: %d' % len(ref_line_filtered_split))
                    rst_file.write('\nERRORS: %d' % errors)
                    rst_file.write('\nWER: %.4f\n' % (errors/float(len(ref_line_filtered_split))))



def match_phrases(text1, text2, phrase_length):
    """ Find matching phrases of at least the specified number of words in the
    two provided strings.
    """
    print "Finding exact phrase matches, min length {}...".format(phrase_length)
    exact_matches = exact_phrase_matching(text1, text2, min_len=phrase_length)
    if len(exact_matches) < 1:
        print "No exact matches found."
    else:
        print "Found {} exact matches!".format(len(exact_matches))
        for m in exact_matches:
            print "\t{}".format(m)

    print "Finding similar phrase matches..."
    # TODO min length for similar phrase matching too?
    similar_matches = similar_phrase_matching(text1, text2)
    if len(similar_matches) < 1:
        print "No similar matches found."
    else:
        print "Found {} similar matches!".format(len(similar_matches))
        for m in similar_matches:
            print "\t{}".format(m)



def get_text(infile, case_sensitive, stopwords):
    """ Read in the text of the provided file, remove all punctuation, remove
    extraneous whitespace, and make it all lowercase if the case-sensitive flag
    is not set. Return a string containing the processed text.
    """
    # Open the file for reading.
    with open(infile, "r") as f:
        # Read the file contents.
        contents = f.read()

    # Tokenize (naively, just split on whitespace) and remove stopwords.
    # Remove extra whitespace via the split, and rejoin words with a space as
    # the delimiter back into a single string.
    contents = " ".join([word for word in \
            contents.translate(None, string.punctuation).split() \
            if word not in stopwords])

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
    description = """Given one or more text files to match against, find the
            number of matching phrases in each of a provided list of text files,
            and print the results. """

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-s, --stopwords", type=str, dest="stopwords",
            help="Text file containing custom stopwords, one per line")
    parser.add_argument("-m, --match", required=True, action="append",
            dest="match_files", help="""Text file to match against. Can specify
            this argument more than once if you want to match against multiple
            files.""")
    parser.add_argument("-o, --outfile", dest="outfile", help="""A file to write
            the results to (otherwise printed to stdout).""")
    parser.add_argument("-c, --case-sensitive", dest="case_sensitive",
            action="store_true", default=False, help="""Do case-sensitive phrase
            matching. By default, the phrase matching is case-insensitive.""")
    parser.add_argument("infiles", type=str, nargs="+", help="""One or more text
            files to process.""")
    parser.add_argument("-p, --phrase-length", dest="phrase_length",
            default=3, help="How many words to match when matching phrases")

    args = parser.parse_args()

    # Open stopword file and get the list of custom stopwords. We remove any
    # punctuation and change to lowercase unless the case-sensitive flag is set.
    stopwords = nltk.corpus.stopwords.words("english")
    if args.stopwords:
        with open(args.stopwords, "r") as f:
            print "Reading custom stopword list..."
            sw = f.readlines()
            sw = [w.strip().translate(None, string.punctuation) for w in sw]
            if not args.case_sensitive:
                sw = [w.lower() for w in sw]
            print "Custom stopwords: {}\nAdding to stopword list...".format(sw)
            stopwords += sw

    # Read in the text files to match against.
    match = []
    for mf in args.match_files:
        match.append(get_text(mf, args.case_sensitive, stopwords))

    # For each text file to match, find the phrase matching score for each of
    # the text files to match against.
    for infile in args.infiles:
        # Open text file and read in text.
        filename = os.path.splitext(os.path.basename(infile))[0]
        text = get_text(infile, args.case_sensitive, stopwords)

        # Do phrase matching.
        for m in match:
            match_phrases(text, m, args.phrase_length)
            # TODO get results back, print them out.

    # If there is an output file set, write the results to it. Otherwise, just
    # print the results to stdout.
    # TODO Output should be: "a log list of matches, and a csv with the number
    # of matches for each processed file"
