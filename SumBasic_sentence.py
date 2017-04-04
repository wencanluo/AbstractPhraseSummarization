# -*- coding: utf-8 -*-
"""
Kathryn Nichols
Stefan Behr
LING 571
Project 2

Uses SumBasic algorithm to summarize a text.

"""

from __future__ import division
from collections import defaultdict
import nltk
import re

from SumBasic_word import get_sentences
import SumBasic_word as SumBasic

def summarize(distribution, clean_sentences, processed_sentences, N):
    """
    Recursively runs SumBasic algorithm on sentences.

    @param distribution -- probability distribution over words
    @param clean_sentences -- list of clean sentences
    @param processed_sentences -- list of processed sentences
    @param N -- maximum length of summary in sentences

    @return -- summary pieces as strings

    """

    Summary = []
    
    if N == 0: return Summary
    
    # sort words by probability
    words = sorted(distribution, key=distribution.get, reverse=True)

    for word in words:

        # get candidate sentences containing word
        candidates = [sentence for sentence in processed_sentences if word in sentence]
        
        # sort candidates by average probability
        candidates = SumBasic.sentence_averages(distribution, candidates)

        for candidate in candidates:
            original = clean_sentences[processed_sentences.index(candidate)]
            processed_sentences.remove(candidate)
            clean_sentences.remove(original)

            # if sentence fits, add sentence to summary
            if N > 0:
                Summary += [original]
                # update distribution
                for word in candidate: distribution[word] = distribution[word]**2

                return Summary + summarize(distribution, \
                        clean_sentences, processed_sentences, (N - 1))
       
    return Summary

def main():
    """
    Reads in sentences, uses SumBasic to find sentences for summary, prints
    summary to console and writes html version to file title summary_[N].html

    """
    
    import argparse, sys
    
    args = argparse.ArgumentParser()
    args.add_argument('sentences', help='file of line-separated sentences to summarize')
    args.add_argument('N', help='number of words in summary', type=int)
    args.add_argument('directory', help='path to directory for html file')
    args.parse_args()
    
    N = int(sys.argv[2])
    
    if N < 1:
        sys.stderr.write('N must be greater than 0. No output produced.')
        sys.exit()
    
    distribution, clean_sentences, processed_sentences = get_sentences(sys.argv[1])
    summary = summarize(distribution, clean_sentences, processed_sentences, N)
    print summary
    html =  SumBasic.convert_to_html(summary, N)
    open(sys.argv[3] + '/summary_' + str(N) + '.html', 'w').write(html)

if __name__=='__main__':
    main()
