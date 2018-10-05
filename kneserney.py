#!/usr/bin/env python
from __future__ import division

"""
Build a Kneser-Ney smoothed language model from data with (possibly
fractional) sentence weights.

author: David Chiang
version: 2013-05-07

For usage, run: python fractional.py -h

Please note that this code is primarily for instructional purposes
and is not the same code as that used for the experiments in the
paper.

Copyright (c) 2013, University of Southern California

Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.
"""

import sys
from collections import defaultdict
import math

start = "<s>"
stop = "</s>"
unk = "<unk>"
logzero = -99

verbose = 0

class CountDistribution(object):
    """A (partial) representation of a distribution over nonnegative
       integers."""
    def __init__(self, s=0):
        """We only store the first s values of the probability mass
           function. In other words, s-1 is the maximum value of r
           that we will want to know P(count=r) for."""
        self.mean = 0.
        self.p = [0.]*s
        if s > 0:
            self.p[0] = 1.

    def add(self, p=1., c=1):
        """Add c events occurring together with probability p."""
        self.mean += p * c
        for r in xrange(len(self.p)-1, -1, -1):
            self.p[r] = self.p[r] * (1-p)
            if r-c >= 0:
                self.p[r] += self.p[r-c] * p
                    
    def p_ge(self, r):
        """Probability that the the count is greater than or equal to r."""
        # The max is a sanity check against rounding errors. More
        # sophisticated corrections don't have any effect in practice.
        return max(1 - sum(self.p[r1] for r1 in xrange(r)), self.p[r])

class LanguageModel(object):
    def __init__(self, order):
        self.order = order
        self.base = None
        self.p = defaultdict(dict)
        self.b = {}

    def prob(self, context, word, interpolate=False):
        p = 0.
        found = context in self.p and word in self.p[context]
        if found:
            p += self.p[context][word]
        if self.base and (not found or interpolate):
            p += self.b.get(context, 1.) * self.base.prob(context[1:], word)
        return p

    def add(self, context, word, p):
        if len(context)+1 == self.order:
            self.p[context][word] = p
        else:
            self.base.add(context, word, p)

    def interpolate(self):
        """Interpolate this model with the lower-order model."""
        if not self.base:
            return
        self.base.interpolate()

        for context in self.p:
            for word in self.p[context]:
                self.p[context][word] += self.b[context] * self.base.prob(context[1:], word)

    def trim_start(self):
        """If an n-gram starts with two <s> symbols, trim the first one
           and transfer the (n-1)-gram to the lower-order model."""
        if not self.base:
            return

        to_del = []
        for context in self.p:
            if len(context) >= 2 and context[1] == start:
                self.base.p[context[1:]] = self.p[context]
                self.base.b[context[1:]] *= self.b[context]
                to_del.append(context)
        for context in to_del:
            del self.p[context]
            del self.b[context]

        self.base.trim_start()

    def _write_length(self, outfile):
        if self.base:
            self.base._write_length(outfile)
        if self.order >= 1:
            size = sum(len(self.p[context]) for context in self.p)
            outfile.write("ngram %d=%d\n" % (self.order, size))

    def _write_ngrams(self, outfile, parent=None):
        if self.base:
            self.base._write_ngrams(outfile, self)
        if self.order >= 1:
            outfile.write("\\%d-grams:\n" % self.order)
            for context in self.p:
                for word, prob in self.p[context].iteritems():
                    ngram = context+(word,)
                    prob = math.log10(prob) if prob > 0. else logzero
                    if parent and ngram in parent.b:
                        bow = math.log10(parent.b[ngram])
                        outfile.write("%f\t%s\t%f\n" % (prob, " ".join(ngram), bow))
                    else:
                        outfile.write("%f\t%s\n" % (prob, " ".join(ngram)))
            outfile.write("\n")

    def write_arpa(self, outfile):
        outfile.write("\\data\\\n")
        self._write_length(outfile)
        outfile.write("\n")
        self._write_ngrams(outfile)
        outfile.write("\\end\\\n")
        
class MaximumLikelihood(object):
    def __init__(self, order):
        self.order = order
        self.c = defaultdict(lambda: defaultdict(float))

    def add(self, context, word, p=1., c=1):
        self.c[context][word] += p * c

    def estimate(self):
        m = LanguageModel(self.order)
        m.p = defaultdict(dict)
        m.b = {}
        for context in self.c:
            n = sum(self.c[context][word] for word in self.c[context])
            for word in self.c[context]:
                m.p[context][word] = self.c[context][word] / n
            m.b[context] = 0.
        return m

class Uniform(object):
    def __init__(self, order=0):
        self.order = order
        self.c = defaultdict(set)

    def add(self, context, word, p=1., c=1):
        self.c[context].add(word)

    def estimate(self):
        m = LanguageModel(self.order)
        for context in self.c:
            for word in self.c[context]:
                m.p[context][word] = 1./len(self.c[context])
            m.b[context] = 0.
        return m

class KneserNey(object):
    def __init__(self, order, s=1, zerogram=None):
        """For types seen r<s times, discount by D_r (similar to Good-Turing);
           for types seen r>=s times, discount by D (absolute discounting).
           For Modified Kneser-Ney, use s=3."""
        self.order = order
        self.c = defaultdict(lambda: defaultdict(lambda: CountDistribution(s+2)))

        if order > 1:
            self.base = KneserNey(order-1, s, zerogram)
        else:
            self.base = zerogram or Uniform(order=0)
            
        self.s = s

    def add(self, context, word, p=1., c=1):
        if len(context)+1 == self.order:
            self.c[context][word].add(p, c)
        else:
            # delegate to base
            self.base.add(context, word, p, c)

    def estimate(self, recursion='uniform'):
        # Collect count-counts over all contexts
        n = []
        for r in xrange(self.s+2):
            n_r = 0.
            for context, c_context in self.c.iteritems():
                for word, c_word in c_context.iteritems():
                    n_r += c_word.p[r]
            n.append(n_r)
        if verbose:
            for r in xrange(1, self.s+2):
                sys.stderr.write("n_%s = %s\n" % (r, n[r]))

        # Compute discounts
        y = n[1] / (n[1] + 2*n[2])
        d = [0.] + [r-y*(r+1)*n[r+1]/n[r] for r in xrange(1, self.s+1)]
        if verbose:
            for r in xrange(1, self.s+1):
                sys.stderr.write("D_%s = %s\n" % (r, d[r]))
                if d[r]-d[r-1] > 1.:
                    sys.stderr.write("warning: D_%d - D_%d > 1\n" % (r, r-1))

        # Estimate model
        m = LanguageModel(order=self.order)
        for context, c_context in self.c.iteritems():
            b = 0.
            n = sum(c_context[word].mean for word in c_context)
            p_context = m.p[context]
            for word, c_word in c_context.iteritems():
                # Calculate discount for this word
                discount = 0.
                for r in xrange(1, self.s):
                    discount += c_word.p[r] * d[r]
                discount += c_word.p_ge(self.s) * d[self.s]

                # Smoothed probability estimate and backoff weight
                p_context[word] = (c_word.mean - discount) / n
                b += discount / n

                # Send events to lower-order model
                if recursion == 'uniform':
                    # The lower-order model sees one of each type.
                    self.base.add(context[1:], word, p=c_word.p_ge(1))
                elif recursion == 'exact':
                    # The lower-order model sees the discounted part
                    # of each type.  This way seems more correct but
                    # doesn't reduce to unweighted KN.
                    for r in xrange(1, self.s+1):
                        p = min(1., c_word.p_ge(r) * (d[r]-d[r-1]))
                        self.base.add(context[1:], word, p=p)
            m.b[context] = b

        if isinstance(self.base, KneserNey): # maybe recursion should be a field instead?
            m.base = self.base.estimate(recursion=recursion)
        else:
            m.base = self.base.estimate()

        return m

if __name__ == "__main__":
    kneserney.verbose = 1

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('infile', nargs='?', 
                        type=argparse.FileType('r'), default=sys.stdin, 
                        metavar='file', help='input file, tokenized')
    parser.add_argument('weightfile', nargs='?', 
                        type=argparse.FileType('r'), default=None, 
                        metavar='file', help='weights file, one weight per line')
    parser.add_argument('-o', dest='outfile', 
                        type=argparse.FileType('w'), default=sys.stdout, 
                        metavar='file', help='output file in ARPA format (default: stdout)')
    parser.add_argument('-n', dest='order', 
                        type=int, default=3,
                        metavar='n', help='order of language model (default: 3)')
    parser.add_argument('-s', dest='discount_min', 
                        type=int, default=1, 
                        metavar='count', help='minimum count for absolute discounting: *1 = original Kneser-Ney, 3 = modified Kneser-Ney')
    parser.add_argument('--start-ngrams', dest='start', 
                        choices=['short', 'long'], default='short', 
                        metavar='*short|long', help='how to handle start-of-sentence n-grams')
    parser.add_argument('--recursion', dest='recursion', 
                        choices=['uniform', 'exact'], default='uniform', 
                        metavar='*uniform|exact', help='how to count n-grams in lower-order models')
    parser.add_argument('--zerogram', dest='zerogram', 
                        choices=['uniform', 'fractional'], default='uniform', 
                        metavar='*uniform|fractional', help='use uniform or fractional counts to estimate 0-gram model')
    parser.add_argument('--interpolate', dest='interpolate', 
                        choices=['yes', 'no'], default='yes', 
                        metavar='*yes|no', help='create an interpolated model')
    args = parser.parse_args()

    infile = args.infile
    weightfile = args.weightfile
    outfile = args.outfile
    order = args.order

    if args.zerogram == 'uniform':
        # The 0-gram model is the uniform distribution over types.
        zerogram = Uniform(order=0)
    elif args.zerogram == 'fractional':
        # The 0-gram probability is proportional to the probability of
        # a type being seen.
        zerogram = MaximumLikelihood(order=0)

    estimator = KneserNey(order, s=args.discount_min, zerogram=zerogram)

    sys.stderr.write("Counting\n")
    for line in infile:
        if weightfile:
            p = float(weightfile.readline())
        else:
            p = 1.

        if args.start == 'long':
            # Prepend n-1 start symbols so that first word is an
            # n-gram.
            words = [start]*(order-1) + line.split() + [stop]
            i0 = order-1
        elif args.start == 'short':
            # Prepend 1 start symbol so that first word is a 2-gram.
            # This means that modified and unmodified counts will get
            # mixed together.
            words = [start] + line.split() + [stop]
            i0 = 1

        for i in xrange(i0, len(words)):
            context = tuple(words[max(0,i-order+1):i])
            word = words[i]
            estimator.add(context, word, p=p)

    # Add unknown word
    # Bug: not clear what p should be if 0-gram model is not uniform
    zerogram.add((), unk, p=1.) 
    estimator.add((), unk, p=0.)

    # Estimate models
    sys.stderr.write("Estimating\n")
    model = estimator.estimate(recursion=args.recursion)

    if args.interpolate == 'yes':
        sys.stderr.write("Interpolating\n")
        model.interpolate()

    if args.start == 'long':
        sys.stderr.write("Trimming start-of-sentence events\n")
        model.trim_start()

    # The start symbol is the only symbol that has a bow but no prob.
    # Insert it with zero prob so it will get output
    model.add((), start, 0.)

    sys.stderr.write("Writing\n")
    model.write_arpa(outfile)
