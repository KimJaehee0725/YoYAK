''' 
This Code is based on https://github.com/Diego999/py-rouge
'''

# Rouge 계산을 위한 코드

from os import stat
import numpy as np
import pandas as pd

from transformers import PreTrainedTokenizerFast
from tqdm import tqdm

import collections
import re
import six


class Rouge:
    '''
    you have to speicifiy correct column names
    '''
    def __init__(self, data, rouge_types, reference_column='reference_summary', system_column='generated_summary', tokenizer=None, epsilon = 1e-7):
        self.data = pd.read_csv(data) # data_path
        self.rouge_types = rouge_types
        if tokenizer is None: # default tokenizer is kobart tokenizer
            self.tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-base-v2')
        else:
            self.tokenizer = tokenizer
        self.reference_col = reference_column
        self.system_col = system_column
        self.epsilon = epsilon
    
    def score(self, target, prediction):
        """Calculates rouge scores between the target and prediction.
		Args:
		target: Text containing the target (ground truth) text.
		prediction: Text containing the predicted text.
		Returns:
		A dict mapping each rouge type to a Score object.
		Raises:
		ValueError: If an invalid rouge type is encountered.
		"""

        target_tokens = self.tokenizer.tokenize(target)
        prediction_tokens = self.tokenizer.tokenize(prediction)
        result = {}

        for rouge_type in self.rouge_types:
            if rouge_type == "rougeL":
                # Rouge from longest common subsequences.
                scores = self._score_lcs(target_tokens, prediction_tokens)
            elif re.match(r"rouge[0-9]$", six.ensure_str(rouge_type)):
            # Rouge from n-grams.
                n = int(rouge_type[5:])
                if n <= 0:
                    raise ValueError("rougen requires positive n: %s" % rouge_type)
                target_ngrams = self._create_ngrams(target_tokens, n)
                prediction_ngrams = self._create_ngrams(prediction_tokens, n)
                scores = self._score_ngrams(target_ngrams, prediction_ngrams)
            else:
                raise ValueError("Invalid rouge type: %s" % rouge_type)
            result[rouge_type] = scores

        return result

    def _create_ngrams(self, tokens, n):
        """Creates ngrams from the given list of tokens.
        Args:
        tokens: A list of tokens from which ngrams are created.
        n: Number of tokens to use, e.g. 2 for bigrams.
        Returns:
        A dictionary mapping each bigram to the number of occurrences.
        """

        ngrams = collections.Counter()
        for ngram in (tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)):
            ngrams[ngram] += 1
        return ngrams


    def _score_lcs(self, target_tokens, prediction_tokens):
        """Computes LCS (Longest Common Subsequence) rouge scores.
        Args:
        target_tokens: Tokens from the target text.
        prediction_tokens: Tokens from the predicted text.
        Returns:
        A Score object containing computed scores.
        """

        if not target_tokens or not prediction_tokens:
            return dict(precision=0, recall=0, f1_score=0)

        # Compute length of LCS from the bottom up in a table (DP appproach).
        lcs_table = self._lcs_table(target_tokens, prediction_tokens)
        lcs_length = lcs_table[-1][-1]

        precision = lcs_length / len(prediction_tokens)
        recall = lcs_length / len(target_tokens)
        f1_score = 2*precision*recall /(precision+recall+self.epsilon)

        return dict(precision=precision, recall=recall, f1_score=f1_score)


    def _lcs_table(self, ref, can):
        """Create 2-d LCS score table."""
        rows = len(ref)
        cols = len(can)
        lcs_table = [[0] * (cols + 1) for _ in range(rows + 1)]
        for i in range(1, rows + 1):
            for j in range(1, cols + 1):
                if ref[i - 1] == can[j - 1]:
                    lcs_table[i][j] = lcs_table[i - 1][j - 1] + 1
                else:
                    lcs_table[i][j] = max(lcs_table[i - 1][j], lcs_table[i][j - 1])
        return lcs_table


    def _score_ngrams(self, target_ngrams, prediction_ngrams):
        """Compute n-gram based rouge scores.
        Args:
        target_ngrams: A Counter object mapping each ngram to number of
        occurrences for the target text.
        prediction_ngrams: A Counter object mapping each ngram to number of
        occurrences for the prediction text.
        Returns:
        A Score object containing computed scores.
        """

        intersection_ngrams_count = 0
        for ngram in six.iterkeys(target_ngrams):
            intersection_ngrams_count += min(target_ngrams[ngram],prediction_ngrams[ngram])

        target_ngrams_count = sum(target_ngrams.values())
        prediction_ngrams_count = sum(prediction_ngrams.values())

        precision = intersection_ngrams_count / max(prediction_ngrams_count, 1)
        recall = intersection_ngrams_count / max(target_ngrams_count, 1)
        f1_score = 2*precision*recall /(precision+recall+self.epsilon)
        
        return dict(precision=precision, recall=recall, f1_score=f1_score)


    def rouge_scores(self):

        results = dict()
        
        for rouge_type in self.rouge_types:
            if rouge_type == "rougeL":
                results['rougeL'] = dict(recall=list(),precision=list(),f1_score=list())
            elif re.match(r"rouge[0-9]$", six.ensure_str(rouge_type)):
                n = int(rouge_type[5:])
                results['rouge{}'.format(n)] = dict(recall=list(),precision=list(),f1_score=list())
                if n <= 0:
                    raise ValueError("rougen requires positive n: %s" % rouge_type)
            else :
                raise ValueError("Invalid rouge type: %s" % rouge_type)

        for i in tqdm(range(len(self.data))):
            rouge_dict = self.score(self.data[self.reference_col][i],self.data[self.system_col][i])
            for rouge_type in self.rouge_types:
                if rouge_type == "rougeL":
                    results['rougeL']['recall'].append(rouge_dict['rougeL']['recall'])
                    results['rougeL']['precision'].append(rouge_dict['rougeL']['precision'])
                    results['rougeL']['f1_score'].append(rouge_dict['rougeL']['f1_score'])
                else:
                    n = int(rouge_type[5:])
                    results['rouge{}'.format(n)]['recall'].append(rouge_dict['rouge{}'.format(n)]['recall'])
                    results['rouge{}'.format(n)]['precision'].append(rouge_dict['rouge{}'.format(n)]['precision'])
                    results['rouge{}'.format(n)]['f1_score'].append(rouge_dict['rouge{}'.format(n)]['f1_score'])
                
        # Average metric
        for keys in results:
            for metric in results[keys]:
                results[keys][metric] = np.mean(results[keys][metric])

        return results