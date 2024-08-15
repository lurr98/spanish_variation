#!/usr/bin/env python3
"""
Author: Laura Zeidler
Last changed: 14.08.2024

This script provides utility functions for handling sparse matrices, initializing BERT tokenizers and tokenizing text data.

"""

import numpy as np
import numpy.typing as npt
from typing import Sequence
from scipy.sparse import csr_matrix, spmatrix
from transformers import BertTokenizer


def load_sparse_csr(filename: str) -> spmatrix:
    # helper fnction to load the sparse matrix again
    # here we need to add .npz extension manually
    loader = np.load(filename + '.npz')

    return csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])


def save_sparse_csr(filename: str, array: spmatrix) -> None:
    # helper function to save sparse matrix (e.g. the ngram frequencies per document)
    # note that .npz extension is added automatically
    np.savez(filename, data=array.data, indices=array.indices, indptr=array.indptr, shape=array.shape)

            
def initialise_tokeniser(model_name: str) -> BertTokenizer:
    # initialises BERT tokeniser of pre-trained model

    tokeniser = BertTokenizer.from_pretrained(model_name, do_lower_case=True)

    return tokeniser


def tokenise_data(text_data: list, tokeniser: BertTokenizer) -> Sequence:
    # tokenises the data using the BERT tokeniser

    encodings = tokeniser(text_data, truncation=True, padding=True, return_tensors='pt')

    return encodings