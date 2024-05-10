import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_matrix, spmatrix


def load_sparse_csr(filename: str) -> spmatrix:
    # helper fnction to load the sparse matrix again
    # here we need to add .npz extension manually
    loader = np.load(filename + '.npz')

    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                      shape=loader['shape'])


def save_sparse_csr(filename: str, array: spmatrix) -> None:
    # helper function to save sparse matrix (e.g. the ngram frequencies per document)
    # note that .npz extension is added automatically
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)