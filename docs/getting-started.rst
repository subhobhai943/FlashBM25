Getting Started
===============

FlashBM25 indexes a corpus of strings and scores every document against a text
query. The Python classes hold onto the original documents, while the compiled
extension owns the inverted index and scoring loops.

Installation
------------

Install the package from PyPI:

.. code-block:: bash

   pip install flashbm25

FlashBM25 has no mandatory runtime dependencies. Source installs require a
C++17 compiler, CMake, pybind11, and scikit-build-core because the extension is
compiled during installation.

Build Your First Index
----------------------

.. code-block:: python

   from flashbm25 import BM25

   corpus = [
       "the quick brown fox jumps over the lazy dog",
       "machine learning systems rank documents by relevance",
       "BM25 is a bag-of-words retrieval function",
   ]

   bm25 = BM25(corpus)
   scores = bm25.get_scores("bm25 retrieval")
   top_docs = bm25.get_top_n_docs("bm25 retrieval", n=2)

``get_scores`` returns one score per indexed document. ``get_top_n`` returns
``(score, doc_index)`` pairs sorted from highest to lowest score, and
``get_top_n_docs`` resolves those indices back to the original corpus strings.

Choose a BM25 Variant
---------------------

The default class implements classic Okapi BM25. You can select another variant
through the factory-style ``variant`` argument:

.. code-block:: python

   bm25_l = BM25(corpus, variant="l", delta=0.5)
   bm25_plus = BM25(corpus, variant="plus", delta=1.0)
   bm25_adpt = BM25(corpus, variant="adpt")

The concrete classes :class:`flashbm25.BM25L`,
:class:`flashbm25.BM25Plus`, and :class:`flashbm25.BM25Adpt` can also be
instantiated directly when you want the type to be explicit.

Customize Tokenization
----------------------

FlashBM25 uses its C++ whitespace tokenizer by default. The Python tokenizer
layer is useful when you need Unicode-aware tokenization, stopword removal,
minimum token lengths, or a custom stemmer.

.. code-block:: python

   from flashbm25 import BM25, Tokenizer

   tokenizer = Tokenizer(
       mode="unicode_word",
       stopwords="english",
       min_token_len=2,
   )

   bm25 = BM25(corpus, tokenizer=tokenizer)

You can also pass a callable tokenizer. It must accept a string and return an
iterable of string tokens.

Persist an Index
----------------

Use :meth:`flashbm25.BM25.save` and :meth:`flashbm25.BM25.load` when you want to
reuse an index without rebuilding it from the original corpus:

.. code-block:: python

   bm25.save("search-index.fbm25")
   restored = BM25.load("search-index.fbm25")

Persisted indexes include the original corpus, scoring parameters, core index
bytes, and serializable tokenizer state. Callable tokenizers and callable
stemmers cannot be persisted because FlashBM25 cannot reconstruct arbitrary
Python callables on load.

Update an Existing Index
------------------------

Append documents in place:

.. code-block:: python

   bm25.add_documents([
       "new retrieval document",
       "another document about ranking",
   ])

Remove a document by zero-based document id:

.. code-block:: python

   bm25.remove_document(0)

Removal rebuilds the remaining corpus so document ids stay compact.

Field-Weighted Search
---------------------

The current :class:`flashbm25.BM25F` API supports field-weighted scoring for
documents represented as dictionaries:

.. code-block:: python

   from flashbm25 import BM25F

   field_corpus = [
       {"title": "BM25 overview", "body": "BM25 ranks text documents"},
       {"title": "Neural search", "body": "Dense vectors rank passages"},
   ]

   bm25f = BM25F(field_corpus, field_weights={"title": 2.0, "body": 1.0})
   top = bm25f.get_top_n("bm25", n=1)

The roadmap treats BM25F as a Phase 1 skeleton, with deeper multi-field scoring
features planned for Phase 3.
