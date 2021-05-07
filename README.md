[Colab version for the one that searches only arXiv and SemanticScholar](https://colab.research.google.com/drive/1lyKg7tWZBGxNAJ2mVf-6u_N_d8_aIRHp?usp=sharing)
[Colab version for the one with manual search](https://colab.research.google.com/drive/1Dv-bMMaTOGux5MNs1giqzOlNUNxXiIZb?usp=sharing)

Supporting functions to query from [arXiv](https://arxiv.org/) database and then use that to extract metadata from [Semantic Scholar](https://www.semanticscholar.org/), then possibly do multi-level search based on the references from semantic scholar. Finally, based on papers that were not part of the original search results, the `search_manual.py` allows to do a recursive SemanticScholar search with manually chosen paper IDs (from their website) to further look for more references not found in the original search. Also constructs some simple plots of papers and citations per year if run on a Jupyter-like environment.

To use just run `query.py`. Can manually edit the keywords (`keyword_list_1` and `keyword_list_2`) depending on what you're looking for or the level of recursive search by changing
`max_level` argument in `semantic_recursive` function.

Credit to the creators of the wrappers for the arXiv and Semantic Scholar APIs:

https://github.com/lukasschwab/arxiv.py

https://github.com/danielnsilva/semanticscholar