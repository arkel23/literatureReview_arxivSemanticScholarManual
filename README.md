Supporting functions to query from [arXiv](https://arxiv.org/) database and then use that to extract metadata from [Semantic Scholar](https://www.semanticscholar.org/), then possibly do multi-level search based on the references from semantic scholar. Finally, based on papers that were not part of the original search results, perform a second recursive search to further look for more references.

Credit to the creators of the wrappers for the arXiv and Semantic Scholar APIs:

https://github.com/lukasschwab/arxiv.py

https://github.com/danielnsilva/semanticscholar