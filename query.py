#%%
import time
import pandas as pd
from query_arxiv import query
from query_semantic import paper as semantic_paper

# title is for the file name
# keyword queries are for the arxiv search

'''
title = 'anime_all_1'
#keyword_list_1 = ['anime', 'animation', 'caricature', 'cartoon', 'comic', 'drawing', 'line art', 'manga', 'sketch', 'toon']
keyword_list_1 = ['anime', 'cartoon', 'comic', 'manga', 'toon']
keyword_list_2 = ['automatic', 'understanding', 'analysis', 'learning',
'color', 'create', 'creation', 'generate', 'generation', 'estimation'
'classification', 'detection', 'segmentation', 'recognition',
'painting', 'retrieve', 'retrieval', 'extraction', 'recommendation',
'synthesis', 'transfer', 'transform', 'translation', 'interpolation'
'dataset', 'classification', 'clustering', 'detect', 'segment',
'artificial intelligence', 'deep', 'deep learning', 'machine learning', 
'computer vision', 'computer graphic*', 'neural network', 'NN', 
'artificial neural network', 'ANN', 'convolutional neural network', 
'ConvNet', 'CNN', 'recurrent neural network', 'RNN', 
'long short term memory', 'LSTM', 'transformer', 'attention',
'generative adversarial network', 'GAN', 'autoencoder']
'''

'''
title = 'accelerator'
keyword_list_1 = ['processing-in-memory', 'near-memory', 
                  'fpga', 'field programmable gate array', 'field-programmable-gate-array', 
                  'accelerat*', 'hardware', 'software']
keyword_list_2 = ['transformer*', 'multi-head', 'attention', 'softmax', 'soft-max', 'position-wise feed-forward', 'convolution*', 'neural network']
'''

'''
title = 'denoising_signal'
keyword_list_1 = ['ECG', 'EKG', 'PPG', 
                  'RPPG', 'EEG', 
                  'bio-signal', 'biosignal', 'physiological signal',
                  'electro-cardiogram', 'photo-plethysmogram',
                  'electrocardiogram', 'photoplethysmogram',
                  'photoplethysmography',
                  'plethysmogram', 'remote photoplethysmogram'
                  'remote PPG', 'electroencephalogram']
keyword_list_2 = ['denoising', 'denoise', 
                  'improvement', 'cleaning',
                  'translation', 'generation', 'generate',
                   'prediction', 'forecasting']
'''

title = 'weakly_supervised_2'
keyword_list_1 = ['weakly supervised', 'weakly-supervised', 'wsod', 'weak supervision', 'image-level supervision', 'weak label', 'weakly labeled']
keyword_list_2 = ['detect', 'localization' , 'recognition', 'segment', 'network']

#%%
# basic search query style
search_query_basic_1 = ' OR '.join(keyword_list_1)
search_query_basic_2 = ' OR '.join(keyword_list_2)
search_query_basic = '(' + search_query_basic_1 + ') AND ('+ search_query_basic_2 + ')'
print(search_query_basic)

# arxiv queries
# https://arxiv.org/help/api/basics
# https://arxiv.org/help/api/user-manual
# https://github.com/lukasschwab/arxiv.py
# https://github.com/titipata/arxivpy/wiki
# ti: title, abs: abstract
search_query_arxiv_1 = 'ti:%22' + r'%22%20OR%20ti:%22'.join(keyword_list_1) + '%22'
search_query_arxiv_2 = 'ti:%22' + r'%22%20%20OR%20ti:%22'.join(keyword_list_2) + '%22'
search_query_arxiv = '%28' + search_query_arxiv_1 + r'%29%20AND%20%28'+ search_query_arxiv_2 + '%29'
print(search_query_arxiv)

#%%
# if iterative=True gets an interator over query results
# max_chunk_results max can be 2000 (leave at 1k for safety)
# take at least 3 seconds between subsequent queries for "niceness"
# sort_by can be "relevance", "lastUpdatedDate", "submittedDate"

result_raw_arxiv = query(
  query="{}".format(search_query_arxiv),
  max_chunk_results=1000,
  max_results=None,
  iterative=False,
  prune=True,
  sort_by="submittedDate",
  sort_order='descending',
  time_sleep=2
)

result_df_arxiv = pd.DataFrame(result_raw_arxiv)
result_df_arxiv = result_df_arxiv[['id', 'published', 'title', 'authors']]
result_df_arxiv['paperId'] = 'arXiv:' + result_df_arxiv['id'].str.split('abs/').str[1].str.split('v').str[0].astype(str)
result_df_arxiv = result_df_arxiv[~result_df_arxiv['title'].str.contains("animal", case=False)]
result_df_arxiv = result_df_arxiv

result_df_arxiv.to_csv('prelim_results_arxiv_{}.csv'.format(title), index=False)
print(len(result_df_arxiv))
result_df_arxiv.head()

#%%
# https://api.semanticscholar.org/
# semantic scholar search by arxiv id then return references
def semantic_recursive(max_level, recursive_list, titles_list, 
paper_data, curr_level):
  time.sleep(3)
  # verify the keys are appropriate before appending
  try:
    paper = semantic_paper('{}'.format(paper_data['paperId']), timeout=2)
    paper['no_citations'] = len(paper['citations'])
    recursive_list.append(paper)
    titles_list.append(paper['title'])
    print(curr_level, paper['title'])
    #print(paper['title'])

    if curr_level < max_level:
      for ref in paper['references']:
        if any(keyword in ref['title'].lower() for keyword in keyword_list_1):
          if any(keyword in ref['title'].lower() for keyword in keyword_list_2):
            if (ref['title'] not in titles_list):
              semantic_recursive(max_level=max_level, recursive_list=recursive_list, 
              titles_list=titles_list, paper_data=ref, curr_level=curr_level+1)
  except:
    print('Paper cannot be retrieved in proper format. Skipping to next.')

def extract_authors(df):

  authors_list_namesonly = []
  for authors_list in df['authors']:
    curr_paper_authors_list = []
    
    for author in authors_list:
      curr_paper_authors_list.append(author['name'])
      
    curr_paper_authors = ', '.join(curr_paper_authors_list)
    authors_list_namesonly.append(curr_paper_authors)
  
  return authors_list_namesonly

result_raw_semantic = []
titles_semantic = []

for index, curr_row in result_df_arxiv.iterrows():
  print(index)
  semantic_recursive(recursive_list=result_raw_semantic, 
  titles_list=titles_semantic, 
  paper_data=curr_row, curr_level=0, max_level=2)
  
result_df_semantic = pd.DataFrame(result_raw_semantic)
result_df_semantic = result_df_semantic[['authors', 'paperId', 'title', 'venue', 'year', 'no_citations', 'abstract']]

# extract paper authors from list with dictionary entries
authors_list = extract_authors(result_df_semantic)
result_df_semantic['authors'] = authors_list

# sort by year (latest to earliest) and by citations (highest to lowest)
result_df_semantic = result_df_semantic.sort_values(by=['year', 'no_citations'], ascending=False)
# remove duplicates
result_df_semantic.drop_duplicates(subset="title", keep='first', inplace=True)

result_df_semantic.to_csv('prelim_results_semantic_{}.csv'.format(title), index=False)
print(len(result_df_semantic))
result_df_semantic.head()

# %%
# total no of publications and citations per year
no_unique_years = result_df_semantic['year'].nunique()
unique_years = result_df_semantic['year'].unique()[::-1]
print(no_unique_years)
print(unique_years)

results_year = result_df_semantic.groupby(['year'])['authors'].agg('count')
print(results_year)

results_year_cit = result_df_semantic.groupby(['year'])['no_citations'].agg('sum')
print(results_year_cit)

# %%
import numpy as np
import matplotlib.pyplot as plt
# plot publications vs year
fig, axs = plt.subplots()
fig.set_size_inches(10, 7)

axs.bar(unique_years, results_year.values, linewidth=4)
#%%
# plot publications and citations vs year
fig, axs = plt.subplots()
fig.set_size_inches(10, 7)

axs.bar(unique_years, results_year.values, linewidth=4)
axs2 = axs.twinx()
axs2.plot(unique_years, results_year_cit.values, linestyle='--', color='purple')

# %%
# add a column of 'accept' with 1 (accept) or 0 (not)
# add column with nationality of main author
# add column with main goal/topic of paper
# add column with method or technique to achieve results
# wordcloud of titles