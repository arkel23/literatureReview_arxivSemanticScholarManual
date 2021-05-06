#%%
import time
import pandas as pd
from query_arxiv import query
from query_semantic import paper as semantic_paper

title = 'weakly_supervised_2'
keyword_list_1 = ['weakly supervised', 'weakly-supervised', 'wsod', 'weak supervision', 'image-level supervision', 'weak label', 'weakly labeled']
keyword_list_2 = ['detect', 'localization' , 'recognition', 'segment', 'network']

file_name = 'manual_incomplete_{}.csv'.format(title)
df_incomplete = pd.read_csv(file_name)
print(len(df_incomplete))
df_incomplete.head()

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

#%%
result_raw_semantic = []
titles_semantic = []

for index, curr_row in df_incomplete.iterrows():
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

#result_df_semantic.to_csv('results_manual_semantic_{}.csv'.format(title), index=False)
print(len(result_df_semantic))
result_df_semantic.head()

# %%
# read the other csv file
original_file = 'prelim_results_semantic_{}.csv'.format(title)
df_original = pd.read_csv(original_file)
print(len(df_original))
df_original.head()

#%%
# merge with the new df obtained from manual search
merged_df = pd.concat([df_original, result_df_semantic], axis=0)

# sort by year (latest to earliest) and by citations (highest to lowest)
merged_df = merged_df.sort_values(by=['year', 'no_citations'], ascending=False)

# remove duplicates
merged_df.drop_duplicates(subset="title", keep='first', inplace=True)

merged_df.to_csv('final_results_{}.csv'.format(title), index=False)
print(len(merged_df))
merged_df.head()


# %%
