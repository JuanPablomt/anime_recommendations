import pandas as pd
import numpy as np
import joblib
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate

from fastapi import FastAPI, HTTPException
from typing import Optional
from pydantic import BaseModel

import warnings; warnings.simplefilter('ignore')


app = FastAPI()

# if traing make sure to have rating_df = pd.read_csv("./datasets/rating_complete.csv")
# and have rating_complete localy and change filepath accordingly
skip_train = True

print("skip training: " + str(skip_train))

anime = "https://drive.google.com/file/d/1yo4ZdHoS-j5dJKUF8TRDt3_TQ0KnV3wb/view?usp=sharing"
anime = "https://drive.google.com/uc?id=" + anime.split("/")[-2]

anime_with_synopsis = "https://drive.google.com/file/d/1R7tnsvZvif5iaMzJ7L8U3KdveZeCYXu7/view?usp=sharing"
anime_with_synopsis = "https://drive.google.com/uc?id=" + anime_with_synopsis.split("/")[-2]

rating_complete = "https://drive.google.com/u/0/uc?id=1ohR2nBdb_Dj6ywydBtJtZk96qlQaPCc8&export=download&confirm=t&uuid=fe5baf63-ecd4-4459-90fd-a565e203acb5"

anime_info_df = pd.read_csv(anime)
anime_desc_df = pd.read_csv(anime_with_synopsis)
rating_df = pd.read_csv(rating_complete)

anime_df = pd.merge(anime_desc_df,anime_info_df[['MAL_ID','Type','Popularity','Members','Favorites']],on='MAL_ID')

anime_df = anime_df[(anime_df["Score"] != "Unknown")] 

anime_df['sypnopsis'] = anime_df['sypnopsis'].fillna('')

tfidf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
tfidf_matrix = tfidf.fit_transform(anime_df['sypnopsis'])

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

anime_df = anime_df.reset_index()
titles = anime_df['Name']
indices = pd.Series(anime_df.index, index=anime_df['Name'])

def content_recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]
    anime_indices = [i[0] for i in sim_scores]
    
    anime_lst = anime_df.iloc[anime_indices][['Name', 'Members', 'Score']]
    favorite_count = anime_lst[anime_lst['Members'].notnull()]['Members'].astype('int')
    score_avg = anime_lst[anime_lst['Score'].notnull()]['Score'].astype('float')
    C = score_avg.mean()
    m = favorite_count.quantile(0.60)
    qualified = anime_lst[(anime_lst['Members'] >= m) & (anime_lst['Members'].notnull()) & (anime_lst['Score'].notnull())]
    qualified['Members'] = qualified['Members'].astype('int')
    qualified['Score'] = qualified['Score'].astype('float')
    def weighted_rating(x):
        v = x['Members']
        R = x['Score']
        return (v/(v+m) * R) + (m/(m+v) * C)   
    
    qualified['wr'] = qualified.apply(weighted_rating, axis=1)
    qualified = qualified.sort_values('wr', ascending=False).head(10)
    
    return qualified


svd = SVD()
filename = 'finalized_model.sav'

if not skip_train:
    rating_df = rating_df.drop(rating_df.index[100_000: ])
    rating_df = rating_df[(rating_df["rating"] != -1)] 
    reader = Reader()
    rating_data = Dataset.load_from_df(rating_df, reader)
    print("training")
    trainset = rating_data.build_full_trainset()
    svd.fit(trainset)
    joblib.dump(svd, filename)

if skip_train:
    svd = joblib.load(filename)

print("model done")

id_map = anime_df[['MAL_ID']]
id_map['id'] = list(range(1,anime_df.shape[0]+1,1))
id_map = id_map.merge(anime_df[['MAL_ID', 'Name']], on='MAL_ID').set_index('Name')

indices_map = id_map.set_index('id')

def hybrid_recommendations(user_id,title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]    
    anime_indices = [i[0] for i in sim_scores]
            
    anime_lst = anime_df.iloc[anime_indices][['MAL_ID','Name', 'Members', 'Score','Genres']]
    favorite_count = anime_lst[anime_lst['Members'].notnull()]['Members'].astype('int')
    score_avg = anime_lst[anime_lst['Score'].notnull()]['Score'].astype('float')
    C = score_avg.mean()
    m = favorite_count.quantile(0.60)
    qualified = anime_lst[(anime_lst['Members'] >= m) & (anime_lst['Members'].notnull()) & (anime_lst['Score'].notnull())]    
    qualified['Members'] = qualified['Members'].astype('int')
    qualified['Score'] = qualified['Score'].astype('float')
    def weighted_rating(x):
        v = x['Members']
        R = x['Score']
        return (v/(v+m) * R) + (m/(m+v) * C)   
    
    qualified['wr'] = qualified.apply(weighted_rating, axis=1)
    qualified = qualified.sort_values('wr', ascending=False).head(30)    
    
    qualified['id'] = list(range(1,qualified.shape[0]+1,1))  
    qualified['est'] = qualified['id'].apply(lambda x: svd.predict(user_id, indices_map.loc[x]['MAL_ID']).est)
    qualified = qualified.sort_values('est', ascending=False)
    result = qualified[['MAL_ID','Name','Genres','Score']]
    return result.head(10)    


list_of_tuples = [(x,y) for x,y in zip(rating_df['user_id'], rating_df['anime_id'])]

user_dict = {}
for entry in list_of_tuples:
    if entry[0] not in user_dict:
        user_dict[entry[0]] = [entry[1]]
    else:
        user_dict[entry[0]].append(entry[1])

@app.get('/animes/{malID}/{animeName}')
def user_detail(malID: int, animeName: str):
    recommendations = hybrid_recommendations(malID, animeName)
    aray_name = recommendations['Name']
    main_anime = [malID, animeName]
    searched = anime_df.loc[anime_df['Name'] == main_anime[1]]
    # validation
    id_main = searched['MAL_ID'].values[0]

    users_have_seen_main = {}
    for user in user_dict.items():
        if id_main in user[1]:
            users_have_seen_main[user[0]] = user[1]
            
    ids_of_recom = list(recommendations['MAL_ID'].values)
    score = []
    for recom in ids_of_recom:
        count = 0
        for user in users_have_seen_main.items():
            if recom in user[1]:
                count+=1    
        # search_name = anime_df.loc[anime_df['MAL_ID'] == recom]["Name"].values[0]
        score.append(count/len(users_have_seen_main))

    # print(type(aray), type(score))
    # 'Recommendations': aray, 
    
    return {'Recommendations': aray_name, 'Scores': score}