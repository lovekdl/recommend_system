import pandas as pd
import math
def load_movie_mapping(mapping_path):
    mapping_df = pd.read_csv(mapping_path)
    
    mapping_df['tmdbId'] = pd.to_numeric(mapping_df['tmdbId'], errors='coerce')
    
    movieid_to_imdbid = dict(zip(mapping_df['movieId'], mapping_df['tmdbId']))
    return movieid_to_imdbid

def convert(id, mapping_dict):
    tmdb_id = mapping_dict.get(id)
    if not math.isnan(tmdb_id):
        return int(tmdb_id)
    return -1

def convert_movieid_to_imdbid(recommendations, mapping_dict):
    converted_recommendations = {}
    for user_id, movie_ids in recommendations.items():
        tmdb_ids = []
        for movie_id in movie_ids:
            if movie_id in mapping_dict:
                tmdb_id = mapping_dict.get(movie_id)
                if not math.isnan(tmdb_id):
                    tmdb_ids.append(int(tmdb_id))
        converted_recommendations[user_id] = tmdb_ids
    return converted_recommendations