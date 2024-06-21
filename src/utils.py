import pandas as pd

def load_movie_mapping(mapping_path):
    mapping_df = pd.read_csv(mapping_path)
    movieid_to_imdbid = dict(zip(mapping_df['movieId'], mapping_df['imdbId']))
    return movieid_to_imdbid

def convert_movieid_to_imdbid(recommendations, mapping_dict):
    converted_recommendations = {}
    for user_id, movie_ids in recommendations.items():
        imdb_ids = [mapping_dict.get(movie_id) for movie_id in movie_ids if movie_id in mapping_dict]
        converted_recommendations[user_id] = imdb_ids
    return converted_recommendations