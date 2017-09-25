import pandas as pd
import numpy as np
import graphlab as gl


if __name__ == "__main__":
    sample_sub_fname = "data/sample_submission.csv"
    ratings_data_fname = "data/ratings.dat"
    output_fname = "data/our_test_ratings.csv"

    ratings = gl.SFrame(ratings_data_fname, format='tsv')
    sample_sub = pd.read_csv(sample_sub_fname)
    for_prediction = gl.SFrame(sample_sub)
    rec_engine = gl.ranking_factorization_recommender.create(observation_data=ratings,
                                                     user_id="user_id",
                                                     item_id="joke_id",
                                                     target='rating',
                                                     solver='auto',
                                                     ranking_regularization=0,
                                                     num_factors=16)

    sample_sub.rating = rec_engine.predict(for_prediction) #update with ratings
    sample_sub.to_csv(output_fname, index=False)
