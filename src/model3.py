import pandas as pd
import numpy as np
import graphlab as gl


if __name__ == "__main__":
    sample_sub_fname = "data/sample_submission.csv"
    ratings_data_fname = "data/ratings.dat"
    side_features_fname = "side_features.csv"
    output_fname = "data/our_test_ratings.csv"

    ratings = gl.SFrame(ratings_data_fname, format='tsv')
    sf = pd.read_csv(side_features_fname)
    side_features = gl.SFrame(sf)
    sample_sub = pd.read_csv(sample_sub_fname)
    for_prediction = gl.SFrame(sample_sub)

    train, test = gl.recommender.util.random_split_by_user(ratings, user_id='user_id', item_id='joke_id')
    # rec_eng = [32, 64, 128]
    # item_data = [side_features, None]


    rec_eng_128 = gl.ranking_factorization_recommender.create(observation_data=train,
                                                     user_id="user_id",
                                                     item_id="joke_id",
                                                     target='rating',
                                                     side_data_factorization=True,
                                                     item_data=side_features,
                                                     solver='auto',
                                                     num_factors=128)

    rec_eng_32 = gl.ranking_factorization_recommender.create(observation_data=train,
                                                     user_id="user_id",
                                                     item_id="joke_id",
                                                     target='rating',
                                                     side_data_factorization=True,
                                                     solver='auto',
                                                     num_factors=32
                                                     )
    rec_eng_128_sf = gl.ranking_factorization_recommender.create(observation_data=train,
                                                     user_id="user_id",
                                                     item_id="joke_id",
                                                     target='rating',
                                                     item_data=side_features,
                                                     solver='auto',
                                                     num_factors=128)

    rec_eng_32_sf = gl.ranking_factorization_recommender.create(observation_data=train,
                                                     user_id="user_id",
                                                     item_id="joke_id",
                                                     target='rating',
                                                     side_data_factorization=True,
                                                     item_data=side_features,
                                                     solver='auto',
                                                     num_factors=32)


    pr = gl.recommender.util.compare_models(test, [rec_eng_32, rec_eng_128, rec_eng_32_sf, rec_eng_128_sf], model_names=['128 no side features', '32 no side features', '128 side features', '32 side features'], metric='precision_recall')


    # sample_sub.rating = rec_engine.predict(for_prediction) #update with ratings
    # sample_sub.to_csv(output_fname, index=False)
