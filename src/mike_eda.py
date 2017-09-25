import graphlab
import pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', -1)

actual_data = pd.read_csv('data/dont_use.csv')
train_data = pd.read_table('data/ratings.dat')


# get data means and user means
def get_data_info(data):
    mean = data['rating'].mean()
    user_means = data.groupby(['user_id'])['rating'].mean()
    return mean, user_means

train_mean, train_user_mean = get_data_info(train_data)
actual_mean, actual_user_mean = get_data_info(actual_data)


# plot histograms and violin plots
def plots(train_data, actual_data):
    train_data.hist()
    plt.show()
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(6, 6))
    ax[0].violinplot(train_data.values, showmeans=True,
                     showextrema=True, showmedians=True)
    ax[0].set_ylim([-10, 10])
    ax[1].violinplot(actual_data.values, showmeans=True,
                     showextrema=True, showmedians=True)
    plt.show()

plots(train_user_mean, actual_user_mean)

rating_counts = train_data.groupby(['user_id']).count()



rating_counts.head()

rating_counts.hist()
plt.show()
