import pandas as pd
import numpy as np


def pearson(list1, list2):
    return np.corrcoef(list1, list2)[0][1]


def cos_sim(list1, list2):
    return np.dot(list1, list2) / ((np.linalg.norm(list1)) * (np.linalg.norm(list2)))


def create_data_frame(file_path, columns):
    data_frame = pd.read_csv(
        file_path,
        delimiter=';',
        error_bad_lines=False,
        encoding='ISO-8859-1',
        low_memory=False
    )

    data_frame.columns = columns
    return data_frame


def merge_and_drop(df1, df2, drop_list, onParameter):

    merged_frames = pd.merge(
        df1,
        df2,
        on=onParameter
    )

    merged_frames = merged_frames.drop(
        columns=drop_list
    )
    return merged_frames


def calc_most_similar_user(pivot_train, dict_train, isbn, user_id, k):
    most_similar_users = []
    case = 0
    flag = 0

    # handling key errors
    if user_id not in dict_train.keys():
        flag = 1
        case = 1
    if isbn not in list(dict_train.values())[0]:
        case = (3 if flag == 1 else 2)

    if case == 0:
        most_similar_users = [(id, pearson(list(pivot_train[user_id]), list(pivot_train[id])), dict_train[id][isbn]) for id in pivot_train if dict_train[id][isbn] != 0  and id != user_id]
        most_similar_users = sorted(most_similar_users, key=lambda tup: tup[1])
        most_similar_users = most_similar_users[::-1][0:k]

    return most_similar_users, case


def avg_user(dict_train, isbn):
    # if user id not in train data
    total = 0
    count = 0
    for user in dict_train:
        user_rate = dict_train[user][isbn]
        if user_rate != 0:
            total += user_rate
            count += 1

    return total / count


def avg_book(dict_train, user_id):
    # if isbn not in train data
    total = 0
    count = 0
    for isbn in dict_train[user_id]:
        rate = dict_train[user_id][isbn]
        if rate != 0:
            total += rate
            count += 1

    return total / count


def weighted_knn(most_similar_users):
    total_neighbour_rate = 0
    total_neighbour_sim = 0

    for i in range(len(most_similar_users)):

        neighbour_sim = most_similar_users[i][1]
        neighbour_rate = most_similar_users[i][2]

        if neighbour_rate != 0:
            total_neighbour_rate += neighbour_sim * neighbour_rate
            total_neighbour_sim += neighbour_sim

    return total_neighbour_rate / total_neighbour_sim


def knn(most_similar_users, k):
    # sum of rates that are belong to neighbours over k
    return sum(neighbour[2] for neighbour in most_similar_users if neighbour[2] != 0) / k


def predict(most_similar_users, case, dict_train, isbn, user_id, weighted, k):
    if case == 0:
        if weighted:
            return weighted_knn(most_similar_users)
        else:
            return knn(most_similar_users, k)

    if case == 1:
        # there is no user id
        return avg_user(dict_train, isbn)

    if case == 2:
        # there is no isbn
        return avg_book(dict_train, user_id)

    return "gg"



def main():

    train_path = './data/train-sklearn.csv'
    test_path = './data/test-sklearn.csv'

    # reading train.csv and creating dataframe
    df_train = create_data_frame(train_path, ['', 'user_id', 'ISBN', 'book_rating'])
    df_train = df_train.drop(columns='')

    # reading test.csv and creating dataframe
    df_test = create_data_frame(test_path, ['', 'user_id', 'ISBN', 'book_rating'])
    df_test = df_test.drop(columns='')

    print("##### Reading and dropping unnecessary columns are done. #####")

    # creating pivot table from dataset (train)
    # creating dictionary from pivot table (train)
    pivot_train = df_train.pivot_table(index='ISBN', columns='user_id', values='book_rating', fill_value=0)
    dict_train  = pivot_train.to_dict()

    total = 0
    count = 0
    for user in dict_train:
        for isbn in dict_train[user]:
            if dict_train[user][isbn] != 0:
                total += dict_train[user][isbn]
                count += 1
    not_found_both_avg = total / count

    print("##### Creating pivot table and dictionary from pivot table are done. #####")

    mae = 0
    count = 0
    hit_the_bullseye = 0
    not_found_id = 0
    not_found_isbn = 0
    not_found_both = 0

    """
     -loop over test dataset
     -find test users' k nearest neighbours from trained data
     -prediction step
     -key error: 
        *if dataset does not contain user that we try to predict
     -zerodivision error:
        *if user does not have nearest neighbour
    """
    k = 11               # nearest neighbour count
    if_weighted = True  # false => normal_knn  = = = = = true => weighted_knn
    for index, row in df_test.iterrows():

        user_id = row['user_id']
        isbn = row['ISBN']
        rating = row['book_rating']

        most_similar_users, case = calc_most_similar_user(pivot_train, dict_train, isbn, user_id, k)

        prediction = 0

        if case == 0:
            prediction = predict(most_similar_users, 0, dict_train, isbn, user_id, if_weighted, k)

        if case == 1:
            # There is no user id in train data
            prediction = predict(most_similar_users, 1, dict_train, isbn, user_id, if_weighted, k)
            not_found_id += 1

        if case == 2:
            # There is no ISBN in train data
            prediction = predict(most_similar_users, 2, dict_train, isbn, user_id, if_weighted, k)
            not_found_isbn += 1

        if case == 3:
            # There is no user id and ISBN in train data
            prediction = not_found_both_avg
            not_found_both += 1

        prediction = (10 if prediction > 10 else (1 if prediction < 0 else prediction))

        hit_the_bullseye += (1 if rating == prediction else 0)

        #print("Actual: ", rating, " Prediction: ", prediction)
        mae += abs(rating - prediction)
        count += 1

    print("-------------------------------------------")
    print("K =", k)
    print("Weighted_KNN" if if_weighted else "Normal_KNN")
    print("Prediction that are hit the bullseye:", hit_the_bullseye)
    print("There is no User ID in Train Data:", not_found_id)
    print("There is no ISBN in Train Data", not_found_isbn)
    print("There is no User ID and ISBN in Train Data", not_found_both)
    print("Mean Absolute Error Value:", mae/count)
    print("Total Prediction:", count)
    print("-------------------------------------------")


main()




# Code is used for cleaning data and dividing the data into two parts which are train and test
"""
    filename_users = './data/BX-Users.csv'
    filename_books = './data/BX-Books.csv'
    filename_ratings = './data/BX-Book-Ratings-Train.csv'

    dataframe_users = create_data_frame(filename_users, ['User-ID', 'Location', 'Age'])
    dataframe_books = create_data_frame(filename_books, ['ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher', 'Image-URL-S', 'Image-URL-M', 'Image-URL-L'])
    dataframe_ratings = create_data_frame(filename_ratings, ['User-ID', 'ISBN', 'Book-Rating'])

    dropList_books = ['Image-URL-S', 'Image-URL-M', 'Image-URL-L', 'Year-Of-Publication', 'Publisher', 'Book-Author']

    merge_ratings_books = merge_and_drop(dataframe_ratings, dataframe_books, dropList_books, 'ISBN')
    merge_ratings_books_users = merge_and_drop(merge_ratings_books, dataframe_users, ["Age"], 'User-ID')

    merge_ratings_books_users = merge_ratings_books_users[merge_ratings_books_users.Location.str.contains("usa|canada") == True]\
        .drop(columns = ["Location"])



    merge_ratings_books_users = merge_ratings_books_users[merge_ratings_books_users['Book-Rating']>0]


    frame_groupby_ISBN = merge_ratings_books_users.groupby(['ISBN']).filter(lambda x: (len(x) >= 2))
    frame_groupby_User_ID = frame_groupby_ISBN.groupby(['User-ID']).filter(lambda x: (len(x)) >= 2)

    pivot_table = frame_groupby_User_ID.pivot_table(index='ISBN', columns='User-ID', values='Book-Rating', fill_value=0)


    dict_table = pivot_table.to_dict()
"""