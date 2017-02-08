#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import math
import collections
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def write_set_in_file(filename, set):
    file_to_write = open(filename, mode='w')
    for instance in set:
        file_to_write.write(str(instance[0]) + '\t' + str(instance[1]) + '\t' + str(instance[2]))
        file_to_write.write('\n')

    file_to_write.close()

#Get's the review text content.
def extract_review_content(rawLine):

    def convert_formatted_time_to_seconds(formatted_time):
        time_parts = formatted_time.strip().split('-')
        time = (
        datetime(int(time_parts[0]), int(time_parts[1]), int(time_parts[2])) - datetime(1970, 1, 1)).total_seconds()
        return time

    review = json.loads(rawLine.strip())
    review_id = review['review_id']
    item_id = review['business_id']
    user_id = review['user_id']
    rating = review['stars']
    votes = review['votes']
    votes_count = int(votes['funny']) + int(votes['useful']) + int(votes['cool'])
    timestamp = convert_formatted_time_to_seconds(review['date'])
    return (user_id, item_id, rating, votes_count, timestamp, review_id, review['text'])


#writes the set with review id.
def write_set(filename, instances):
    setToWrite = open(filename, mode='w')
    for instance in instances:
        setToWrite.write(str(instance[0]) + '\t' + str(instance[1]) + '\t' + str(instance[2]) + '\t' + str(instance[5]))
        setToWrite.write('\n')

    setToWrite.close()

def create_dataset_for_time_svd(source='../dataset/yelp_training_set_review.json'):
    reviews_by_user = {}
    for line in open(source, mode='r'):
        review_parts = extract_review_content(line)
        user_id = review_parts[0]
        reviews_from_user = reviews_by_user.get(user_id, [])
        reviews_from_user.append(review_parts)
        reviews_by_user[user_id] = reviews_from_user

    training_set = []
    test_set = []

    for user_id, reviews_from_user in reviews_by_user.items():
        reviews_from_user_sorted = sorted(reviews_from_user, key=lambda x: x[3])
        num_reviews_for_user = len(reviews_from_user_sorted)
        for i in range(0, num_reviews_for_user):
            if (1.0 * i) / num_reviews_for_user <= 0.8:
                training_set.append(reviews_from_user_sorted[i])
            else:
                test_set.append(reviews_from_user_sorted[i])


    write_set('training.txt', training_set)
    write_set('test.txt', test_set)


def get_items_with_less_than_N_reviews_and_min_S_stars(items_counter, item_vote_counts, n=1, S = 5):
    items_with_reviews_counts_less_than_N_and_reviews_stars_more_than_S = \
        {item: vote_counts for item, vote_counts in item_vote_counts.iteritems()
         if items_counter[item] <= n and max(vote_counts) >= S}

    return items_with_reviews_counts_less_than_N_and_reviews_stars_more_than_S.keys()


def create_dataset_for_conv_net_learning(source='../dataset/yelp_training_set_review.json'):
    items = []
    all_reviews = []
    vote_counts_for_items = {}
    for line in open(source, mode='r'):
        review_parts = extract_review_content(line)
        review_text = review_parts[6]
        if review_text is not None and len(review_text) > 0:
            all_reviews.append(review_parts)
            item_id = review_parts[1]
            votes_count = review_parts[3]
            items.append(item_id)
            votes_for_item = vote_counts_for_items.get(item_id, [])
            votes_for_item.append(votes_count)
            vote_counts_for_items[item_id] = votes_for_item

    items_counts = collections.Counter(items)

    items_to_take_from_both_halves = int(math.ceil(len(items_counts.keys()) * 0.1))

    items_with_less_than_4_reviews_for_test = get_items_with_less_than_N_reviews_and_min_S_stars(items_counts, vote_counts_for_items, 4, 5)

    starting_index = items_to_take_from_both_halves - (2 * items_to_take_from_both_halves - len(items_with_less_than_4_reviews_for_test))
    items_with_most_reviews_for_test = [item[0] for item in items_counts.most_common(items_to_take_from_both_halves)[starting_index:]]

    training_set = []
    test_set_for_items_with_few_reviews = []
    test_set_for_items_with_much_reviews = []
    for review in all_reviews:
        item_id = review[1]
        if item_id in items_with_less_than_4_reviews_for_test:
            test_set_for_items_with_few_reviews.append(review)
        elif item_id in items_with_most_reviews_for_test:
            test_set_for_items_with_much_reviews.append(review)
        else:
            training_set.append(review)

    write_set_in_file('training_set.txt', training_set)
    write_set_in_file('test_set_for_items_with_few_reviews.txt', test_set_for_items_with_few_reviews)
    write_set_in_file('test_set_for_items_with_much_reviews.txt', test_set_for_items_with_much_reviews)
    write_set_in_file('test_set_whole.txt',test_set_for_items_with_few_reviews +  test_set_for_items_with_much_reviews)

    #write set with review id informartion
    write_set('training_set_with_review_id.txt', training_set)


create_dataset_for_conv_net_learning()


