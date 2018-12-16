import os
import requests
import json
import time
import csv
import app_name_dict

def getAllReviews(appID, f, page):
    url = 'https://itunes.apple.com/rss/customerreviews/id={}/page={}/sortby=mostrecent/json'.format(appID, page)

    #assume correct appID given
    data = requests.get(url)
    print("Response received")

    try:
        json_data = data.json().get('feed')
    except ValueError:
        return

    print("Getting reviews from page {}".format(page))
    if json_data.get('entry') == None:
        page += 1
        getAllReviews(appID, f, page)
        return
    for json_review in json_data.get('entry'):
        if json_review.get('im:name'): continue
        review_id = json_review.get('id').get('label').replace(",", " ")
        title = json_review.get('title').get('label').replace(",", " ")
        author = json_review.get('author').get('name').get('label').replace(",", " ")
        author_url = json_review.get('author').get('uri').get('label').replace(",", " ")
        version = json_review.get('im:version').get('label').replace(",", " ")
        rating = json_review.get('im:rating').get('label').replace(",", " ")
        review = json_review.get('content').get('label').replace("\n", " ").replace(",", " ")
        vote_count = json_review.get('im:voteCount').get('label').replace(",", " ")
        if (int(rating) > 3):
            sentiment = '+'
        else:
            sentiment = '-'
        try:
            f.write('"'+author+'","'+review+'","'+sentiment+'","'+rating+'","'+title+'",'+review_id+',"'+author_url+'",'+version+','+vote_count+'\n')
        except UnicodeEncodeError :
            continue

    page += 1
    getAllReviews(appID, f, page)


for category_name, category_dict in app_name_dict.name_to_dict.items():
    filename = category_name + '.csv'
    f = open(filename, 'w')
    print("=============================")
    print("GETTING {}".format(filename))
    print("=============================")
    for app_id, app_name in category_dict.items():
        page = 1
        # order of csv file with delimite = ','
        # author,comment,sentiment,rating,review_id,title,author_url,version,vote_count

        print("=======Retrieving reviews for...{}=======".format(app_name))
        #write all the reviews found into csv file
        getAllReviews(app_id, f, page)
    f.close()
    print("=============================")
    print("=============================")
