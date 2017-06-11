import numpy as np

glove = {}
with open('glove.6B.50d.txt', 'r') as f:
    for line in f.readlines():
        pieces = line.strip().split(' ')
        glove[pieces[0]] = np.array([float(i) for i in pieces[1:]])

def word_vec(word):
    if glove[word]:
        return glove[word]
    if word == "destination1":
        return glove["destination"]
    if word == "origin1":
        return glove["origin"]
    if word == "flightDate2":
        return (glove["arrival"] + glove["date"]) / 2
    if word == "flightDate1":
        return (glove["departure"] + glove["date"]) / 2
    if word == "taskcomplete":
        return (glove["task"] + glove["complete"]) / 2
    if word == "critic_rating":
        return (glove["critic"] + glove["rating"]) / 2
    if word == "distanceconstraints":
        return (glove["distance"] + glove["constraints"]) / 2
    if word == "implicit_value":
        return (glove["implicit"] + glove["value"]) / 2
    if word == "movie_series":
        return (glove["movie"] + glove["series"]) / 2
    if word == "moviename":
        return (glove["movie"] + glove["name"]) / 2
    if word == "mpaa_rating":
        return glove["rating"]
    if word == "numberofpeople":
        return (glove["number"] + glove["of"] + glove["people"]) / 3
    if word == "numberofkids":
        return (glove["number"] + glove["of"] + glove["kids"]) / 3
    if word == "starttime":
        return (glove["start"] + glove["time"]) / 2
    if word == "theater_chain":
        return (glove["theater"] + glove["chain"]) / 2
    if word == "video_format":
        return (glove["video"] + glove["format"]) / 2
    if word == "mc_list":
        return glove['list']
    if word == "confirm_question":
        return (glove['confirm'] + glove['question']) / 2
    if word == "confirm_answer":
        return (glove['confirm'] + glove['answer']) / 2
    if word == "multiple_choice":
        return (glove['multiple'] + glove['choice']) / 2
    if word == "not_sure":
        return (glove['not'] + glove['sure']) / 2
