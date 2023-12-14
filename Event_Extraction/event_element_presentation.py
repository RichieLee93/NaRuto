import pickle, json
import pandas as pd
from nltk.wsd import lesk
from nltk.corpus import wordnet as wn
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")


# from pandarallel import pandarallel
# pandarallel.initialize(nb_workers=8)


def get_event_element(event, words):
    elements = {"verb": event["verb"], "event_id": event["event_id"]}
    ####get arguments
    argus_B = []
    argus_I = []
    arg_name = []
    for i in range(len(event["tags"])):
            if event["tags"][i] != "O" and "B-V" not in event["tags"][i] and "I-V" not in event["tags"][i]:
                if event["tags"][i][:2] == "B-":
                    arg_name.append(event["tags"][i][2:])
                    argus_B.append(i)
                if event["tags"][i][:2] == "I-":
                    argus_I.append(i)
    for B_i in range(len(argus_B)):
        temp = [argus_B[B_i]]
        if B_i != len(argus_B) -1:
            for I in argus_I:
                if I> argus_B[B_i] and I < argus_B[B_i+1]:
                    temp.append(I)
        else:
            for I in argus_I:
                if I > argus_B[B_i] and I<= len(event["tags"]):
                    temp.append(I)
        elements[arg_name[B_i]] = " ".join([words[t] for t in temp])

    return elements

def convert_event_to_element(df):
    if type(df["event"]) is str:
        df["event"] = eval(df["event"])
    if type(df["event_words"]) is str:
        df["event_words"] = eval(df["event_words"])
    return get_event_element(df["event"], df["event_words"])

def convert_ACE_to_element(df):
    if type(df["ACE"]) is str:
        df["ACE"] = eval(df["ACE"])
    if type(df["event_words"]) is str:
        df["event_words"] = eval(df["event_words"])
    all_events_elements = []
    for event in df["ACE"]:
        all_events_elements.append(get_event_element(event, df["event_words"]))
    return all_events_elements


def event_element(data):
    data["event_elements"] = data.apply(lambda x: convert_event_to_element(x), axis=1)
    data["ACE_elements"] = data.apply(lambda x: convert_ACE_to_element(x), axis=1)
    return data





