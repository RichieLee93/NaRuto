import string
import pandas as pd
import pickle
from pycorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP('http://localhost:9000')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

def get_all_possible_sent(all_sent):
    sent_pos_dep_dict = {}
    for original_sent in all_sent:
        sent = " ".join(eval(original_sent))
        dep = nlp.annotate(sent.replace("%", ""),
            properties={
                'annotators': 'depparse',
                'tokenize.language': 'Whitespace',
                'outputFormat': 'json',
                'timeout': 10000,
            })
        NN_list = []
        word_len = []
        dependencyRel = []
        dependencyRel_basic = []
        if len(dep["sentences"]) > 0:
            for sent in dep["sentences"]:
                word_len.append(len(sent["tokens"]))
                for i in sent["tokens"]:
                    if i["pos"] not in string.punctuation:
                            NN_list.append([i["originalText"], i["characterOffsetBegin"], i["characterOffsetEnd"], i["pos"]])
                for dep_pair in sent["enhancedPlusPlusDependencies"]:
                    if "punct" not in dep_pair["dep"]:
                        if len(word_len) > 1:
                            dependencyRel.append([dep_pair["dep"], dep_pair["governorGloss"], dep_pair["dependentGloss"], [dep_pair["governor"] + sum(word_len[:-1]), dep_pair["dependent"] + sum(word_len[:-1])]])
                        else:
                            dependencyRel.append([dep_pair["dep"], dep_pair["governorGloss"], dep_pair["dependentGloss"], [dep_pair["governor"], dep_pair["dependent"]]])
                for dep_pair in sent["basicDependencies"]:
                    if "punct" not in dep_pair["dep"]:
                        if len(word_len) > 1:
                            dependencyRel_basic.append([dep_pair["dep"], dep_pair["governorGloss"], dep_pair["dependentGloss"], [dep_pair["governor"] + sum(word_len[:-1]), dep_pair["dependent"] + sum(word_len[:-1])]])
                        else:
                            dependencyRel_basic.append([dep_pair["dep"], dep_pair["governorGloss"], dep_pair["dependentGloss"], [dep_pair["governor"], dep_pair["dependent"]]])

        else:
            dependencyRel = 0
            dependencyRel_basic = 0
        sent_pos_dep_dict[original_sent] = [NN_list, dependencyRel, dependencyRel_basic]

    return sent_pos_dep_dict

def get_pos_dep_for_NN(df, all_sent_pos_dep_dict):
    df = str(df)
    return pd.Series(all_sent_pos_dep_dict[df])


def check_main_event(df, modal_list):
    if type(df["event"]) is str:
        df["event"] = eval(df["event"])
    if type(df["event_words_POS"]) is str:
        df["event_words_POS"] = eval(df["event_words_POS"])
    flag = 0
    for pos_tag in df["event_words_POS"]:
        if df["event"]["verb"] == pos_tag[0] and pos_tag[-1][0] == "V" and lemmatizer.lemmatize(df["event"]["verb"]) not in modal_list:
            flag = 1
    return flag

def remove_not_v_event_ACE(df, modal_list):
    if type(df["ACE"]) is str:
        df["ACE"] = eval(df["ACE"])
    if type(df["event_words_POS"]) is str:
        df["event_words_POS"] = eval(df["event_words_POS"])
    possible_V_candidate = [tag for tag in df["event_words_POS"] if tag[-1][0] == "V"]
    updated_ACE = []

    if df["ACE"]:
        for event in df["ACE"]:
            flag = False
            for tag in possible_V_candidate:
                if event["verb"] == tag[0] and lemmatizer.lemmatize(event["verb"]) not in modal_list:
                    flag = True
            if flag:
                updated_ACE.append(event)
    return updated_ACE


def filter_not_main_event(df):
    #### filter out all events that are subevents of others
    all_doc_id = list(dict.fromkeys(df["d_id"].tolist()))
    flag = []
    for d_id in all_doc_id:
        df_selected = df[df["d_id"] == d_id]
        all_main_events = df_selected["event"].tolist()
        all_ACE_events = []
        for i in df_selected["ACE"].tolist():
            if type(i) is str:
                i = eval(i)
            if i:
                for ace in i:
                    all_ACE_events.append(str(ace))
        for main_event in all_main_events:
            if main_event in all_ACE_events:
                flag.append(0)
            else:
                flag.append(1)
    df["is_actually_subevent"] = flag
    df = df[df["is_actually_subevent"] == 1]
    modal_list = ["can", "will", "may", "shall", "would", "could", "might", "must", "need",
                  "ought", "should"]
    df["if_event_real"] = df.apply(lambda x: check_main_event(x, modal_list), axis=1)
    df = df[df["if_event_real"] == 1]
    del df["if_event_real"]
    df["updated_ACE"] = df.apply(lambda x: remove_not_v_event_ACE(x, modal_list), axis=1)
    del df["ACE"]
    df.rename(columns={"updated_ACE": "ACE"},  inplace=True)
    # print("after remove those fake main events, the data size is ", df.shape)
    return df


def POS_DEP_tagging(data):

    all_sent = data["event_words"].tolist()
    all_sent = [str(s) for s in all_sent]
    distinct_sent = list(set(all_sent))
    # print("need to parse {} sents".format(len(distinct_sent)), len(all_sent))
    all_sent_pos_dep_dict = get_all_possible_sent(distinct_sent)
    # print("finished parsing", len(all_sent_pos_dep_dict))
    data[["event_words_POS", "event_words_dep", "event_words_dep_basic"]] = data["event_words"].apply(lambda x: get_pos_dep_for_NN(x, all_sent_pos_dep_dict))
    data = filter_not_main_event(data)
    data = data.drop(columns=['is_actually_subevent'])
    return data



