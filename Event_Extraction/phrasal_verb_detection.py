import string, pickle
import pandas as pd
from nltk.stem import WordNetLemmatizer
wn_lemma = WordNetLemmatizer()
# from pycorenlp import StanfordCoreNLP
#
# nlp = StanfordCoreNLP('http://localhost:9000')


# from pandarallel import pandarallel
# pandarallel.initialize(nb_workers=8)


def get_phrasal_verbs(df, pv_wiki):

    all_verb_id_index = {df["event"]["event_id"]: [i+1 for i in range(len(df["event"]["tags"])) if df["event"]["tags"][i] == "B-V"][0]}
    phrasal_verbs = {df["event"]["event_id"]: ""}
    if df["ACE"]:
        for se in df["ACE"]:
             all_verb_id_index[se["event_id"]] = [i+1 for i in range(len(se["tags"])) if se["tags"][i] == "B-V"][0]
             phrasal_verbs[se["event_id"]] = ""
    ######based on prep relation detect phrasal verbs
    case_dep = []
    for dep in df["event_words_dep"]:
        if dep[0] == "case" or dep[0] == "mark":
            case_dep.append(dep)
    if case_dep:
        the_other_ele = []
        for event_id, vp in all_verb_id_index.items():
            for dep in df["event_words_dep"]:
                if vp == dep[-1][0]:
                    vp_word = dep[1]
                    the_other_ele.append(dep[-1][1])
            if the_other_ele:
                for case in case_dep:
                    for e in the_other_ele:
                        if e == case[-1][0] and vp not in case:
                            if vp+1 == case[-1][1] or vp-1 == case[-1][1]:
                                element2 = wn_lemma.lemmatize(case[2]).lower()
                                element1 = wn_lemma.lemmatize(vp_word, "v").lower()
                                if " ".join([element1, element2]) in pv_wiki:
                                    phrasal_verbs[event_id] = " ".join([element1, element2])
    ###check if phrasal verbs exist based on the compound:prt relation
    prt_dep = []
    for dep in df["event_words_dep"]:
        if dep[0] == "compound:prt":
            prt_dep.append(dep)
    if prt_dep:
        for event_id, vp in all_verb_id_index.items():
            for dep in prt_dep:
                if vp in dep[-1]:
                    element1 = wn_lemma.lemmatize(dep[1], "v").lower()
                    element2 = wn_lemma.lemmatize(dep[2]).lower()
                    if " ".join([element1, element2]) in pv_wiki:
                        phrasal_verbs[event_id] = " ".join([element1, element2])
    return phrasal_verbs


def get_overall_phrasal_verbs(df, pv_wiki):
    if type(df["event_words_dep"]) is str:
        df["event_words_dep"] = eval(df["event_words_dep"])
    if type(df["event"]) is str:
        df["event"] = eval(df["event"])
    if type(df["ACE"]) is str:
        df["ACE"] = eval(df["ACE"])

    phrasal_verb = get_phrasal_verbs(df, pv_wiki)

    return phrasal_verb



def phrasal_verb(data):
    pv_wiki = pd.read_csv("./phrasal_verb_list.csv")["wiki_pv"].tolist() ## download from wikipedia
    data["event_phrasal_verbs"] = data.apply(lambda x: get_overall_phrasal_verbs(x, pv_wiki), axis=1)
    return data



