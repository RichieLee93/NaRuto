import pandas as pd

def get_one_two_para_predicates(df):
    all_predicates = df['precondition'].tolist() + df['effect'].tolist()

    one_para_conflict_term = []
    two_para_conflict_term = []
    for term_set in all_predicates:
        if type(term_set) is str:
            term_set = eval(term_set)
        for term in term_set:
            if len(term) == 2:
                one_para_conflict_term.append(term[0])
            else:
                two_para_conflict_term.append(term[0])
    for i in list(set(one_para_conflict_term)):
        print(("-".join(i.split(" ")), "?x"))
    for i in list(set(two_para_conflict_term)):
        print(("-".join(i.split(" ")), "?x ?y"))
    return one_para_conflict_term, two_para_conflict_term

def get_negations(df):
    if type(df['precondition']) is str:
        df["precondition"] = eval(df["precondition"])
    if type(df["effect"]) is str:
        df["effect"] = eval(df["effect"])
    neg_effect = []
    neg_precond = []
    if df["precondition"]:
        for p in df["precondition"]:
            if len(p) == 2 and p[0] in one_para_conflict_term_dict:
                for term in one_para_conflict_term_dict[p[0]]:
                    if [term] + p[1:] not in neg_precond:
                        neg_precond.append([term] + p[1:])
            if len(p) > 2 and p[0] in two_para_conflict_term_dict:
                for term in two_para_conflict_term_dict[p[0]]:
                    if [term] + p[1:] not in neg_precond:
                        neg_precond.append([term] + p[1:])
    if df["effect"]:
        for e in df["effect"]:
            if len(e) == 2 and e[0] in one_para_conflict_term_dict:
                for term in one_para_conflict_term_dict[e[0]]:
                    if [term] + e[1:] not in neg_effect:
                        neg_effect.append([term] + e[1:])
            if len(e) > 2 and e[0] in two_para_conflict_term_dict:
                for term in two_para_conflict_term_dict[e[0]]:
                    if [term] + e[1:] not in neg_effect:
                        neg_effect.append([term] + e[1:])
    neg_effect = [e for e in neg_effect if e not in neg_precond]
    return pd.Series([neg_effect, neg_precond])

from sentence_transformers import SentenceTransformer, util, CrossEncoder
sentence_contradict_model = CrossEncoder('cross-encoder/nli-deberta-v3-base')
def sent_contradict(sentence1, sentence2):

    scores = sentence_contradict_model.predict([(sentence1, sentence2)])
    #Convert scores to labels
    # label_mapping = ['contradiction', 'entailment', 'neutral']
    # labels = [label_mapping[score_max] for score_max in scores.argmax(axis=1)]
    if scores.argmax(axis=1)[0] == 0:
    # if labels[0] == 'contradiction':
        return True
    else:
        return False

def get_contradict_pair(l):
    pair = []
    for i in range(len(l)-1):
        for j in range(i+1, len(l)):
            if sent_contradict(l[i], l[j]) and (l[i], l[j]) not in pair:
                pair.append((l[i], l[j]))
    return pair



if __name__ == '__main__':
    df = pd.read_csv("/")
    one_para_conflict_term, two_para_conflict_term = get_one_two_para_predicates(df)

    one_para_conflict_term_pair = get_contradict_pair(one_para_conflict_term)
    two_para_conflict_term_pair = get_contradict_pair(two_para_conflict_term)

    one_para_conflict_term_dict = {}
    for pair in one_para_conflict_term_pair:
        if pair[0] not in one_para_conflict_term_dict:
            one_para_conflict_term_dict[pair[0]] = [pair[1]]
        else:
            one_para_conflict_term_dict[pair[0]].append(pair[1])
        if pair[1] not in one_para_conflict_term_dict:
            one_para_conflict_term_dict[pair[1]] = [pair[0]]
        else:
            one_para_conflict_term_dict[pair[1]].append(pair[0])

    two_para_conflict_term_dict = {}
    for pair in two_para_conflict_term_pair:
        if pair[0] not in two_para_conflict_term_dict:
            two_para_conflict_term_dict[pair[0]] = [pair[1]]
        else:
            two_para_conflict_term_dict[pair[0]].append(pair[1])
        if pair[1] not in two_para_conflict_term_dict:
            two_para_conflict_term_dict[pair[1]] = [pair[0]]
        else:
            two_para_conflict_term_dict[pair[1]].append(pair[0])


    df[["neg_effect", "neg_precondition"]] = df.apply(lambda x: get_negations(x), axis=1)
