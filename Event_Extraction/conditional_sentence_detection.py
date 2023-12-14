import string, pickle
import pandas as pd

from pycorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP('http://localhost:9000')


def convert_pos_index_position(event_words_POS, event, event_words):
    index_pos_dict = {}
    for index in range(len(event["tags"])):
        if index == 0:
            first_token_index = 0
        else:
            first_token_index = len(" ".join(event_words[:index])) + 1

        for pos in event_words_POS:
                if pos[1] == first_token_index:
                    index_pos_dict[index] = pos[-1]

    return index_pos_dict


def find_sub_list(sl,l):
    sll=len(sl)
    all_sub = []
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if l[ind:ind+sll]==sl:
            all_sub.append((ind,ind+sll-1))
    return all_sub



def check_if_sent_match_structure(event1_id, event2_id, event_verb_id, event_dict, sent_word, sent_POS):

    flag = 0

    #### event words ==sent word
    # cue_short = [cue for cue in ["if", "unless", "whenever"] if cue in list(map(str.lower,sent_word))]
    cue_short = [cue for cue in ["if", "whenever"] if cue in list(map(str.lower,sent_word))]
    cue_long = [cue for cue in ["as long as", "on the condition that", "provided that"] if cue in " ".join(list(map(str.lower,sent_word)))]
    if cue_short or cue_long:

        all_cue_index = []
        if cue_short:
            for cue in cue_short:
                start = [i for i,d in enumerate(sent_word) if d==cue]
                for s in start:
                    all_cue_index.append((s, s))
        if cue_long:
            for cue in cue_long:
                poss_indices = find_sub_list(cue.split(), list(map(str.lower,sent_word)))
                for (s,e) in poss_indices:
                    all_cue_index.append((s, e))
        pos_position_index_dict_sub = convert_pos_index_position(sent_POS, event_dict[event1_id], sent_word)
        for (cue_index_start, cue_index_end) in all_cue_index:
            contained_event1_distance = len([index for index in event_verb_id.values() if (cue_index_start < index < event_verb_id[event1_id] or cue_index_start > index > event_verb_id[event1_id]) and index != event_verb_id[event2_id]])
            contained_event2_distance = len([index for index in event_verb_id.values() if (cue_index_start < index < event_verb_id[event2_id] or cue_index_start > index > event_verb_id[event2_id]) and index != event_verb_id[event1_id]])
            #####for xxxifxxx
            if contained_event1_distance == 0 and contained_event2_distance == 0:
                if (event_verb_id[event1_id] < cue_index_start <= cue_index_end < event_verb_id[event2_id] or cue_index_end < event_verb_id[event2_id] <event_verb_id[event1_id]) and event_verb_id[event1_id] in pos_position_index_dict_sub and event_verb_id[event2_id] in pos_position_index_dict_sub:
                    if pos_position_index_dict_sub[event_verb_id[event2_id]] == "VBP" or pos_position_index_dict_sub[event_verb_id[event2_id]] == "VBZ":
                        ###if satisfy rule 13
                        if pos_position_index_dict_sub[event_verb_id[event1_id]] == "VBP" or pos_position_index_dict_sub[event_verb_id[event1_id]] == "VBZ":
                            flag = 13
                        ###if satisfy rule 1
                        if pos_position_index_dict_sub[event_verb_id[event1_id]] == "VB" and event_verb_id[event1_id] -1 in pos_position_index_dict_sub and pos_position_index_dict_sub[event_verb_id[event1_id] -1] == "MD":
                            flag = 1
                        if pos_position_index_dict_sub[event_verb_id[event1_id]] == "VB" and event_verb_id[event1_id] -1 in pos_position_index_dict_sub and pos_position_index_dict_sub[event_verb_id[event1_id] -1] == "RB" and event_verb_id[event1_id] -2 in pos_position_index_dict_sub and pos_position_index_dict_sub[event_verb_id[event1_id] -2] == "MD":
                            flag = 1
                        ###if satisfy rule 2
                        if event_verb_id[event1_id] >= 1 and (sent_word[event_verb_id[event1_id] -1].lower() == "might" or sent_word[event_verb_id[event1_id] -1].lower() == "may"):
                            flag = 2
                        ###if satisfy rule 3
                        if event_verb_id[event1_id] >= 1 and (sent_word[event_verb_id[event1_id] -1].lower() == "must" or sent_word[event_verb_id[event1_id] -1].lower() == "should"):
                            flag = 3
                    ###if satisfy rule 4
                    if pos_position_index_dict_sub[event_verb_id[event2_id]] == "VBD":
                        if "would" in sent_word[:event_verb_id[event1_id]] and event_verb_id[event1_id] >=1 and sent_word[event_verb_id[event1_id] -1].lower() == "to" and pos_position_index_dict_sub[event_verb_id[event1_id]] == "VB":
                            flag = 4
                        if "would" in sent_word[:event_verb_id[event1_id]] and event_verb_id[event1_id] >=2 and sent_word[event_verb_id[event1_id] -2].lower() == "to" and pos_position_index_dict_sub[event_verb_id[event1_id]] == "VB" and sent_word[event_verb_id[event1_id] -1] == "RB":
                            flag = 4
                        ###if satisfy rule 5

                        if "might" in sent_word[:event_verb_id[event1_id]] or "could" in sent_word[:event_verb_id[event1_id]]:
                            flag = 5
                    ###if satisfy rule 6
                    if (pos_position_index_dict_sub[event_verb_id[event2_id]] == "VBG" and event_verb_id[event2_id] >=1 and event_verb_id[event2_id] - 1 in pos_position_index_dict_sub and pos_position_index_dict_sub[event_verb_id[event2_id] - 1] == "VBD") or \
                            (pos_position_index_dict_sub[event_verb_id[event2_id]] == "VBG" and event_verb_id[event2_id] >=2 and event_verb_id[event2_id] - 2 in pos_position_index_dict_sub and pos_position_index_dict_sub[event_verb_id[event2_id] - 2] == "VBD" and event_verb_id[event2_id] - 1 in pos_position_index_dict_sub and pos_position_index_dict_sub[event_verb_id[event2_id] - 1] == "RB"):
                        if "would" in sent_word[:event_verb_id[event1_id]] and event_verb_id[event1_id] >=1 and sent_word[event_verb_id[event1_id] -1].lower() == "to" and pos_position_index_dict_sub[event_verb_id[event1_id]] == "VB":
                            flag = 6
                        if "would" in sent_word[:event_verb_id[event1_id]] and event_verb_id[event1_id] >=2 and sent_word[event_verb_id[event1_id] -2].lower() == "to" and pos_position_index_dict_sub[event_verb_id[event1_id]] == "VB" and sent_word[event_verb_id[event1_id] -1] == "RB":
                            flag = 6
                    # print(777, pos_position_index_dict_sub)
                    # print(888, pos_position_index_dict_sub[all_verb_id_index[pair[1]]])
                    # print(999, pos_position_index_dict_sub[all_verb_id_index[pair[1] - 1]])
                    ###if satisfy rule 7
                    if (pos_position_index_dict_sub[event_verb_id[event2_id]] == "VBN" and event_verb_id[event2_id] >= 1 and event_verb_id[event2_id] - 1 in pos_position_index_dict_sub and pos_position_index_dict_sub[event_verb_id[event2_id] - 1] == "VBD") or \
                            (pos_position_index_dict_sub[event_verb_id[event2_id]] == "VBN" and event_verb_id[event2_id] >= 2 and event_verb_id[event2_id] - 2 in pos_position_index_dict_sub and pos_position_index_dict_sub[event_verb_id[event2_id] - 2] == "VBD" and event_verb_id[event2_id] - 1 in pos_position_index_dict_sub and pos_position_index_dict_sub[event_verb_id[event2_id] - 1] == "RB"):
                        if "would" in sent_word[:event_verb_id[event1_id]] and event_verb_id[event1_id] >=1 and sent_word[event_verb_id[event1_id] -1].lower() == "to" and pos_position_index_dict_sub[event_verb_id[event1_id]] == "VB":
                            flag = 7
                        if "would" in sent_word[:event_verb_id[event1_id]] and event_verb_id[event1_id] >=2 and sent_word[event_verb_id[event1_id] -2].lower() == "to" and pos_position_index_dict_sub[event_verb_id[event1_id]] == "VB" and sent_word[event_verb_id[event1_id] -1] == "RB":
                            flag = 7
                        ###if satisfy rule 8
                        if event_verb_id[event1_id] >=2 and sent_word[event_verb_id[event1_id] -2].lower() == "would" and sent_word[event_verb_id[event1_id] -1].lower() == "have" and pos_position_index_dict_sub[event_verb_id[event1_id]] == "VBN":
                            flag = 8
                        ###if satisfy rule 9
                        if event_verb_id[event1_id] >=2 and sent_word[event_verb_id[event1_id] -2].lower() in ["could", "might"] and sent_word[event_verb_id[event1_id] -1].lower() == "have" and pos_position_index_dict_sub[event_verb_id[event1_id]] == "VBN":
                            flag = 9
                        ###if satisfy rule 10
                        if event_verb_id[event1_id] >=3 and event_verb_id[event1_id]-3 in pos_position_index_dict_sub and event_verb_id[event1_id]-2 in pos_position_index_dict_sub and event_verb_id[event1_id]-1 in pos_position_index_dict_sub:
                            if pos_position_index_dict_sub[event_verb_id[event1_id]-3] == "MD" and pos_position_index_dict_sub[event_verb_id[event1_id]-2] == "VB" and pos_position_index_dict_sub[event_verb_id[event1_id]-1] == "VBN" and pos_position_index_dict_sub[event_verb_id[event1_id]] == "VBG":
                                flag = 10
                        if event_verb_id[event1_id] >=4 and event_verb_id[event1_id]-4 in pos_position_index_dict_sub and event_verb_id[event1_id]-3 in pos_position_index_dict_sub and event_verb_id[event1_id]-2 in pos_position_index_dict_sub and event_verb_id[event1_id]-1 in pos_position_index_dict_sub:
                            if pos_position_index_dict_sub[event_verb_id[event1_id]-4] == "MD" and pos_position_index_dict_sub[event_verb_id[event1_id]-3] == "RB" and pos_position_index_dict_sub[event_verb_id[event1_id]-2] == "VB" and pos_position_index_dict_sub[event_verb_id[event1_id]-1] == "VBN" and pos_position_index_dict_sub[event_verb_id[event1_id]] == "VBG":
                                flag = 10
                    ###if satisfy rule 11
                    if (pos_position_index_dict_sub[event_verb_id[event2_id]] == "VBG" and event_verb_id[event2_id] >=2 and event_verb_id[event2_id] - 2 in pos_position_index_dict_sub and pos_position_index_dict_sub[event_verb_id[event2_id] - 2] == "VBD" and event_verb_id[event2_id] - 1 in pos_position_index_dict_sub and pos_position_index_dict_sub[event_verb_id[event2_id] - 1] == "VBN") or \
                            (pos_position_index_dict_sub[event_verb_id[event2_id]] == "VBG" and event_verb_id[event2_id] >=3 and event_verb_id[event2_id] - 3 in pos_position_index_dict_sub and pos_position_index_dict_sub[event_verb_id[event2_id] - 3] == "VBD" and event_verb_id[event2_id] - 2 in pos_position_index_dict_sub and pos_position_index_dict_sub[event_verb_id[event2_id] - 2] == "RB" and event_verb_id[event2_id] - 1 in pos_position_index_dict_sub and pos_position_index_dict_sub[event_verb_id[event2_id] - 1] == "VBN"):
                        if event_verb_id[event1_id] >=2 and pos_position_index_dict_sub[event_verb_id[event1_id] -2] == "MD" and pos_position_index_dict_sub[event_verb_id[event1_id] -1] == "VB" and pos_position_index_dict_sub[event_verb_id[event1_id]] == "VBN":
                            flag = 11
                        if event_verb_id[event1_id] >=3 and pos_position_index_dict_sub[event_verb_id[event1_id] -3] == "MD" and pos_position_index_dict_sub[event_verb_id[event1_id] -2] == "RB" and pos_position_index_dict_sub[event_verb_id[event1_id] -1] == "VB" and pos_position_index_dict_sub[event_verb_id[event1_id]] == "VBN":
                            flag = 11
                    ###if satisfy rule 12
                    if (pos_position_index_dict_sub[event_verb_id[event2_id]] == "VBN" and event_verb_id[event2_id] >= 1 and event_verb_id[event2_id] - 1 in pos_position_index_dict_sub and pos_position_index_dict_sub[event_verb_id[event2_id] - 1] == "VBD") or \
                            (pos_position_index_dict_sub[event_verb_id[event2_id]] == "VBN" and event_verb_id[event2_id] >= 2 and event_verb_id[event2_id] - 2 in pos_position_index_dict_sub and pos_position_index_dict_sub[event_verb_id[event2_id] - 2] == "VBD" and event_verb_id[event2_id] - 1 in pos_position_index_dict_sub and pos_position_index_dict_sub[event_verb_id[event2_id] - 1] == "RB"):
                        if event_verb_id[event1_id] >=2 and sent_word[event_verb_id[event1_id] -2].lower()  == "would" and sent_word[event_verb_id[event1_id] -1].lower() == "be" and sent_word[event_verb_id[event1_id]][-3:] == "ing" and pos_position_index_dict_sub[event_verb_id[event1_id]] == "VBG":
                            flag = 12

    return flag != 0






