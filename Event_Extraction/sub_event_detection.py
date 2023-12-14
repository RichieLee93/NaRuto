import string, random
import pandas as pd
import networkx as nx
from collections import Counter, defaultdict
from pycorenlp import StanfordCoreNLP

nlp = StanfordCoreNLP('http://localhost:9000')
from conditional_sentence_detection import check_if_sent_match_structure


def depenency_based_rule(event1_id, event2_id, event_verb_id, sent_dep):
    ####get all verb index for all events  - all_verb_id_index as {event_id: verb_index}
    matched_events = False
    core_arg_pair_verbs = [event_verb_id[event1_id]+1, event_verb_id[event2_id]+1]

    for dep in sent_dep:
        if dep[0][-4:] == "comp" or dep[0] == "csubj":
            if list(dep[-1]) == core_arg_pair_verbs:
                matched_events = True
                # matched_events.append(["comp/csubj", [i["description"] for i in [df["event"]] + df["ACE"] if i["event_id"] == core_arg_pair[0]][0], [i["description"] for i in [df["event"]] + df["ACE"] if i["event_id"] == core_arg_pair[1]][0]])

        if dep[0] in ["cop", "aux:pass"] and not matched_events:
            if list([dep[-1][1], dep[-1][0]]) == core_arg_pair_verbs:
                matched_events = True

    return matched_events


def get_root_tokens_in_ARGM(dep_basic, ARGM_indices):
    ###get the root words of the ARGM
    sub_event_indices = ARGM_indices
    ####get all dependency pairs that consist of two tokens that are both from the sub-event
    #### Meanwhile get all dependency pairs that consist only one token from the sub-event
    sub_tree_tokens = []
    for dep in dep_basic:
        if dep[-1][0] in sub_event_indices and dep[-1][1] not in sub_event_indices and dep[0] != "punct" and dep[-1][
            0] not in sub_tree_tokens:
            sub_tree_tokens.append(dep[-1][0]-1)
        if dep[-1][1] in sub_event_indices and dep[-1][0] not in sub_event_indices and dep[0] != "punct" and dep[-1][
            1] not in sub_tree_tokens:
            sub_tree_tokens.append(dep[-1][1]-1)
    return sub_tree_tokens


def keep_PRP_subevent(event1_id, event2_id, event_verb_id, event_span_id, ARG_tags, sent_dep):
    exist_ARGM = False
    flag = False
    for arg in event_span_id[event1_id]:
        if ARG_tags[0] in arg or ARG_tags[1] in arg:
            exist_ARGM = True
            ARGM_ind =  [ind+1 for ind in event_span_id[event1_id][arg]]
        if exist_ARGM:
            root_words_of_ARGM = get_root_tokens_in_ARGM(sent_dep, ARGM_ind)
            if event_verb_id[event2_id] in root_words_of_ARGM:
                flag = True
    return flag



def keep_W_word_subevent(event1_id, event2_id, event_verb_id, event_dep, W_word_list):
    all_mark_flags = False
    advcl_set = [dep[-1] for dep in event_dep if dep[0] == "advcl"]
    mark_set = [dep for dep in event_dep if dep[0] in ["mark", "case"]]
    if advcl_set and mark_set:
        flag = 0
        v_1 = event_verb_id[event1_id] + 1
        v_2 = event_verb_id[event2_id] + 1
        #### check if event1 and event2 verb are with advcl dep relation
        if [v_1, v_2] in advcl_set or [v_2, v_1] in advcl_set:
            #### check if v_2 has mark dep relation with any other word
            for mark in mark_set:
                if v_2 in mark[-1]:
                    if v_2 == mark[-1][0]:
                        flag = mark[2]
                    else:
                        flag = mark[1]
        if flag != 0 and flag.lower() in W_word_list:
            all_mark_flags = True

    return all_mark_flags

def find_all_main_event_dict(conditional_event_pair, arg_event_pair):
    main_events = []
    main_argevent_pair = {}
    main_leaves = {}
    main_condevent_pair = {}
    left = []
    right = []
    if conditional_event_pair:
        left = [pair[0] for pair in conditional_event_pair]
        right = [pair[1] for pair in conditional_event_pair]
        all_cond_nodes = list(set(left + right))

    if arg_event_pair:
        left += [pair[0] for pair in arg_event_pair]
        right += [pair[1] for pair in arg_event_pair]
        all_arg_nodes = list(set([pair[0] for pair in arg_event_pair] + [pair[1] for pair in arg_event_pair]))
    left = list(set(left))
    right = list(set(right))
    for candid in left:
        if candid not in right and candid not in main_events:
            main_events.append(candid)
    if main_events:
        if arg_event_pair:
            G_arg = nx.DiGraph()
            G_arg.add_edges_from(arg_event_pair)
            all_leaf = [node for node in all_arg_nodes if node not in main_events]
            for event in main_events:
                arg_events = []
                for leaf in all_leaf:
                    if event in G_arg and leaf in G_arg and nx.has_path(G_arg, event, leaf):
                        arg_events.append(leaf)
                main_leaves[event] = arg_events
                main_argevent_pair[event] = []
                for arg_pair in arg_event_pair:
                    if arg_pair[0] in arg_events+[event] and arg_pair[1] in arg_events+[event]:
                        main_argevent_pair[event].append(arg_pair)

        if conditional_event_pair:
            G_cond = nx.DiGraph()
            G_cond.add_edges_from(conditional_event_pair)
            all_leaf = [node for node in all_cond_nodes if node not in main_events]
            for event in main_events:
                cond_events = []
                for leaf in all_leaf:
                    if event in G_cond and leaf in G_cond and nx.has_path(G_cond, event, leaf):
                        cond_events.append(leaf)
                if event in main_leaves:
                    main_leaves[event] += cond_events
                else: main_leaves[event] = cond_events
                main_leaves[event] = list(set(main_leaves[event]))
                main_condevent_pair[event] = []
                for cond_pair in conditional_event_pair:
                    if cond_pair[0] in cond_events+[event] and cond_pair[1] in cond_events+[event]:
                        main_condevent_pair[event].append(cond_pair)

    return main_argevent_pair, main_condevent_pair, main_leaves



def generate_new_df(d_id, s_id, coref, sent_word, sent_dep, sent_POS, event_dict, event_ele_dict, phrasal_verb_dict, main_argevent_pair, main_condevent_pair, main_leaves):
    write_list = []
    for main_event in main_argevent_pair:
        main_event_rep = event_dict[main_event]
        main_event_ele = event_ele_dict[main_event]
        phrasal_verbs = {main_event: phrasal_verb_dict[main_event]}
        if main_event in main_leaves and main_leaves[main_event]:
            sub_event_rep = dict((subevent, event_dict[subevent]) for subevent in main_leaves[main_event])
            sub_event_ele = dict((subevent, event_ele_dict[subevent]) for subevent in main_leaves[main_event])
            for subevent in main_leaves[main_event]:
                phrasal_verbs[subevent] = phrasal_verb_dict[subevent]
        else: sub_event_rep = sub_event_ele = {}
        main_sub_list = [d_id, s_id, coref, sent_word, sent_dep, sent_POS, main_event_rep, main_event_ele,
                         sub_event_rep, sub_event_ele, phrasal_verbs, main_argevent_pair[main_event] if main_event in main_argevent_pair else [], main_condevent_pair[main_event] if main_event in main_condevent_pair else [], main_leaves[main_event]]
        write_list.append(main_sub_list)
    column_names = ['d_id', 's_id', 'coref', 'event_words', 'event_words_dep_basic', 'event_words_POS',
                    'event', 'event_elements','subevents', 'subevent_elements', 'event_phrasal_verbs', 'argument_event_pairs',
                    'conditional_event_pairs', "all_subevents"]
    write_df = pd.DataFrame(write_list, columns = column_names)
    return write_df






def get_overall_core_arg_events(df, W_word_list, ARG_tags):
    d_id, s_id, coref = get_uniformed_info(df)
    event_dict, event_ele_dict, phrasal_verb_dict, sent_word, sent_dep, sent_POS, event_verb_id, event_span_id, all_possible_pairs, all_conjunction_pairs = decompose_df_same_sent_into_dict(df)
    arg_event_pair = []
    conditional_event_pair = []
    for (event1_id, event2_id) in all_possible_pairs:
        if depenency_based_rule(event1_id, event2_id, event_verb_id, sent_dep) or \
                keep_W_word_subevent(event1_id, event2_id, event_verb_id, sent_dep, W_word_list) or \
                keep_PRP_subevent(event1_id, event2_id, event_verb_id, event_span_id, ARG_tags, sent_dep):
            arg_event_pair.append((event1_id, event2_id))
        if check_if_sent_match_structure(event1_id, event2_id, event_verb_id, event_dict, sent_word, sent_POS):
            conditional_event_pair.append((event1_id, event2_id))
    for conj_pair in all_conjunction_pairs:
        for arg_pair in arg_event_pair:
            if arg_pair[-1] == conj_pair[0] and (arg_pair[0], conj_pair[1]) not in arg_event_pair:
                arg_event_pair.append((arg_pair[0], conj_pair[1]))
            if arg_pair[-1] == conj_pair[1] and (arg_pair[0], conj_pair[0]) not in arg_event_pair:
                arg_event_pair.append((arg_pair[0], conj_pair[0]))
        for cond_pair in conditional_event_pair:
            if cond_pair[-1] == conj_pair[0] and (cond_pair[0], conj_pair[1]) not in conditional_event_pair:
                conditional_event_pair.append((cond_pair[0], conj_pair[1]))
            if cond_pair[-1] == conj_pair[1] and (cond_pair[0], conj_pair[0]) not in conditional_event_pair:
                conditional_event_pair.append((cond_pair[0], conj_pair[0]))
    main_argevent_pair, main_condevent_pair, main_leaves = find_all_main_event_dict(conditional_event_pair, arg_event_pair)

    #####checking if any missing independent event with no argevent and conditionals
    all_occur = []
    for e in main_leaves:
        all_occur.append(e)
        if main_leaves[e]:
            for sub in main_leaves[e]:
                all_occur.append(sub)

    for event in event_dict:
        if event not in all_occur:
            main_argevent_pair[event] = main_leaves[event] = main_condevent_pair[event] = []
    write_df = generate_new_df(d_id, s_id, coref, sent_word, sent_dep, sent_POS, event_dict, event_ele_dict, phrasal_verb_dict, main_argevent_pair, main_condevent_pair, main_leaves)
    return write_df

    ####extend based on conj pairs




def check_if_two_events_verb_overlap(event1_id, event2_id, event_verb_id, event_span):
    event2_verb_index = event_verb_id[event2_id]
    event1_event_span = [id for spans in event_span[event1_id] for id in event_span[event1_id][spans]]
    if event2_verb_index in event1_event_span:
        return True
    else:
        return False

def decompose_df_same_sent_into_dict(df):
    event_dict = {}
    event_ele_dict = {}
    event_verb_id = {}
    event_span_id = {}
    phrasal_verb_dict = dict(pair for d in df["event_phrasal_verbs"].tolist() for pair in d.items())
    sent_word = [eval(i) if type(i) is str else i for i in df["event_words"].tolist()][0]
    sent_POS = [eval(i) if type(i) is str else i for i in df["event_words_POS"].tolist()][0]
    sent_dep = [eval(i) if type(i) is str else i for i in df["event_words_dep_basic"].tolist()][0]
    conj_set = [(dep[-1][0], dep[-1][1]) for dep in sent_dep if dep[0] == "conj"]
    # print(999, conj_set)
    all_events = [eval(event) if type(event) is str else event for event in df["event"].tolist()]
    for event_ele in [eval(ele) if type(ele) is str else ele for ele in df["event_elements"].tolist()]:
        event_ele_dict[event_ele["event_id"]] = event_ele
    for ACE_eles in df["ACE_elements"].tolist():
        if type(ACE_eles) is str:
            ACE_eles = eval(ACE_eles)
        if ACE_eles:
            for ACE_ele in ACE_eles:
                event_ele_dict[ACE_ele["event_id"]] = ACE_ele
    for ACE in df["ACE"].tolist():
        if type(ACE) is str:
            ACE = eval(ACE)
        if ACE:
            for contained_event in ACE:
                if contained_event not in all_events:
                    all_events.append(contained_event)

    for event in all_events:
        event_dict[event["event_id"]] = event
        event_verb_id[event["event_id"]] = [tag for tag in range(len(event["tags"])) if "B-V" in event["tags"][tag]][0]
        tagname_index = defaultdict(list)
        for tag in range(len(event["tags"])):
            tagname_index[event["tags"][tag]].append(tag)
        tagname_index.pop("O", None)
        event_span_id[event["event_id"]] = tagname_index
    all_possible_pairs = []
    all_conjunction_pairs = []
    for event1_id in event_dict:
        for event2_id in event_dict:
            if event1_id != event2_id:
                if check_if_two_events_verb_overlap(event1_id, event2_id, event_verb_id, event_span_id) and (
                event1_id, event2_id) not in all_possible_pairs and (event2_id, event1_id) not in all_possible_pairs:
                    all_possible_pairs.append((event1_id, event2_id))
                if conj_set and find_conjunction_event_verb(event1_id, event2_id, event_verb_id, conj_set):
                    all_conjunction_pairs.append((event1_id, event2_id))
    return event_dict, event_ele_dict, phrasal_verb_dict, sent_word, sent_dep, sent_POS, event_verb_id, event_span_id, all_possible_pairs, all_conjunction_pairs


def find_conjunction_event_verb(event1_id, event2_id, event_verb_id, conj_set):
    event1_verb = event_verb_id[event1_id]
    event2_verb = event_verb_id[event2_id]
    if (event1_verb + 1, event2_verb + 1) in conj_set or (event2_verb + 1, event1_verb + 1) in conj_set:
        return True
    else:
        return False

def get_uniformed_info(df):
    d_id = df["d_id"].tolist()[0]
    s_id = df["s_id"].tolist()[0]
    coref = df["coref"].tolist()[0]
    return d_id, s_id, coref


def subevent_detect(data):
    selected_id = list(set(data["d_id"].tolist()))
    W_word_list = ["so", "by", "about", "from", "in"]
    arg_tags = ["ARGM-PRP", "ARGM-PNC"]
    data['event_words'] = data.event_words.astype(str)
    all_write_df = []
    for id in selected_id:
        # print("start processing: ", id)
        data_selected = data[data["d_id"] == id]
        all_sentences = data_selected["event_words"].tolist()
        for sent in all_sentences:
            data_sent_selected = data_selected[data_selected["event_words"] == sent]

            write_df = get_overall_core_arg_events(data_sent_selected, W_word_list, arg_tags)
            all_write_df.append(write_df)
    data_updated = pd.concat(all_write_df)
    return data_updated


