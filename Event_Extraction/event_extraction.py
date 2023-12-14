import pandas as pd
from allennlp.predictors.predictor import Predictor
import spacy
import os, re, copy
# os.environ['OMP_NUM_THREADS'] = "8"
nlp = spacy.load("en_core_web_sm")
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def paragraph_segmentation(para, is_story_cloze):
    sentences = []
    index_length = []
    token_index_length = []

    try:

        id_l = 0
        id_token_l = 0
        if is_story_cloze:
            sent_names = ['InputSentence1', 'InputSentence2', 'InputSentence3', 'InputSentence4', 'InputSentence5','InputSentence6']
            all_sents = [para[sent] for sent in sent_names]
        else:
            all_sents = nlp(para)
            all_sents = list(all_sents.sents)

        for sent in all_sents:
            sent = str(sent)
            sentences.append(sent)
            index_length.append([id_l, id_l + len(sent)])
            id_l += len(sent)
            token_sent = [i.text for i in nlp(sent)]
            token_index_length.append([id_token_l, id_token_l + len(token_sent)])
            id_token_l += len(token_sent)

    except IndexError:
        print("para: ", para)
    return sentences, index_length, token_index_length


def predict_srl(df, predictor, is_story_cloze):
    results = []
    if type(df["plot_summary_coref"]) is str:
        df["plot_summary_coref"] = eval(df["plot_summary_coref"])
    if is_story_cloze:
        sentences, index_length, token_index_length = paragraph_segmentation(df, is_story_cloze)
    else:
        sentences, index_length, token_index_length = paragraph_segmentation(df["plot_summary"], is_story_cloze)
    if sentences:
        for sent in range(len(sentences)):
            try:
                result = predictor.predict(sentence=sentences[sent])
                result["sent"] = sentences[sent]
                result["coref"] = get_included_coref(token_index_length[sent], df["plot_summary_coref"])
                results.append(result)
            except RuntimeError:
                print("catch run time error")
                pass

    return str(results)


def get_included_coref(token_index_set, plot_summaries_coref):
    """get all the related coref exist in the sent"""
    begin = token_index_set[0]
    end = token_index_set[1]
    all_matched_pairs = []
    for sets in plot_summaries_coref[-1]:
        for item in range(1, len(sets)):
            if sets[item][0] >= begin and end >= sets[item][1]:
                if sets[item][0] == sets[item][1]:
                    all_matched_pairs.append(
                        [plot_summaries_coref[0][sets[item][0]], " ".join([plot_summaries_coref[0][word] for word in range(sets[0][0], sets[0][1]+1)]),[[sets[item][0]], [sets[0][0], sets[0][1]+1]]])
                else:
                    all_matched_pairs.append(
                    [" ".join([plot_summaries_coref[0][word] for word in range(sets[item][0], sets[item][1]+1)]),  " ".join([plot_summaries_coref[0][word] for word in range(sets[0][0], sets[0][1]+1)]), [[sets[item][0], sets[item][1]+1], [sets[0][0], sets[0][1]+1]]])

    return all_matched_pairs




def get_lemma_of_verb(verb):
    doc = nlp(verb)
    if len(doc) <= 1:
        return doc[0].lemma_
    else:
        lemmas = []
        for token in doc:
            lemmas.append(token.lemma_)
        return str(lemmas)

def remove_only_v(df):
    """reomve event with only one verb and statement (statement is based on statement_verbs)"""
    reduced_df = []
    state_df = []
    indexed_df = []
    statement_verbs = ["has", "have", "had", "is", "was", "are", "were", "am", "must", "should", "want", "shall", "could",
                "will", "would", "may", "might", "can", "ought", "need", "be", "been", str(["be", "n't"]), str(["ai", "n't"])]
    if type(df) is str:
        df = eval(df)
    if df:
        event_id = 0
        for i in df:
            copy_i_state = copy.copy(i)
            copy_i_act = copy.copy(i)
            temp_act = []
            temp_state = []
            temp_all = []
            for event in i["verbs"]:
                if len([pos for pos, char in enumerate(event["description"]) if char == '[']) > 1 and len(list(set(event["tags"]))) > 2 and "B-V" in event["tags"]:
                    verb_lemma = get_lemma_of_verb(event["verb"])
                    event["event_id"] = event_id
                    event["verb_lemma"] = verb_lemma
                    event_id += 1
                    temp_all.append(event)
                    if verb_lemma in statement_verbs:
                        temp_state.append(event)

                    else:
                        temp_act.append(event)
            copy_i_act["verbs"] = temp_act
            copy_i_state["verbs"] = temp_state
            i["verbs"] = temp_all
            reduced_df.append(copy_i_act)
            state_df.append(copy_i_state)
            indexed_df.append(i)
    return pd.Series([reduced_df, state_df, indexed_df])


def get_distinct_events_and_ACE(whole_df):
    reduced_df = []
    ACE_df = []
    words_df = []
    sent_ids = []
    doc_ids = []
    coref_df = []
    all_events = whole_df["events_updated_all"].tolist()
    all_doc_ids = whole_df["d_id"].tolist()

    for df in range(len(all_events)):
        if type(all_events[df]) is str:
            all_events[df] = eval(all_events[df])
        if all_events[df]:
            all_temp_reduced_df = []
            all_temp_ACE = []
            sent_id = 0
            for events in all_events[df]:
                if len(events["verbs"]) > 1:
                    for i in range(len(events["verbs"])):
                        if events["verbs"][i] not in all_temp_reduced_df and events["verbs"][i] not in all_temp_ACE:
                            reduced_df.append(events["verbs"][i])
                            all_temp_reduced_df.append(events["verbs"][i])
                            words_df.append(events["words"])
                            coref_df.append(events["coref"])
                            sent_ids.append(sent_id)
                            doc_ids.append(all_doc_ids[df])
                            temp_ACE = []
                            for j in range(len(events["verbs"])):
                                if i != j:
                                    word_list_i = [events["words"][w] for w in range(len(events["words"])) if events["verbs"][i]["tags"][w] != "O"]
                                    word_list_j = [events["words"][w] for w in range(len(events["words"])) if events["verbs"][j]["tags"][w] != "O"]
                                    if all(elem in word_list_i for elem in word_list_j): ##### check if all elements in j also in i
                                        if events["verbs"][j] not in temp_ACE:
                                            temp_ACE.append(events["verbs"][j])
                                            all_temp_ACE.append(events["verbs"][j])

                            ACE_df.append(temp_ACE)


                elif len(events["verbs"]) == 1:
                    reduced_df.append(events["verbs"][0])
                    all_temp_reduced_df.append(events["verbs"][0])
                    ACE_df.append([])
                    doc_ids.append(all_doc_ids[df])
                    words_df.append(events["words"])
                    coref_df.append(events["coref"])
                    sent_ids.append(sent_id)
                #     event_sent.append(temp)
                sent_id += 1


    return [doc_ids, reduced_df, ACE_df, words_df, sent_ids, coref_df]


def event_extraction(data, is_story_cloze):
    predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz")
    data["events"] = data.apply(lambda x: predict_srl(x, predictor, is_story_cloze), axis=1)
    data[["events_updated_remove_only", "events_updated_state", "events_updated_all"]] = data["events"].apply(lambda x: remove_only_v(x))
    distinct_events = get_distinct_events_and_ACE(data)
    df_dist_events = pd.DataFrame(distinct_events).transpose()
    df_dist_events.columns = ['d_id', 'event', "ACE", 'event_words', 's_id', 'coref']
    return df_dist_events
