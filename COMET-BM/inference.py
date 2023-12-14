import json
import torch
import argparse
from tqdm import tqdm
from pathlib import Path
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from utils import calculate_rouge, use_task_specific_params, calculate_bleu_score, trim_batch


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


class Comet:
    def __init__(self, model_path):
        # self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = "cpu"
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        task = "summarization"
        use_task_specific_params(self.model, task)

        self.batch_size = 1
        self.decoder_start_token_id = None

    def generate(
            self,
            queries,
            decode_method="beam",
            num_generate=5,
            ):

        with torch.no_grad():
            examples = queries

            decs = []
            scores = []
            for batch in list(chunks(examples, self.batch_size)):

                batch = self.tokenizer(batch, return_tensors="pt", truncation=True, padding="max_length").to(self.device)
                input_ids, attention_mask = trim_batch(**batch, pad_token_id=self.tokenizer.pad_token_id)

                summaries = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_start_token_id=self.decoder_start_token_id,
                    num_beams=num_generate,
                    no_repeat_ngram_size=2,
                    num_return_sequences=num_generate,
                    output_scores=True,
                    return_dict_in_generate=True,
                    temperature=0.7,
                    top_p=0.95,
                    top_k=50
                    )

                beam_scores = [max(0, round(1+s,2)) for s in summaries.sequences_scores.tolist()]
                dec = self.tokenizer.batch_decode(summaries.sequences, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                decs.append(dec)
                scores.append(beam_scores)
            return decs, scores

all_relations = [
    "oEffect",
    "oReact",
    "xEffect",
    "xNeed",
    "xReact"
    ]


def get_related_argname_from_argument(event):
    selected_arg_names = ["ARG0", "ARG1", "ARG2", "ARG3", "ARG4", "ARG5"]
    verb_index = [i for i in range(len(event["tags"])) if event["tags"][i] == "B-V"][0]
    before_verb_phrase = [event["tags"][i].split("-")[-1] for i in range(verb_index) if event["tags"][i].split("-")[-1] in selected_arg_names]
    after_verb_phrase = [event["tags"][i].split("-")[-1] for i in range(verb_index, len(event["tags"])) if event["tags"][i].split("-")[-1] in selected_arg_names]
    before_verb_phrase = list(set(before_verb_phrase))
    after_verb_phrase = list(set(after_verb_phrase))
    before_verb_phrase.sort()
    after_verb_phrase.sort()
    return {"X": before_verb_phrase, "Y": after_verb_phrase}

def get_related_entity_from_argument(event, entity_list, event_words):
    selected_arg_names = ["ARG0", "ARG1", "ARG2", "ARG3", "ARG4", "ARG5"]
    verb_index = [i for i in range(len(event["tags"])) if event["tags"][i] == "B-V"][0]
    before_verb_phrase = " ".join([event_words[i] for i in range(verb_index) if event["tags"][i].split("-")[-1] in selected_arg_names])
    after_verb_phrase = " ".join([event_words[i] for i in range(verb_index, len(event["tags"])) if event["tags"][i].split("-")[-1] in selected_arg_names])
    X_candidate_entity = []
    Y_candidate_entity = []
    for entity in entity_list:
        if entity in before_verb_phrase:
            X_candidate_entity.append([entity, entity_list[entity], before_verb_phrase.index(entity)])
        if entity in after_verb_phrase:
            Y_candidate_entity.append([entity, entity_list[entity], after_verb_phrase.index(entity)])

    #sort the possible X and Y based on their index and discard their index
    if X_candidate_entity:
        X_candidate_entity.sort(key=lambda x: x[-1])
        X_candidate_entity = [entity[:-1] for entity in X_candidate_entity]
    if Y_candidate_entity:
        Y_candidate_entity.sort(key=lambda x: x[-1])
        Y_candidate_entity = [entity[:-1] for entity in Y_candidate_entity]

    return {"X": X_candidate_entity, "Y": Y_candidate_entity}

def get_actual_event_phrase(event_token, coref):
    for pair in coref:
        if pair[0] in event_token:
            event_token = event_token.replace(pair[0], pair[1])
    return event_token

def get_all_info_from_one_doc(df):
    all_events = {}
    all_entity_type = {}
    all_event_words = {}
    all_coref = {}
    main_events = df["event"].tolist()
    # subevents = df["subevents"].tolist()
    entity_types = df["entity_type"].tolist()
    event_words = df["event_words"].tolist()
    corefs = df["coref"].tolist()
    for i in range(len(main_events)):
        if type(main_events[i]) is str:
            main_events[i] = eval(main_events[i])

        if type(entity_types[i]) is str:
            entity_types[i] = eval(entity_types[i])
        if type(event_words[i]) is str:
            event_words[i] = eval(event_words[i])
        if type(corefs[i]) is str:
            corefs[i] = eval(corefs[i])
        all_events[main_events[i]["event_id"]] = main_events[i]
        all_entity_type[main_events[i]["event_id"]] = entity_types[i]
        all_event_words[main_events[i]["event_id"]] = event_words[i]
        all_coref[main_events[i]["event_id"]] = corefs[i]

    return all_events, all_entity_type, all_event_words, all_coref

def get_precond_effect_for_event(all_events, all_entity_type, all_event_words, all_coref):
    list_eventid = list(all_events.keys())
    all_XY = {}
    all_event_phrase = {}
    all_para = {}
    queries = []
    for e in list_eventid:
        event_tokens = [all_event_words[e][i] for i in range(len(all_events[e]["tags"])) if all_events[e]["tags"][i] != 'O']
        event_phrase = get_actual_event_phrase(" ".join(event_tokens), all_coref[e])
        event_XY = get_related_entity_from_argument(all_events[e], all_entity_type[e], all_event_words[e])
        all_XY[e] = event_XY
        all_para[e] = get_related_argname_from_argument(all_events[e])
        all_event_phrase[e] = event_phrase


    return all_event_phrase, all_XY, all_para




import pandas as pd
from sentence_transformers import SentenceTransformer, util, CrossEncoder
sentence_simi_model = SentenceTransformer('all-mpnet-base-v2')
sentence_contradict_model = CrossEncoder('cross-encoder/nli-deberta-v3-base')
def sent_similarity(sentence1, sentence2, sim_thres=0.5):
    embeddings1 = sentence_simi_model.encode(sentence1, convert_to_tensor=True)
    embeddings2 = sentence_simi_model.encode(sentence2, convert_to_tensor=True)
    cosine_score = util.cos_sim(embeddings1, embeddings2)
    if cosine_score > sim_thres:
        return True
    else:
        return False
def sent_contradict(sentence1, sentence2):

    scores = sentence_contradict_model.predict([(sentence1, sentence2)])

    if scores.argmax(axis=1)[0] == 0:
    # if labels[0] == 'contradiction':
        return True
    else:
        return False




from nltk.stem import WordNetLemmatizer
wn_lemma = WordNetLemmatizer()

def clean_results(results_list):
    remove_word_list = {"PersonX": "X", "PersonY": "O"}
    updated_results = []
    for result in results_list:
        if result[0] == " ":
            result = result[1:]
        if result[-1] in [" ", "."]:
            result = result[:-1]
        if result[:3] == "to ":
            result = result[3:]
        if result[:3] == "be ":
            result = result[3:]
        for word in remove_word_list:
            result = result.replace(word, remove_word_list[word]).lower()
        result = " ".join([wn_lemma.lemmatize(w) for w in result.split(" ")])
        updated_results.append(result)

    return updated_results



def remove_results(results_list, results_score, relation):
    #### clean results first
    results_list = clean_results(results_list)
    ####get none and small score location and select the smaller index to cut off
    ###get none

    if "none" in results_list:
        none_index = results_list.index("none")
    else:
        none_index = 10
    ###get lower score
    low_score_index = 10
    threshold = all_parts_threshold[relation]
    for i in range(len(results_score)):
        if threshold > results_score[i]:
            low_score_index = i
            break
    cut_off_index = min([none_index, low_score_index])
    return results_list[:cut_off_index], results_score[:cut_off_index]


def check_similarity_contradict(results_list, results_score, relation):

    results_list, results_score = remove_results(results_list, results_score, relation)
    if len(results_list) > 1:
        names = results_list
        k = len(results_list)
        updated_k = [True]*k
        for i in range(k-1, 0, -1):
            flag_remove = False
            for j in range(i-1, -1, -1):
                if sent_similarity(names[i], names[j]):
                    flag_remove = True
                    break
            if flag_remove:
                updated_k[i] = False
        updated_result = [names[i] for i in range(len(names[:k])) if updated_k[i]]
        results_list, results_score = updated_result, results_score[:len(updated_result)]
    ####contradict
    if len(results_list) > 1:
        names = results_list
        k = len(results_list)
        updated_k = [True]*k
        for i in range(k-1, 0, -1):
            flag_remove = False
            for j in range(i-1, -1, -1):
                if sent_contradict(names[i], names[j]):
                    flag_remove = True
                    break
            if flag_remove:
                updated_k[i] = False
        updated_result = [names[i] for i in range(len(names[:k])) if updated_k[i]]
        return updated_result, results_score[:len(updated_result)]
    else:
        return results_list, results_score

def get_all_event_NL(df):
    if type(df["event_words"]) is str:
        df["event_words"] = eval(df["event_words"])
    if type(df["event"]) is str:
        df["event"] = eval(df["event"])
    NL_event = []
    for tag in range(len(df["event"]["tags"])):
        if df["event"]["tags"][tag] != "O":
            NL_event.append(df["event_words"][tag])
    return " ".join(NL_event)




if __name__ == "__main__":
    comet = Comet("/")
    comet.model.zero_grad()
    candidate_relations = all_relations

    all_parts_threshold = {'xNeed': 0.7, 'xEffect': 0.5, 'xReact': 0.2, 'oEffect': 0.5, 'oReact': 0.2}

    import time
    start_time = time.time()
    all_results = []

    all_e1 = []
    for h in all_e1:
        queries = []
        head = h
        for rel in candidate_relations:
            query = "{} {}".format(head, rel)
            queries.append(query)
        results, scores = comet.generate(queries, decode_method="beam", num_generate=6)
        print(h)
        result = {}
        for i in range(len(candidate_relations)):
            print(candidate_relations[i], " ", check_similarity_contradict(results[i], scores[i], candidate_relations[i]))
            result[candidate_relations[i]] = check_similarity_contradict(results[i], scores[i], candidate_relations[i])
        print("--- %s seconds ---" % (time.time() - start_time))
        all_results.append(result)
        print(result)

