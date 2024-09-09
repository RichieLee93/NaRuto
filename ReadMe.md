# NaRuto
NaRuto: An advanced system that can automatically acquire planning models from narrative texts

NaRuto is an innovative system that initially extracts structured events from text and subsequently generates planning-language-style action models based on predictions of commonsense event relations, as well as textual contradictions and similarities in an unsupervised manner. It comprises two stages.

# Requirements

Python 3.7+

[Stanford CoreNLP tookit](https://stanfordnlp.github.io/CoreNLP/download.html)

Packages:

transformers==3.0.2\
torch==1.5.1\
pytorch-lightning==0.8.1\
pandas\
nltk\
spacy\
allennlp\
tensorboard\
psutil\
sacrebleu\
rouge-score\
tensorflow_datasets\
faiss\
streamlit\
elasticsearch\
nlp\
torchtext



# Stage I: Structured Event Representation 

A pipeline for processing input text and returning events:

- **event_extraction.py**: Get event occurrences from text. Data type should be specified: movieplot or goodnews.
- **resolve_coref.py**: coreference resolution. Map the entity mentions that refer to previous occurred entity names.
- **POS_DEP_tagging.py**: POS tagging and dependency parsing via StandofordcoreNLP toolkit.
- **phrasal_verb.py**: Detect phrasal verbs if exist.
- **subevent_detect.py**: Detect argument events and conditional events from the event occurrences.
- **pipeline_event_representation_acquire.py**: Run this pipeline python file equals to run all the above files one by one as a pipeline.


# Stage II: Predicates Inference

Finetune COMET-BM:

Go to the COMET-BM folder, and run `bash run.sh`. For running, please specify the directory (`--data_dir`) where you download & save the ATOMIC-2020 data.

The results will be saved to `COMET-BM/results/`.

Run `python COMET-BM/inference.py` for predicting using the local negation generation strategy.

Additionally, run `python COMET-BM/global_negation.py` for generating global negated preconditions and effects.




 # Cite us

If you feel the code helpful, please cite:

```  
@inproceedings{li2024naruto,
  title={NaRuto: Automatically Acquiring Planning Models from Narrative Texts},
  author={Li, Ruiqi and Cui, Leyang and Lin, Songtuan and Haslum, Patrik},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={18},
  pages={20194--20202},
  year={2024}
}

```
