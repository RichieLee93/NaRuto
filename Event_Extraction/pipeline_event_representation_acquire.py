#######generate all event representations before extracting temporal relations
import argparse
import pandas as pd
from resolve_coref import coref
from event_extraction import event_extraction
from POS_dependency_parsing import POS_DEP_tagging
from event_element_presentation import event_element
from phrasal_verb_detection import phrasal_verb
from sub_event_detection import subevent_detect

def main():

    
    data = pd.read_csv("./coref.csv")
    data = subevent_detect(phrasal_verb(event_element(POS_DEP_tagging(event_extraction(coref(data))))))
    data = data.drop_duplicates()
    data.to_csv("./all_event.csv", index=False)



if __name__ == '__main__':
    main()
