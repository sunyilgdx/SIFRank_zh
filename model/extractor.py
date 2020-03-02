#! /usr/bin/env python
# -*- coding: utf-8 -*-
# __author__ = "Sponge"
# Date: 2019/6/19
import nltk
from model import input_representation
import thulac
#GRAMMAR1 is the general way to extract NPs

GRAMMAR1 = """  NP:
        {<NN.*|JJ>*<NN.*>}  # Adjective(s)(optional) + Noun(s)"""

GRAMMAR2 = """  NP:
        {<JJ|VBG>*<NN.*>{0,3}}  # Adjective(s)(optional) + Noun(s)"""

GRAMMAR3 = """  NP:
        {<NN.*|JJ|VBG|VBN>*<NN.*>}  # Adjective(s)(optional) + Noun(s)"""

GRAMMAR_zh = """  NP:
        {<n.*|a|uw|i|j|x>*<n.*|uw|x>|<x|j><-><m|q>} # Adjective(s)(optional) + Noun(s)"""

def extract_candidates(tokens_tagged, no_subset=False):
    """
    Based on part of speech return a list of candidate phrases
    :param text_obj: Input text Representation see @InputTextObj
    :param no_subset: if true won't put a candidate which is the subset of an other candidate
    :return keyphrase_candidate: list of list of candidate phrases: [tuple(string,tuple(start_index,end_index))]
    """
    np_parser = nltk.RegexpParser(GRAMMAR_zh)  # Noun phrase parser
    keyphrase_candidate = []
    np_pos_tag_tokens = np_parser.parse(tokens_tagged)
    count = 0
    for token in np_pos_tag_tokens:
        if (isinstance(token, nltk.tree.Tree) and token._label == "NP"):
            np = ''.join(word for word, tag in token.leaves())
            length = len(token.leaves())
            start_end = (count, count + length)
            count += length
            keyphrase_candidate.append((np, start_end))

        else:
            count += 1

    return keyphrase_candidate

# if __name__ == '__main__':
#     #This is an example.
#     zh_model = thulac.thulac(model_path=r'../auxiliary_data/thulac.models/',user_dict=r'../auxiliary_data/user_dict.txt')
#     sent = "以BERT为代表的自然语言预训练模型（Pre-trained Language Model）的出现使自然语言的各个任务领域的效果都得到大幅地提升。"
#     ito = input_representation.InputTextObj(text=sent,zh_model=zh_model)
#     keyphrase_candidate = ito.keyphrase_candidate
#     for kc in keyphrase_candidate:
#         print(kc)