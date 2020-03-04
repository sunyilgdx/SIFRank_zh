#! /usr/bin/env python
# -*- coding: utf-8 -*-
# __author__ = "Sponge"
# Date: 2019/6/19

from model import extractor
from nltk.corpus import stopwords
import thulac
from stanfordcorenlp import StanfordCoreNLP
stopword_dict = set(stopwords.words('english'))
# from stanfordcorenlp import StanfordCoreNLP
# en_model = StanfordCoreNLP(r'E:\Python_Files\stanford-corenlp-full-2018-02-27',quiet=True)
class InputTextObj:
    """Represent the input text in which we want to extract keyphrases"""

    def __init__(self, zh_model, text=""):
        """
        :param is_sectioned: If we want to section the text.
        :param zh_model: the pipeline of Chinese tokenization and POS-tagger
        :param considered_tags: The POSs we want to keep
        """
        self.considered_tags = {'n', 'np', 'ns', 'ni', 'nz','a','d','i','j','x','g'}

        self.tokens = []
        self.tokens_tagged = []
        # self.tokens = zh_model.cut(text)
        word_pos = zh_model.cut(text)
        self.tokens = [word_pos[0] for word_pos in word_pos]
        self.tokens_tagged = [(word_pos[0],word_pos[1]) for word_pos in word_pos]
        assert len(self.tokens) == len(self.tokens_tagged)
        for i, token in enumerate(self.tokens):
            if token.lower() in stopword_dict:
                self.tokens_tagged[i] = (token, "u")
            if token == '-':
                self.tokens_tagged[i] = (token, "-")
        self.keyphrase_candidate = extractor.extract_candidates(self.tokens_tagged, zh_model)

# if __name__ == '__main__':
#     text = "以BERT为代表的自然语言预训练模型（Pre-trained Language Model）的出现使自然语言的各个任务领域的效果都得到大幅地提升。"
#     zh_model = thulac.thulac(model_path=r'../auxiliary_data/thulac.models/')
#     out1 = zh_model.cut(text,text=False)
#