#! /usr/bin/env python
# -*- coding: utf-8 -*-
# __author__ = "Sponge"
# Date: 2019/6/19
import numpy
import torch
import nltk
from nltk.corpus import stopwords
english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']
chinese_punctuations = '！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏.'
stop_words = set(stopwords.words("english"))
wnl=nltk.WordNetLemmatizer()
# considered_tags = {'NN', 'NNS', 'NNP', 'NNPS', 'JJ','VBG'}
considered_tags = {'n', 'np', 'ns', 'ni', 'nz','a','d','i','j','x','g'}
class SentEmbeddings():

    def __init__(self,
                 word_embeddor,
                 weightfile_pretrain='../auxiliary_data/dict.txt',
                 weightfile_finetune='../auxiliary_data/dict.txt',
                 weightpara_pretrain=2.7e-4,
                 weightpara_finetune=2.7e-4,
                 lamda=1.0,database="",embeddings_type="elmo"):

        self.word2weight_pretrain = get_word_weight(weightfile_pretrain, weightpara_pretrain)
        self.word2weight_finetune = get_word_weight(weightfile_finetune, weightpara_finetune)
        self.word_embeddor = word_embeddor
        self.lamda=lamda
        self.database=database
        self.embeddings_type=embeddings_type

    def get_tokenized_sent_embeddings(self, text_obj, if_DS=False, if_EA=False):
        """
        Based on part of speech return a list of candidate phrases
        :param text_obj: Input text Representation see @InputTextObj
        :param if_DS: if take document segmentation(DS)
        :param if_EA: if take  embeddings alignment(EA)
        """
        # choose the type of word embeddings:elmo or elmo_transformer or glove
        if(self.embeddings_type=="elmo" and if_DS==False):
            elmo_embeddings= self.word_embeddor.get_tokenized_words_embeddings([text_obj.tokens])
        elif(self.embeddings_type=="elmo" and if_DS==True and if_EA==False):
            tokens_segmented = get_sent_segmented(text_obj.tokens)
            elmo_embeddings= self.word_embeddor.get_tokenized_words_embeddings(tokens_segmented)
            elmo_embeddings = splice_embeddings(elmo_embeddings,tokens_segmented)
        elif (self.embeddings_type == "elmo" and if_DS == True and if_EA == True):
            tokens_segmented = get_sent_segmented(text_obj.tokens)
            elmo_embeddings = self.word_embeddor.get_tokenized_words_embeddings(tokens_segmented)
            elmo_embeddings = context_embeddings_alignment(elmo_embeddings, tokens_segmented)
            elmo_embeddings = splice_embeddings(elmo_embeddings, tokens_segmented)

        # elif(self.embeddings_type=="elmo_transformer"):
        #     elmo_embeddings= self.word_embeddor.get_tokenized_words_embeddings([text_obj.tokens])
        # elif (self.embeddings_type == "glove"):
        #     elmo_embeddings = self.word_embeddor.get_tokenized_words_embeddings([text_obj.tokens])

        else:
            elmo_embeddings, elmo_mask = self.word_embeddor.get_tokenized_words_embeddings(text_obj.tokens)

        candidate_embeddings_list=[]

        weight_list = get_weight_list(self.word2weight_pretrain, self.word2weight_finetune, text_obj.tokens, lamda=self.lamda, database=self.database)

        sent_embeddings = get_weighted_average(text_obj.tokens, text_obj.tokens_tagged, weight_list, elmo_embeddings[0], embeddings_type=self.embeddings_type)

        for kc in text_obj.keyphrase_candidate:
            start = kc[1][0]
            end = kc[1][1]
            kc_emb = get_candidate_weighted_average(text_obj.tokens, weight_list, elmo_embeddings[0], start, end,
                                                    embeddings_type=self.embeddings_type)
            candidate_embeddings_list.append(kc_emb)

        return sent_embeddings,candidate_embeddings_list

def context_embeddings_alignment(elmo_embeddings, tokens_segmented):

    """
    Embeddings Alignment
    :param elmo_embeddings: The embeddings from elmo
    :param tokens_segmented: The list of tokens list
     <class 'list'>: [['今', '天', '天气', '真', '好', '啊'],['潮水', '退', '了', '就', '知道', '谁', '没', '穿', '裤子']]
    :return:
    """
    token_emb_map = {}
    n = 0
    for i in range(0, len(tokens_segmented)):

        for j, token in enumerate(tokens_segmented[i]):

            emb = elmo_embeddings[i, 1, j, :]
            if token not in token_emb_map:
                token_emb_map[token] = [emb]
            else:
                token_emb_map[token].append(emb)
            n += 1

    anchor_emb_map = {}
    for token, emb_list in token_emb_map.items():
        average_emb = emb_list[0]
        for j in range(1, len(emb_list)):
            average_emb += emb_list[j]
        average_emb /= float(len(emb_list))
        anchor_emb_map[token] = average_emb

    for i in range(0, elmo_embeddings.shape[0]):
        for j, token in enumerate(tokens_segmented[i]):
            emb = anchor_emb_map[token]
            elmo_embeddings[i, 2, j, :] = emb

    return elmo_embeddings

def mat_division(vector_a, vector_b):
    a = vector_a.detach().numpy()
    b = vector_b.detach().numpy()
    A = numpy.mat(a)
    B = numpy.mat(b)
    # if numpy.linalg.det(B) == 0:
    #     print("This matrix is singular, cannot be inversed!")
    #     return
    return torch.from_numpy(numpy.dot(A.I,B))

def get_sent_segmented(tokens):
    min_seq_len = 16
    sents_sectioned = []
    if (len(tokens) <= min_seq_len):
        sents_sectioned.append(tokens)
    else:
        position = 0
        for i, token in enumerate(tokens):
            if (token == '.' or token =='。'):
                if (i - position >= min_seq_len):
                    sents_sectioned.append(tokens[position:i + 1])
                    position = i + 1
        if (len(tokens[position:]) > 0):
            sents_sectioned.append(tokens[position:])

    return sents_sectioned

def splice_embeddings(elmo_embeddings,tokens_segmented):
    new_elmo_embeddings = elmo_embeddings[0:1, :, 0:len(tokens_segmented[0]), :]
    for i in range(1, len(tokens_segmented)):
        emb = elmo_embeddings[i:i + 1, :, 0:len(tokens_segmented[i]), :]
        new_elmo_embeddings = torch.cat((new_elmo_embeddings, emb), 2)
    return new_elmo_embeddings

def get_effective_words_num(tokened_sents):
    i=0
    for token in tokened_sents:
        if(token not in english_punctuations):
            i+=1
    return i

def get_weighted_average(tokenized_sents, sents_tokened_tagged,weight_list, embeddings_list, embeddings_type="elmo"):
    # weight_list=get_normalized_weight(weight_list)
    assert len(tokenized_sents) == len(weight_list)
    num_words = len(tokenized_sents)
    e_test_list=[]
    if (embeddings_type == "elmo" or embeddings_type == "elmo_sectioned"):
        # assert num_words == embeddings_list.shape[1]
        sum = torch.zeros((3, 1024))
        for i in range(0, 3):
            for j in range(0, num_words):
                if(sents_tokened_tagged[j][1] in considered_tags):
                    e_test=embeddings_list[i][j]
                    e_test_list.append(e_test)
                    sum[i] += e_test * weight_list[j]

            sum[i] = sum[i] / float(num_words)
        return sum
    elif(embeddings_type == "elmo_transformer"):
        sum = torch.zeros((1, 1024))
        for i in range(0, 1):
            for j in range(0, num_words):
                if(sents_tokened_tagged[j][1] in considered_tags):
                    e_test=embeddings_list[i][j]
                    e_test_list.append(e_test)
                    sum[i] += e_test * weight_list[j]
            sum[i] = sum[i] / float(num_words)
        return sum
    elif (embeddings_type == "glove"):
        sum = numpy.zeros((1, embeddings_list.shape[2]))
        for i in range(0, 1):
            for j in range(0, num_words):
                if (sents_tokened_tagged[j][1] in considered_tags):
                    e_test = embeddings_list[i][j]
                    e_test_list.append(e_test)
                    sum[i] += e_test * weight_list[j]
            sum[i] = sum[i] / float(num_words)
        return sum

    return 0

def get_candidate_weighted_average(tokenized_sents, weight_list, embeddings_list, start,end,embeddings_type="elmo"):
    # weight_list=get_normalized_weight(weight_list)
    assert len(tokenized_sents) == len(weight_list)
    # num_words = len(tokenized_sents)
    num_words =end - start
    e_test_list=[]
    if (embeddings_type == "elmo" or embeddings_type == "elmo_sectioned"):
        # assert num_words == embeddings_list.shape[1]
        sum = torch.zeros((3, 1024))
        for i in range(0, 3):
            for j in range(start, end):
                e_test=embeddings_list[i][j]
                e_test_list.append(e_test)
                sum[i] += e_test * weight_list[j]
            sum[i] = sum[i] / float(num_words)

        return sum
    elif (embeddings_type == "elmo_transformer"):
        # assert num_words == embeddings_list.shape[1]
        sum = torch.zeros((1, 1024))
        for i in range(0, 1):
            for j in range(start, end):
                e_test = embeddings_list[i][j]
                e_test_list.append(e_test)
                sum[i] += e_test * weight_list[j]
            sum[i] = sum[i] / float(num_words)
        return sum

    elif (embeddings_type == "glove"):
        # assert num_words == embeddings_list.shape[1]
        sum = numpy.zeros((1, embeddings_list.shape[2]))
        for i in range(0, 1):
            for j in range(start, end):
                e_test = embeddings_list[i][j]
                e_test_list.append(e_test)
                sum[i] += e_test * weight_list[j]
            sum[i] = sum[i] / float(num_words)
        return sum

    return 0

def get_oov_weight(tokenized_sents,word2weight,word,method="max_weight"):

    word=wnl.lemmatize(word)

    if(word in word2weight):#
        return word2weight[word]

    if(word in stop_words):
        return 0.0

    if(word in english_punctuations or word in chinese_punctuations):#The oov_word is a punctuation
        return 0.0

    if(method=="max_weight"):#Return the max weight of word in the tokenized_sents
        max=0.0
        for w in tokenized_sents:
            if(w in word2weight and word2weight[w]>max):
                max=word2weight[w]
        return max
    return 0.0

def get_weight_list(word2weight_pretrain, word2weight_finetune, tokenized_sents, lamda, database=""):
    weight_list = []
    for word in tokenized_sents:
        word = word.lower()

        if(database==""):
            weight_pretrain = get_oov_weight(tokenized_sents, word2weight_pretrain, word, method="max_weight")
            weight=weight_pretrain
        else:
            weight_pretrain = get_oov_weight(tokenized_sents, word2weight_pretrain, word, method="max_weight")
            weight_finetune = get_oov_weight(tokenized_sents, word2weight_finetune, word, method="max_weight")
            weight = lamda * weight_pretrain + (1.0 - lamda) * weight_finetune
        weight_list.append(weight)

    return weight_list

def get_normalized_weight(weight_list):
    sum_weight=0.0
    for weight in weight_list:
        sum_weight+=weight
    if(sum_weight==0.0):
        return weight_list

    for i in range(0,len(weight_list)):
        weight_list[i]/=sum_weight
    return weight_list

def get_word_weight(weightfile="", weightpara=2.7e-4):
    """
    Get the weight of words by word_fre/sum_fre_words
    :param weightfile
    :param weightpara
    :return: word2weight[word]=weight : a dict of word weight
    """
    if weightpara <= 0:  # when the parameter makes no sense, use unweighted
        weightpara = 1.0
    word2weight = {}
    word2fre = {}
    with open(weightfile, encoding='UTF-8') as f:
        lines = f.readlines()
    # sum_num_words = 0
    sum_fre_words = 0
    for line in lines:
        word_fre = line.split()
        # sum_num_words += 1
        if (len(word_fre) >= 2):
            word2fre[word_fre[0]] = float(word_fre[1])
            sum_fre_words += float(word_fre[1])
        else:
            print(line)
    for key, value in word2fre.items():
        word2weight[key] = weightpara / (weightpara + value / sum_fre_words)
        # word2weight[key] = 1.0 #method of RVA
    return word2weight