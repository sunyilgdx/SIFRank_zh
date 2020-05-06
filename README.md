# SIFRank_zh
这是我们论文的相关代码 [SIFRank: A New Baseline for Unsupervised Keyphrase Extraction Based on Pre-trained Language Model](https://ieeexplore.ieee.org/document/8954611)
原文是在对英文关键短语进行抽取，这里迁移到中文上，部分pipe进行了改动。英文原版在[这里](https://github.com/sunyilgdx/SIFRank)。

## 版本介绍
* 2020/03/03——``最初版本``
本版本中只包含了最基本的功能，部分细节还有待优化和扩充。

## 核心算法
### 预训练模型ELMo+句向量模型SIF
词向量ELMo优势：1）经过大规模预训练，较TFIDF、TextRank等基于统计和图的具有更多的语义信息；2）ELMo是动态的，可以改善一词多义问题；3）ELMo通过Char-CNN编码，对生僻词非常友好；4）不同层的ELMo可以捕捉不同层次的信息

句向量SIF优势：1）根据词频对词向量进行平滑反频率加权，能更好地捕捉句子的中心话题；2）更好地过滤通用词
### 候选关键短语识别
首先对句子进行分词和词性标注，再利用正则表达式确定名词短语（例如：形容词+名词），将名词短语作为候选关键短语
### 候选关键短语重要程度排序（SIFRank）
利用相同的算法计算整个文档（或句子）和候选关键短语的句向量，再依次进行相似度计算（余弦距离），作为重要程度
### 文档分割（document segmentation，DS）+词向量对齐（embeddings alignment，EA）
DS：通过将文档分为较短且完整的句子（如16个词左右），并行计算来加速ELMo；
EA：同时利用锚点词向量对不同句子中的相同词的词向量进行对齐，来稳定同一词在相同语境下的词向量表示。
### 位置偏权（SIFRank+）
核心思想：对于长文本，`先出现的词往往具有更重要的地位`

因此利用每个词第一次出现的位置来产生权重:1/p+u（还要经过一个softmax拟合），u是一个超参数，经过实验设置为3.4

## 环境
```
Python 3.6
nltk 3.4.3
elmoformanylangs 0.0.3
thulac 0.2.1
torch 1.1.0
```
## 提示
哈工大的elmoformanylangs 0.0.3中有个较为明显的问题，当返回所有层Embeddings的时候代码写错了，当output_layer=-2时并不是返回所有层的向量，只是返回了倒数第二层的。问题讨论在这里[#31](https://github.com/HIT-SCIR/ELMoForManyLangs/issues/31)
```
elmo.sents2elmo(sents_tokened,output_layer=-2)
```
建议这样修改elmo.py里class Embedder(object)类中的代码。

原代码：
```
if output_layer == -1:
     payload = np.average(data, axis=0)
else:
     payload = data[output_layer]
```
修改后：
```
if output_layer == -1:
     payload = np.average(data, axis=0)
 #code changed here
 elif output_layer == -2:
     payload = data
 else:
     payload = data[output_layer]
```


## 下载
* 哈工大ELMo ``zhs.model`` 请从[这里](https://github.com/HIT-SCIR/ELMoForManyLangs) 下载,将其解压保存到 ``auxiliary_data/``目录下（注意要按照其要求更改config文件），本项目中已经将部分文件上传了，其中比较大的模型文件``encoder.pkl``和``token_embedder.pkl``请自行添加。
* 清华分词工具包THULAC ``thulac.models`` 请从[这里](http://thulac.thunlp.org/)下载, 将其解压保存到 ``auxiliary_data/``目录下。

## 用法
```
from embeddings import sent_emb_sif, word_emb_elmo
from model.method import SIFRank, SIFRank_plus
import thulac

#download from https://github.com/HIT-SCIR/ELMoForManyLangs
model_file = r'../auxiliary_data/zhs.model/'

ELMO = word_emb_elmo.WordEmbeddings(model_file)
SIF = sent_emb_sif.SentEmbeddings(ELMO, lamda=1.0)
#download from http://thulac.thunlp.org/
zh_model = thulac.thulac(model_path=r'../auxiliary_data/thulac.models/',user_dict=r'../auxiliary_data/user_dict.txt')
elmo_layers_weight = [0.5, 0.5, 0.0]

text = "计算机科学与技术（Computer Science and Technology）是国家一级学科，下设信息安全、软件工程、计算机软件与理论、计算机系统结构、计算机应用技术、计算机技术等专业。 [1]主修大数据技术导论、数据采集与处理实践（Python）、Web前/后端开发、统计与数据分析、机器学习、高级数据库系统、数据可视化、云计算技术、人工智能、自然语言处理、媒体大数据案例分析、网络空间安全、计算机网络、数据结构、软件工程、操作系统等课程，以及大数据方向系列实验，并完成程序设计、数据分析、机器学习、数据可视化、大数据综合应用实践、专业实训和毕业设计等多种实践环节。"
keyphrases = SIFRank(text, SIF, zh_model, N=15,elmo_layers_weight=elmo_layers_weight)
keyphrases_ = SIFRank_plus(text, SIF, zh_model, N=15, elmo_layers_weight=elmo_layers_weight)
```
## 用例展示
我们选取了一段百度百科中关于“计算机科学与技术”的描述作为被抽取对象（如下），用Top10来观察其效果。
> text = "计算机科学与技术（Computer Science and Technology）是国家一级学科，下设信息安全、软件工程、计算机软件与理论、计算机系统结构、计算机应用技术、计算机技术等专业。 [1]主修大数据技术导论、数据采集与处理实践（Python）、Web前/后端开发、统计与数据分析、机器学习、高级数据库系统、数据可视化、云计算技术、人工智能、自然语言处理、媒体大数据案例分析、网络空间安全、计算机网络、数据结构、软件工程、操作系统等课程，以及大数据方向系列实验，并完成程序设计、数据分析、机器学习、数据可视化、大数据综合应用实践、专业实训和毕业设计等多种实践环节。"
* SIFRank_zh抽取结果
```
关键词         权重
大数据技术导论  0.9346
计算机软件      0.9211
计算机系统结构  0.9182
高级数据库系统  0.9022
计算机网络      0.8998
媒体大数据案例  0.8997
数据结构       0.8971
软件工程       0.8955
大数据         0.8907
计算机技术     0.8838
```
* SIFRank+_zh抽取结果
```
关键词         权重
计算机软件       0.9396
计算机科学与技术  0.9286
计算机系统结构    0.9245
大数据技术导论    0.9222
软件工程         0.9213
信息             0.8787
计算机技术       0.8778
高级数据库系统    0.8770
computer        0.8717
媒体大数据案例    0.8687
```
* jieba分词TFIDF抽取结果
```
关键词         权重
数据         0.8808
可视化       0.5891
技术         0.3726
机器         0.3496
毕业设计     0.3369
专业         0.3260
网络空间     0.3235
数据库系统   0.2983
数据结构     0.2801
计算技术     0.2738
```
* jieba分词TextRank抽取结果
```
关键词         权重
数据        1.0000
技术        0.4526
可视化      0.3170
计算机系统  0.2488
机器        0.2420
结构        0.2371
计算机      0.2365
专业        0.2121
网络空间    0.2103
计算技术    0.1954
```

## 分析
我们的SIFRank和SIFRank+采用了动态预训练词向量模型ELMo和句向量模型SIF，用完全无监督的方法进行关键短语（keyphrase）的抽取，相比于jieba的TFIDF和TextRank算法，不仅抽取的关键词更加完整，且由于引入了预训练的知识，关键词之间的关系更为丰富，不再仅限于句子结构本身。

此外，清华的分词模型支持自定义用户词典，可以保持专有名词的完整性，并且通过ELMo的CNN编码层，对专有名词的识别和编码效果会更好。

## 引用
If you use this code, please cite this paper
```
@article{DBLP:journals/access/SunQZWZ20,
  author    = {Yi Sun and
               Hangping Qiu and
               Yu Zheng and
               Zhongwei Wang and
               Chaoran Zhang},
  title     = {SIFRank: {A} New Baseline for Unsupervised Keyphrase Extraction Based
               on Pre-Trained Language Model},
  journal   = {{IEEE} Access},
  volume    = {8},
  pages     = {10896--10906},
  year      = {2020},
  url       = {https://doi.org/10.1109/ACCESS.2020.2965087},
  doi       = {10.1109/ACCESS.2020.2965087},
  timestamp = {Fri, 07 Feb 2020 12:04:22 +0100},
  biburl    = {https://dblp.org/rec/journals/access/SunQZWZ20.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
