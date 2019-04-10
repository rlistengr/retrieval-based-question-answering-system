import json
import math
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import numpy as np


def read_corpus():
    """读取语料库
    读取../data/train-v2.0.json的语料库，并把问题列表和答案列表分别写入到 qlist, alist 里面。
    qlist = ["问题1"， “问题2”， “问题3” ....]
    alist = ["答案1", "答案2", "答案3" ....]
    每一个问题和答案对应（下标一致）
    
    Returns：
        问题list和答案list
    """
    qlist = []
    alist = []
    
    with open("../data/train-v2.0.json") as file:
        info = json.load(file)
        data = info['data']
        for one_doc in data:
            paragraphs = one_doc['paragraphs']
            for qadict in paragraphs:
                qas = qadict['qas']
                for qa in qas:
                    qlist.append(qa['question'])
                    if qa['is_impossible']:
                        alist.append('impossible')
                    else:
                        alist.append(qa['answers'][0]['text'])
    
    assert len(qlist) == len(alist)  # 确保长度一样
    return qlist, alist


_qlist, _alist = read_corpus()

# 做简单的分词，对于英文我们根据空格来分词即可，其他过滤暂不考虑（只需分词）   
# 生成每个单词为key，词频为value的dict
_word_frequency = {}
_word_total = 0

for question in _qlist:
    # 去除标点符号
    for punctuation in '~!@#$%^&*()_+-={}|:"<>?[]\;,./—':
        question = question.replace(punctuation, '')
    question = question.split()
    for word in question:
        _word_frequency[word] = _word_frequency.get(word, 0) + 1
        _word_total += 1

 
# 根据词频排序
_word_frequency_list = list(_word_frequency.items())
_word_frequency_list.sort(key=lambda x:x[1], reverse=True)


def text_preprocess(text, min_frequency):
    """文本预处理
    对问题和答案做预处理，
        1. 停用词过滤 
        2. 转换成lower_case
        3. 去掉出现频率小于min_frequency的词
        4. 把所有数字看做同一个单词，这个新的单词定义为 "#number"
        5. stemming，提取词干
        
    Args：
        text:待处理的文本，list格式，实际上是上面读取到的问题list或者答案list
        min_frequency：词语出现的最小频率
    Returns：
        处理过之后的文本，list格式
        词汇表，词频在min_frequency以上
    """

    stops = set(stopwords.words('english'))
    porter_stemmer = PorterStemmer()
    word_frequency = {}
    newtext = []
    tmptext = []
            
    for sentence in text:
        newsentence = []
        # 去除标点符号和停用词
        for punctuation in '~!@#$%^&*()_+-={}|:"<>?[]\;,./—':
            sentence = sentence.replace(punctuation, '')
        
        # 转小写
        sentence = sentence.lower()
        
        # 分词
        sentence = sentence.split()
        
        # 停用词过滤
        sentence = [word for word in sentence if word not in stops]

        # 统计词频，后续只保留min_frequency以上的单词，
        # 同时把数字转化为#number
        # 同时提取词干
        for word in sentence:
            word = porter_stemmer.stem(word)

            if word.isdigit():
                word = '#number'

            newsentence.append(word)

            word_frequency[word] = word_frequency.get(word, 0) + 1
            
        tmptext.append(newsentence)
        
    # 只保留min_frequency以上的单词
    word_dict = [word for (word, frequency) in word_frequency.items() if frequency > min_frequency]
    for sentence in tmptext:
        sentence = [word for word in sentence if word in word_dict]
        newtext.append(' '.join(sentence))
    
    return newtext, word_dict
    
_stem_qlist, _qword_dict = text_preprocess(_qlist, 10)
_stem_alist = text_preprocess(_alist, 10)


qlist, alist = _stem_qlist, _stem_alist   # 更新后的


# 把qlist中的每一个问题字符串转换成tf-idf向量, 转换之后的结果存储在X矩阵里。 
# X的大小是： N* D的矩阵。 这里N是问题的个数（样本个数），
# D是字典库的大小。 

def generate_tfidf_vertor(text, word_dict, word_index_dict):
    """生成tf-idf向量
    根据输入的文本，就是上面经过处理之后的问题list，把每个问题表示成一个tf-idf向量
    
    Args：
        text：文本，list格式
        word_dict：词汇表
        word_index_dict：每个单词对应的索引dict，即每个单词在词袋中的下标
        
    Returns：
        tfidf矩阵
        每个单词的词频
    """

    tfidfmatrix = [[0]*len(word_dict) for i in range(len(text))]
    word_idf = {}
    nsentence = len(text)
    
    # 遍历每一个question，统计各单词数量，填入tfidfmatrix
    for row, sentence in enumerate(text):
        sentence = sentence.split()
        sentence_word_frequency = {}
        for word in sentence:
            sentence_word_frequency[word] = sentence_word_frequency.get(word, 0) + 1
        
        for word, frequency in sentence_word_frequency.items():
            # 计算tf
            tfidfmatrix[row][word_index_dict[word]] = frequency
            # 统计每个单词的question出现数
            word_idf[word] = word_idf.get(word, 0) + 1

    # 遍历每一个question，对其每个单词做tfidf计算
    for row, sentence in enumerate(text):
        sentence = sentence.split()
        sentence = set(sentence)
        for word in sentence:
            tfidfmatrix[row][word_index_dict[word]] = tfidfmatrix[row][word_index_dict[word]] * math.log(nsentence/word_idf[word], 10)
    return tfidfmatrix, word_idf

# 根据word_list生成每个单词对应的数组下标
_word_index_dict = {}
for index, word in enumerate(_qword_dict):
    _word_index_dict[word] = index
    
_word_idf = {}
X, _word_idf = generate_tfidf_vertor(_stem_qlist, _qword_dict, _word_index_dict)


def question_preprocess(input_q):
    """输入问题预处理
        1. 停用词过滤 
        2. 转换成lower_case
        3. 去掉出现频率小于min_frequency的词
        4. 把所有数字看做同一个单词，这个新的单词定义为 "#number"
        5. stemming，提取词干
        
    Args:
        input_q:用户的提问
    
    Returns：
        预处理之后的问题
    """
    stops = set(stopwords.words('english'))
    porter_stemmer = PorterStemmer()
    newquestion = []

    # 去除标点符号
    for punctuation in '~!@#$%^&*()_+-={}|:"<>?[]\;,./—':
        input_q = input_q.replace(punctuation, '')
    # 转小写
    input_q = input_q.lower()

    # 分词
    input_q = input_q.split()
    
    # 停用词过滤
    input_q = [word for word in input_q if word not in stops]

    # 数字转化为#number
    # 同时提取词干
    for word in input_q:
        word = porter_stemmer.stem(word)

        if word.isdigit():
            word = '#number'

        newquestion.append(word)

    # 只保留词典里的单词
    newquestion = [word for word in newquestion if word in _qword_dict]

    return newquestion  
    

# 利用倒排表对问题的搜索进行优化
def inverted_index_generate(text):
    """生成倒排表
    以单词为key，问题下标为value的倒排表，格式为dict
    
    Args：
        text：问题list
    
    Returns：
        倒排表
    """
    inverted_index = {}
    
    for index, sentence in enumerate(text):
        sentence = sentence.split()
        
        sentence = set(sentence)
        
        for word in sentence:
            if word in inverted_index:
                inverted_index[word].append(index)
            else:
                inverted_index[word] = [index, ]

    return inverted_index
    

def candidate_filter(input_q, inverted_index):
    """候选者过滤
    根据输入问题中的单词，找到至少有一个单词相同的问题的集合
    
    Args:
        input_q:输入问题
        inverted_index：候选问题
        
    Returns:
        候选问题集合
    """

    candidate_question = set()

    for word in input_q:
        if word in inverted_index:
            candidate_question = candidate_question.union(inverted_index[word])

    return candidate_question


_inverted_idx = inverted_index_generate(_stem_qlist)# 定一个一个简单的倒排表

def top5results_invidx(input_q):
    """
    给定用户输入的问题 input_q, 返回最有可能的TOP 5答案。
    1. 利用倒排表来筛选 candidate
    2. 对于用户的输入 input_q 首先做一系列的预处理，然后再转换成tf-idf向量（利用上面的vectorizer)
    3. 计算跟每个库里的问题之间的相似度
    4. 找出相似度最高的top5问题的答案
    
    Args:
        input_q:输入问题
    
    Returns：
        匹配度最高的五个问题的答案，但是有些问题的答案可能不存在，显示为impossible，这里没做过滤
    """
    word_position = {}
    ndoc = len(_qlist)
    top_idxs = []
    min_similarity = 0
    nonzero_position = None
    nonzero_value = None
    nonzero = {}
    res = []
    
    input_q = question_preprocess(input_q)
    
    candidate_question = candidate_filter(input_q, _inverted_idx)

    # 统计每个单词的出现数量,关键字是单词索引值
    for word in input_q:
        nonzero[_word_index_dict[word]] = nonzero.get(_word_index_dict[word], 0) + 1
        word_position[_word_index_dict[word]] = word
        
    nonzero_position = list(nonzero.keys())
    nonzero_value = list(nonzero.values())
    # 转换成tf-idf
    for i in range(len(nonzero_position)):
        nonzero_value[i] = nonzero_value[i] * math.log(ndoc/_word_idf[word_position[nonzero_position[i]]], 10)

    abs_question = np.sqrt(sum((np.power(nonzero_value, 2))))
    
    for index in candidate_question:
        vector = X[index]
        abs_vector = [x for x in vector if x != 0]
        abs_vector = np.sqrt(sum((np.power(abs_vector, 2))))
        if abs_vector == 0:
            continue
        tmp = [vector[position] for position in nonzero_position]
        
        similarity = sum(np.array(tmp)*np.array(nonzero_value)) / (abs_question * abs_vector)
        # print(_qlist[index])
        # print(similarity)
        if similarity > min_similarity or len(top_idxs) < 5:
            if len(top_idxs) == 5:
                top_idxs.pop()
            top_idxs.append((similarity, index))
            top_idxs.sort(reverse = True)
            min_similarity = min(top_idxs)[0]
   
    for k, v in top_idxs:
        # print(_qlist[v])
        # print(_alist[v])
        res.append(_alist[v]) 
    
    return res
    
    
def emb_dict_generate(path):
    """读取词向量文件，生成词向量dict
    
    Args：
        path：词向量文件的目录
    Returns：
        词向量的dict，value是词向量
    """
    emb_dict = {}

    with open(path, 'r', encoding = 'utf-8') as text:
        for line in text:
            line = line.split()
            emb_dict[line[0]] = list(map(float, line[1:]))

    return emb_dict
    
def one_emb_generate(emb_dict, sentence):
    """生成一个问题的词向量表达
    使用所有单词的平均词向量表示整个句子
    
    Args：
        emb_dict：词向量dict
        sentence：问题或者句子
    
    Returns：
        问题或者句子的词向量表达
    """
    count = 0  
    sum = [0] * 100
    
    # 去除标点符号
    for punctuation in '~!@#$%^&*()_+-={}|:"<>?[]\;,./—':
        sentence = sentence.replace(punctuation, '')
    # 转小写
    sentence = sentence.lower() 
    sentence = sentence.split()
    
    for word in sentence:
        if word in emb_dict:
            sum = np.add(sum, emb_dict[word])
            count += 1
            
    if count != 0:
        sum = np.divide(sum, count)

    return sum
 
def emb_matrix_generate(emb_dict, text):
    """将问题list转化为词向量表示的矩阵
    
    Args：
        emb_dict：词向量dict
        text：问题list
    Returns：
        用词向量表示的问题矩阵
    """
    sum = [0] * 100
    count = 0
    emb_matrix = []
    
    for sentence in text:
        emb_matrix.append(one_emb_generate(emb_dict, sentence))

    return emb_matrix

# 词向量文件不做上传，可以从这里下载https://nlp.stanford.edu/projects/glove/
_emb_dict = emb_dict_generate('../data/glove.6B.100d.txt')

_emb = emb_matrix_generate(_emb_dict, _qlist)

def top5results_emb(input_q):
    """
    给定用户输入的问题 input_q, 返回最有可能的TOP 5答案
    1. 利用倒排表来筛选 candidate
    2. 对于用户的输入 input_q，转换成句子向量
    3. 计算跟每个库里的问题之间的相似度
    4. 找出相似度最高的top5问题的答案
    """
    input_vector = one_emb_generate(_emb_dict, input_q)
    tmp_q = question_preprocess(input_q)
    candidate_question = candidate_filter(tmp_q, _inverted_idx)
    top_idxs = []
    min_similarity = 0
    res = []
    
    abs_question = np.sqrt(sum((np.power(input_vector, 2))))
 
    for index in candidate_question:
        vector = _emb[index]
        abs_vector = np.sqrt(sum((np.power(vector, 2))))
        if abs_vector == 0:
            continue
        
        similarity = sum(input_vector * vector) / (abs_question * abs_vector)
        if similarity > min_similarity or len(top_idxs) < 5:
            if len(top_idxs) == 5:
                top_idxs.pop()            
            top_idxs.append((similarity, index))
            top_idxs.sort(reverse = True)
            min_similarity = min(top_idxs)[0]
    
    for k, v in top_idxs:
        # print(_qlist[v])
        # print(_alist[v])
        res.append(_alist[v]) 
    
    return res  