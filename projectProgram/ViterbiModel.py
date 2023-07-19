import nltk
from nltk.corpus import brown
import pandas as pd
import sklearn.feature_selection as fs
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2,SelectKBest
import re
import string
import json
import os

# 检查本地是否有nltk文件，若没有则下载
if not os.path.exists(nltk.data.find('corpora/stopwords')):
    nltk.download()


class Viterbi():
    def __init__(self):
        self.testSize = 0.1 #测试集占比
        self.featureNum = 12 #额外特征数量
        self.modelPath = "./models/viterbiModel13.txt" #模型数据储存地址

    def load_data(self):
        dataset = self.brownData()
        trainSet, testSet = train_test_split(dataset, test_size = self.testSize, random_state = 42)
        self.trainSet = trainSet
        self.testSet = testSet
        return trainSet, testSet

    def load_trainSet(self):
        filename = "../dataSet/trainSet.txt"
        if(os.path.exists(filename)):
            file = open(filename,'r',encoding = "utf-8")
            trainSet = json.loads(file.read())
            file.close()
        return trainSet

    def load_testSet(self):
        filename = "../dataSet/testSet.txt"
        if(os.path.exists(filename)):
            file = open(filename,'r',encoding = "utf-8")
            testSet = json.loads(file.read())
            file.close()
        return testSet

    # trainSet: 训练集; featureNum: 除必选feature外的feature数量
    def train(self, featureNum):
        '''
            使用训练数据计算模型。包含预处理数据、卡方检验筛选特征，计算返回模型
            返回 model[feature][label][feature_value] = feature的feature_value（0，1）在label下的出现频率
        '''
        trainSet = self.load_trainSet()
        # 分离feature与label
        X_train, y_train = self.preprocess(trainSet)
        # 选取binary feature. 因为'word'和'prev_word'两个一定要有
        featureList = list(X_train[0].keys())[2:]
        # chi-square select feature, update featureList
        X = []
        for word in X_train:
            X.append(list(word.values())[2:])
        y = list(y_train)
        #print(X[0])
        #print(y)
        c2, pval = chi2(X,y)
        for i in range(len(c2)):
            if(pd.isna(c2[i])):
                c2[i] = 0
        result = []
        for i in range(featureNum):
            max = -1
            maxindex = -1
            for v in range(len(c2)):
                if(c2[v] > max):
                    max = c2[v]
                    maxindex = v
            c2[maxindex] = 0
            result.append(featureList[maxindex])
        featureList = result
        # 对每一个feature,求它与label的频率矩阵
        model = self.featureModel(X_train, y_train, featureList)
        # 两个必选feature的featureMatrix
        model['word'] = self.wordFreq(trainSet)
        model['prev_word'] = self.prev_label(trainSet)
        # 将featuerList, labelFreq存入模型
        model["featureList"] = featureList
        model["labelFreq"] = self.labelFreq(trainSet)
        # 保存模型到本地
        self.save_model(model, "viterbiModel" + str(featureNum) + ".txt")
        # 测试
        # 我也不懂为什么,但是直接用model就是不行,把model存入本地再载入回来就对了
        model = self.load_model("viterbiModel" + str(featureNum) + ".txt")
        self.test(model)
        return model

    def test(self, model):
        '''
            计算测试数据，并计算结果的F1-score
            返回F1-score
        '''
        testSet = self.load_testSet()
        X_test, y_test = self.preprocess(testSet)
        viterbiMat, backTraceMat = self.viterbi(X_test, model)
        y_pred = self.backTrace(viterbiMat, backTraceMat, model["labelList"])


        confMat = {} # confusion matrix
        for label in model["labelList"]:
            confMat[label] = {}
            confMat[label]["TP"] = 0
            confMat[label]["FN"] = 0
            confMat[label]["FP"] = 0
        for i in range(0,len(y_test)):
            if y_test[i] == y_pred[i]:
                confMat[y_test[i]]["TP"] = confMat[y_test[i]].get("TP") + 1
            if y_test[i] != y_pred[i]:
                confMat[y_test[i]]["FN"] = confMat[y_test[i]].get("FN") + 1
                confMat[y_pred[i]]["FP"] = confMat[y_test[i]].get("FP") + 1

        # macro F1-measure
        # F1_measure = 0
        # for label in model["labelList"]:
        #     if confMat[label]["TP"] == 0:
        #         precision = 0
        #         recall = 0
        #     else:
        #         precision = confMat[label]["TP"] / (confMat[label]["TP"] + confMat[label]["FP"])
        #         recall = confMat[label]["TP"] / (confMat[label]["TP"] + confMat[label]["FN"])
        #         F1_measure += (2*precision*recall)/(precision+recall)
        # F1_measure = F1_measure/len(model["labelList"])

        # micro F1-measure
        TP, FN, FP = 0, 0, 0
        for label in model["labelList"]:
            TP += confMat[label]["TP"]
            FN += confMat[label]["FN"]
            FP += confMat[label]["FP"]
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        F1_measure = (2*precision*recall)/(precision+recall)
        print("precision:", precision, "recall:", recall)
        print("micro f1-measure: ", F1_measure)
        return F1_measure

    def predict(self, inputSentence, model):
        '''
            读入输入语句，处理成字符串数组
            用viterbi和backTrace获取预测标签数组
        '''
        # 处理输入
        X = re.findall(r'\w+|\S+', inputSentence)
        X_test = [self.features(X, i) for i in range(len(X))]
        # 预测
        viterbiMat, backTraceMat = self.viterbi(X_test, model)
        y_pred = self.backTrace(viterbiMat, backTraceMat, model["labelList"])
        for i in range(len(y_pred)):
            print(X[i],"-",y_pred[i])
        return X, y_pred


    def brownData(self):
        '''
            从brown语料库加载数据,并运用brown库自带的pos_tag()得出需要的训练数据
            返回数据库 dataset[sentence_index][word_index][word:0, label:1]
        '''
        tag = []
        dataset = []
        lenth = len(brown.sents())
        print("Loading data from brown dataset...")
        count = 0
        flag = 0
        for sent in brown.sents():
            result = nltk.pos_tag(sent)
            dataset.append(result)
            count += 1
            #打印加载进度
            if(count/lenth > 0.3 and flag == 0):
                print("30%...")
                flag += 1
            elif(count/lenth > 0.5 and flag == 1):
                print("50%...")
                flag += 1
            elif(count/lenth > 0.7 and flag == 2):
                print("70%...")
                flag += 1
        print("Loading complete")
        return dataset



    def features(self, sentence, index):
        '''
            将每个单词向量化
            返回每个单词的特征向量（字典类型）
        '''
        return {
            'word': sentence[index], # word本身
            'prev_word':'BOS' if index==0 else sentence[index-1], # 前一个word
            # 'next_word':'EOS' if index==len(sentence)-1 else sentence[index+1], # 后一个word
            'is_punctuation' : 1 if sentence[index] in string.punctuation else 0, # word是否完全是标点符号
            'is_first_capital':int(sentence[index][0].isupper()), # 首字母是否是大写
            'is_first_word': int(index==0), # 是否是句子中第一个单词
            'is_last_word':int(index==len(sentence)-1), # 是否是句子中最后一个单词
            'is_complete_capital': int(sentence[index].upper()==sentence[index]), # 是否全部大写
            'is_numeric':int(sentence[index].isdigit()), # 是否完全是数字
            'is_alphanumeric': int(bool((re.match('^(?=.*[0-9]$)(?=.*[a-zA-Z])',sentence[index])))), # 是否包含数字
            'word_has_hyphen': 1 if '-' in sentence[index] else 0, # 是否包含 "-"
            'prefix_un-': 1 if sentence[index][:2] == "un" else 0,
            'prefix_in-': 1 if sentence[index][:2] == "in" else 0,
            'prefix_pre-': 1 if sentence[index][:3] == "pre" else 0,
            'prefix_dis-': 1 if sentence[index][:3]== "dis" else 0,
            'prefix_mis-': 1 if sentence[index][:3]== "mis" else 0,
            'prefix_non-': 1 if sentence[index][:3]== "non" else 0,
            'prefix_post-': 1 if sentence[index][:4]== "post" else 0,
            'suffix_-ed': 1 if sentence[index][-2:] == "ed" else 0,
            'suffix_-ly': 1 if sentence[index][-2:] == "ly" else 0,
            'suffix_-ing': 1 if sentence[index][-3:] == "ing" else 0,
            'suffix_-ful': 1 if sentence[index][-3:] == "ful" else 0,
            'suffix_-able': 1 if sentence[index][-4:] == "able" else 0,
            'suffix_-less': 1 if sentence[index][-4:] == "less" else 0,
            'suffix_-ness': 1 if sentence[index][-4:] == "ness" else 0,
            'suffix_-ment': 1 if sentence[index][-4:] == "ment" else 0,
            'suffix_-tion': 1 if sentence[index][-4:] == "tion" else 0
            }

    def untag(self, sentence):
        return [word for word,tag in sentence]

    def preprocess(self, dataset):
        '''
            将数据预处理
            返回X：单词的特征向量 y：单词对应的label
        '''
        X, y=[],[] # X是特征, y是label
        for sentence in dataset:
            for i in range(len(sentence)):
                X.append(self.features(self.untag(sentence), i))
                y.append(sentence[i][1])
        return X, y

    def save_model(self, model, filename):
        '''
            存储模型数据到本地文件
            返回文件地址
        '''
        filename = "../models/" + filename
        file = open(filename,'w',encoding = 'utf-8')
        file.write(json.dumps(model))
        file.close()
        self.modelPath = filename
        return filename

    def load_model(self, filename):
        '''
            加载本地存储的模型数据
            返回模型model
        '''
        filename = "../models/" + filename

        if(os.path.exists(filename)):
            file = open(filename,'r',encoding = "utf-8")
            model = json.loads(file.read())
            file.close()
        else:
            print("no such model")
            exit(0)
        self.model = model
        return model

    def wordFreq(self, sentences):
        '''
            得出每个词汇标为各个label的概率（包含平滑操作）
            返回wordsProb[label][word] = P(label | word)
        '''
        wordsProb = {} #词语在每个标铅出现的概率

        ## test
        filename = "../models/wordFreq.txt"
        if(os.path.exists(filename)):
            file = open(filename,'r',encoding = "utf-8")
            wordsProb = json.loads(file.read())
            #print(type(wordsProb))
            file.close()
            #print(type(wordsProb[',']))
            return wordsProb
        ## test

        tags = [] #动态添加遇到的tags
        wordsFreq = [] #一个装字典的数组，每个字典对应一个tag
        dictionary = [] #出现的所有词
        #统计每个单词在各个label中的出现频率
        for sentence in sentences:
            for word in sentence:
                w,t = word[0],word[1]
                if(w not in dictionary):
                    dictionary.append(w)
                if(t not in tags):
                    tags.append(t)
                    dic = {}
                    dic[w] = dic.get(w,0) + 1
                    wordsFreq.append(dic)
                else:
                    index = tags.index(t)
                    wordsFreq[index][w] = wordsFreq[index].get(w,0) + 1
        for tag in tags:
            wordsProb[tag] = {}
        BOS = {}    #加入额外的BOS(begining of sentence)label和token
        BOS['BOS'] = 1
        wordsProb['BOS'] = BOS
        #根据以上的单词出现频率得出其概率
        for word in dictionary:
            wordFreq = []
            for tag in tags:
                wordFreq.append(wordsFreq[tags.index(tag)].get(word,0))
            for tag in tags:
                wordsProb[tag][word] = (wordFreq[tags.index(tag)] + 1)/(sum(wordFreq)+len(tags)) #smoothing
            wordsProb['BOS'][word] = 0
        for tag in tags:
            wordsProb[tag]['BOS'] = 0
        ## test
        file = open(filename,'w',encoding = 'utf-8')
        file.write(json.dumps(wordsProb))
        file.close()
        ## test
        return wordsProb

    def prev_label(self, trainSentences):
        '''
            计算每对label相邻的概率
            返回prev_label_dict[label][prev_label]=P(label | prev_label)
        '''
        prev_label_dict = {}
        count_label = {}
        count_label["EOS"] = len(trainSentences)
        count_label["BOS"] = len(trainSentences)
        for sentence in trainSentences:
            prev_label = "BOS"
            prev_label_dict["EOS"] = {}
            for word, label in sentence:
                if label not in prev_label_dict:
                    prev_label_dict[label] = {}
                    prev_label_dict["BOS"] = {}
                prev_label_dict[label][prev_label] = prev_label_dict[label].get(prev_label, 0) + 1
                count_label[label] = count_label.get(label,0) + 1
                prev_label = label
            if prev_label not in prev_label_dict["EOS"]:
                prev_label_dict["EOS"][prev_label] = 0
            prev_label_dict["EOS"][prev_label] = prev_label_dict["EOS"].get(prev_label,0) + 1

        for label in prev_label_dict:
            for l in count_label:
                if l not in prev_label_dict[label]:
                    prev_label_dict[label][l] = 0
            for prev_label in prev_label_dict[label]:
                prev_label_dict[label][prev_label] = (prev_label_dict[label][prev_label] + 1)/(count_label[label] + len(count_label))
        return prev_label_dict

    def labelFreq(self, trainSentences):
        labelFreq = {}
        cntLabel = 0
        cntSentence = 0
        for sentence in trainSentences:
            for word, label in sentence:
                labelFreq[label] = labelFreq.get(label, 0) + 1
                cntLabel += 1
            cntSentence += 1
        labelFreq["BOS"] = cntSentence
        for label in list(labelFreq.keys()):
            labelFreq[label] = labelFreq[label] / (cntLabel + cntSentence)
        return labelFreq

    # X: feature vector集; y: label集; featureList: 选择的feature, 不包含必选的'word','prev_word'
    def featureModel(self, X_train, y_train, featureList):
        '''
            对每一个feature,求featureMatrix - 它与label的频率矩阵
            返回model[feature][label][feature value] = feature的feature_value（0，1）在label下的出现频率
        '''
        model = {}
        labelList = list(set(y_train))
        labelList.append("BOS")
        model["labelList"] = labelList
        # 其它feature都是binary classify, feature value = 0/1
        for feature in featureList:
            model[feature] = {}
        # cntLabel[label]: label在train set中出现次数
        cntLabel = {}
        # 初始化model[feature]和cntLabel
        for label in labelList:
            cntLabel[label] = 0
            for feature in featureList:
                model[feature][label] = {}
        # 对于training set中每一个feature vector
        for i, x in enumerate(X_train, 0):
            # curY: 这个feature vector对应的label
            curY = y_train[i]
            cntLabel[curY] += 1
            for feature in featureList:
                # curX: 这个feature vector中这个feature的feature value
                curX = x[feature]
                # 目前model中存储的是feature的curX在curY下的出现次数.
                model[feature][curY][curX] = model[feature][curY].get(curX, 0) + 1
        # 修改model[feature][label][feature_value]为feature的feature_value在label下的出现频率.
        for feature in featureList:
            for label in list(set(y_train)):
                for feature_value in model[feature][label]:
                    model[feature][label][feature_value] = model[feature][label][feature_value] / cntLabel[label]
        return model

    def viterbi(self, X_test, model):
        '''
            实现viterbi encoding算法（包含额外的特征分析）

        '''
        viterbiMat = [] # 存储每个word的每个label的viterbi值. 用list解决重复word问题
        backTraceMat = [] # 存储每个word的每个label的max preLabel(提供最大viterbi的preLabel)
        featureList = model["featureList"] # 从Model中获取featureList, 其中不包含word和prev_word
        labelList = model["labelList"] # 从model中获取labelList
        wordFreq = model["word"] # wordFreq[label][word] = P(word | label)
        prevLabel = model["prev_word"] # prevLabel[label][prev_label] = P(prev_label | label)
        labelFreq = model["labelFreq"] # labelFreq[label] = P(label)
        # 遍历Xtest中每一个word(feature set)
        for i, x in enumerate(X_test, 0):
            # 每个viterbiList中存储一个x对所有label的概率 P(label|word)
            viterbiList = []
            # backTraceList中存储一个x，使每个label得到最大viterbi值的prev_label
            backTraceList = []
            # 求x对每一个label的viterbi值, 即 P(label|word) = P(f0|label)*P(f1|label)...P(fn|label)
            for label in labelList:
                viterbiValue = 1 # 初始化viterbi值
                # *P(word|label)
                word = x["word"]
                # 如果word是OutOfVocabulary, 用label frequency替代
                if word not in wordFreq[labelList[0]]:
                    viterbiValue *= labelFreq[label]
                    # viterbiValue *= 1
                else:
                    viterbiValue *= wordFreq[label][word]

                # *P(prev_word|label) = *P(prev_label|label)*P(prev_label|prev_word)
                # 如果x的prev_word是BeginOfSentence, 则prev_label=BOS, P(prev_label|prev_word) = 1
                if x["prev_word"] == "BOS":
                    viterbiValue *= prevLabel[label]["BOS"] * 1
                    backTraceList.append("BOS")
                else:
                    # 通过index获取prev_word的viterbiList
                    prev_viterbiList = viterbiMat[i-1]
                    # 遍历prev_word的prev_label, 使P(prev_label|label)*P(prev_label|prev_word)最大
                    prevLabel_viterbi = [] # 将从每个prev_label得到的viterbi值存在这里
                    for j, prev_label in enumerate(labelList, 0):
                        # label在viterbiList与labelList中的顺序一样
                        prevLabel_viterbi.append(prevLabel[label][prev_label] * prev_viterbiList[j])
                    # 取prevLabel_viterbi中最大的作为 P(prev_word|label)
                    viterbiValue *= max(prevLabel_viterbi)
                    # 添加prev_label到backTraceList
                    backTraceList.append(labelList[prevLabel_viterbi.index(max(prevLabel_viterbi))])

                # 计算其它每个feature的P(feature|label)
                for feature in featureList:
                    feature_value = str(x[feature])
                    # print(feature, label, feature_value)
                    viterbiValue *= model[feature][label].get(feature_value, 0)
                viterbiList.append(viterbiValue)
            # 特殊处理标点符号. 所有标点符号 word = label. 如 word = "," -> label = ","
            if string.punctuation in x or x == "''":
                y = x
                viterbiList[:] = 0
                viterbiList[labelList.indexOf(y)] = 1
            viterbiMat.append(viterbiList)
            backTraceMat.append(backTraceList)

        return viterbiMat, backTraceMat

    def backTrace(self, viterbi, backTraceMat, labelList):
        '''
            反向计算找出路径，得出每个单词的判断结果
            返回判断结果pred_y[word_index][word:0, label:1]
        '''
        pred_y = []
        curLabel = "BOS"
        i = len(viterbi) - 1
        while i >= 0:
            if curLabel == "BOS":
                curLabel = labelList[viterbi[i].index(max(viterbi[i]))]
            pred_y.insert(0, curLabel)
            curLabel = backTraceMat[i][labelList.index(curLabel)]
            i -= 1

        return pred_y
