from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
import numpy
from ast import literal_eval
import re
import math
from collections import defaultdict
import string
import gensim
import sys
import random
import copy

classification_matrix = defaultdict(dict)
evaluation = defaultdict()
trainset = defaultdict(dict)
tale_word_dict_trainset = defaultdict(dict)

def load_docs():
    with open("corpora.txt", "r") as f:
        whole_text = f.read()
        new_t = literal_eval(whole_text)
        evalset = defaultdict(list)
        whole_trainset = defaultdict(list)
        tale_word_dict_evalset = defaultdict(defaultdict)
        tale_word_whole_dict = defaultdict(defaultdict)
        vocabular = defaultdict(int)

        def save_by_tale_type(tales,talesupertype, talestuple, tokenss):

            tales[talesupertype][talestuple[1]] = [talestuple[0], talestuple[2], tokenss]
            for token in tokenss:
                if talestuple[1] not in tale_word_dict_trainset[talesupertype].keys():
                    tale_word_dict_trainset[talesupertype][talestuple[1]] = {}

                if token in tale_word_dict_trainset[talesupertype][talestuple[1]]:
                    tale_word_dict_trainset[talesupertype][talestuple[1]][token] += 1
                else:
                    tale_word_dict_trainset[talesupertype][talestuple[1]][token] = 1

            return tales

        def create_testsets(corpus,modus):
            for doc in corpus:
                if doc in ["English","German","French","Spanish"]:
                    print("Load document:" + doc)
                    for tuple in new_t[doc]:

                        mod_text = (tuple[4]).replace("\'", "\"").replace("»", "\"").replace("«", "\"").lower()
                        mod_text = re.sub(r"<.*?>", r" ", mod_text)
                        # split into tokens by white space
                        if modus == 1:
                            tokens = mod_text.split()
                        else:
                            tokens = word_tokenize(mod_text)
                        # convert to lower case
                        tokens = [w.lower() for w in tokens]
                        # prepare regex for char filtering
                        re_punc = re.compile('[%s]' % re.escape(string.punctuation))
                        # remove punctuation from each word
                        tokens = [re_punc.sub('', w) for w in tokens]
                        # remove remaining tokens that are not alphabetic
                        tokens = [word for word in tokens if word.isalpha()]

                        if modus == 3:
                        # filter out stop words

                            if doc == "English":
                                stop_words = set(stopwords.words('english'))
                                tokens = [w for w in tokens if not w in stop_words]
                                #stemmer(tokens)

                            elif doc == "German":
                                stop_words = set(stopwords.words('german'))
                                tokens = [w for w in tokens if not w in stop_words]
                                #stemmer(tokens)

                            elif doc == "Spanish":
                                stop_words = set(stopwords.words('spanish'))
                                tokens = [w for w in tokens if not w in stop_words]
                                #stemmer(tokens)

                            else:
                                stop_words = set(stopwords.words('french'))
                                tokens = [w for w in tokens if not w in stop_words]
                                #stemmer(tokens)

                        # stemming of words
                        if tuple[2] =="UNKNOWN":
                            evalset[(tuple[1])] = [tuple[0], tokens]

                            for token in tokens:
                                if token not in tale_word_dict_evalset[tuple[1]]:
                                    tale_word_dict_evalset[tuple[1]][token] = 1
                                else:
                                    tale_word_dict_evalset[tuple[1]][token] +=1

                        elif ((list(tuple[2]))[-1]).isdigit():
                            # Zeile sorgt dafür, dass String zu Buchstaben werden

                            if int(tuple[2]) < 300:
                                save_by_tale_type(trainset, "Animal Tales", tuple,tokens)

                            elif int(tuple[2]) < 750:
                                save_by_tale_type(trainset, "Tales of Magic",tuple,tokens)

                            elif int(tuple[2]) < 850:
                                save_by_tale_type(trainset, "Religious Tales",tuple,tokens)

                            elif int(tuple[2]) < 1000:
                                save_by_tale_type(trainset, "Realistic Tales",tuple,tokens)

                            elif int(tuple[2]) < 1200:
                                save_by_tale_type(trainset, "Tales of the stupid Ogre",tuple,tokens)

                            elif int(tuple[2]) < 2000:
                                save_by_tale_type(trainset, "Anecdotes and Jokes",tuple,tokens)

                            elif int(tuple[2]) < 2400:
                                save_by_tale_type(trainset, "Formula Tales",tuple,tokens)

                            else:
                                whole_trainset[tuple[1]] =[tuple[0], tuple[2], tokens]
                                for token in tokens:
                                    if token not in tale_word_whole_dict[tuple[1]]:
                                        tale_word_whole_dict[tuple[1]][token] = 1
                                    else:
                                        tale_word_whole_dict[tuple[1]][token] += 1

                        else:
                            if int("".join((list(tuple[2]))[0:-1])) < 300:
                                save_by_tale_type(trainset,"Animal Tales", tuple,tokens)

                            elif int("".join((list(tuple[2]))[0:-1])) < 750:
                                save_by_tale_type(trainset,"Tales of Magic", tuple,tokens)

                            elif int("".join((list(tuple[2]))[0:-1])) < 850:
                                save_by_tale_type(trainset, "Religious Tales",tuple,tokens)

                            elif int("".join((list(tuple[2]))[0:-1])) < 1000:
                                save_by_tale_type(trainset, "Realistic Tales",tuple,tokens)

                            elif int("".join((list(tuple[2]))[0:-1])) < 1200:
                                save_by_tale_type(trainset, "Tales of the stupid Ogre",tuple,tokens)

                            elif int("".join((list(tuple[2]))[0:-1])) < 2000:
                                save_by_tale_type(trainset, "Anecdotes and Jokes",tuple,tokens)

                            elif int("".join((list(tuple[2]))[0:-1])) < 2400:
                                save_by_tale_type(trainset, "Formula Tales",tuple,tokens)

                            else:
                                whole_trainset[(tuple[1])] = [tuple[0], tuple[2], tokens]
                                for token in tokens:
                                    if token not in tale_word_whole_dict[tuple[1]]:
                                        tale_word_whole_dict[tuple[1]][token] = 1
                                    else:
                                        tale_word_whole_dict[tuple[1]][token] += 1
            for stype in trainset:
                print(str(stype) + ": " + str(len(trainset[stype].keys())))

            with open('trainset_word_per_tale_count_' + str(modus)+'.txt','w+')as savefile:
                savefile.write(str(tale_word_dict_trainset))

            with open('trainset_' +str(modus)+'.txt', 'w+') as outfile:
                outfile.write(str(trainset))

        n = 1
        while n <= 3:
            print("Create Testset " +str(n) + ":")
            create_testsets(new_t,n)

            n += 1
        print("Anzahl der Animal Tales annotierten Märchen: " + str(len(tale_word_dict_trainset.values())))
        print("Anzahl aller Märchen: " + str(
            len(tale_word_whole_dict.values()) + len(tale_word_dict_evalset.values()) + len(
                tale_word_dict_trainset.values())))

    def save_list(lines, filename):
        # convert lines to a single blob of text
        data = '\n'.join(lines)
        # open file
        file = open(filename, 'w')
        # write text
        file.write(data)
        # close file
        file.close()

        # save tokens to a vocabulary file
        save_list(vocabular, 'vocabular.txt')  # Gestemmete Tokens

        with open("trainset.txt", "w") as trainout:
            trainout.write(str(trainset.items()))
        with open("evalset.txt", "w") as evalout:
            evalout.write(str(evalset.items()))

def Doc2Vec(trainfile, binary):
    def read(train):
        trainingsdoc = train
        for tokens,id in trainingsdoc:

            yield gensim.models.doc2vec.TaggedDocument(tokens,[id])

    with open(trainfile) as d:

        bl = d.read().replace("defaultdict(<class 'dict'>, ", "(").replace("\\", "")
        f = literal_eval(bl)
        c = 0
        simplified_matrix = []
        only_MT_s_matrix = []
        mapping_dict = dict()
        testtalelist= []
        classcount = defaultdict()
        #if not binary:

        for category,tales in f.items():
            if category not in classcount.keys():
                classcount[category] = 1
            else:
                classcount[category] += 1
            for tale_id,tale in tales.items():
                mapping_dict[int(tale_id)] = category
                simplified_matrix.append((tale[2],tale_id))

        n = (len(simplified_matrix) // 5)

        while n > 1:
            x = random.randrange(0, len(simplified_matrix))
            if binary:
                try:
                    pass
            #        only_MT_s_matrix.pop(x)
                except:
                    pass
            testtalelist.append(simplified_matrix.pop(x))

            n -= 1

        print(len(testtalelist))

        #if binär:
         #   traindata = list(read(only_MT_s_matrix))
        #else:
        traindata = list(read(simplified_matrix))
        model = gensim.models.doc2vec.Doc2Vec(traindata,vector_size=50, min_count=2, epochs=80)

        model.save("d2v.model")
        print("Model Saved")

        for elem in testtalelist:
            inferred_vector = model.infer_vector(elem[0])

            sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))

            cat = int((sims[0])[0])
            c += 1

            try:
               # print(mapping_dict[cat])
                if mapping_dict[elem[1]] in classification_matrix[mapping_dict[cat]].keys():
                    classification_matrix[mapping_dict[cat]][mapping_dict[elem[1]]] += 1
                else:
                    classification_matrix[mapping_dict[cat]][mapping_dict[elem[1]]] = 1
            except KeyError:
                print("Error")


        if not binary:
            return Eval_Multi_Classification(classification_matrix, classcount)
        else:
            return Eval_Binary_Classification(classification_matrix,classcount)

def stemmer(tokens):

    porter = PorterStemmer()
    stemmed = [porter.stem(word) for word in tokens]
    tokens = [word for word in stemmed if len(word) > 1]
    return tokens

def classify(testset, mode,binary):
    with open(testset) as x:

        bla = x.read().replace("defaultdict(<class 'dict'>, ","(").replace("\\","")
        tale_word_dict = literal_eval(bla)
        resultlist = [0, 0, 0, 0]
        r = 0
        test_classify_dict_list = defaultdict(list)
        test_tfidf_dict = defaultdict(defaultdict)
        res = []

        while r < 50:
            test_classify_dict_list.clear()
            test_tfidf_dict.clear()
            test_tale_list = []

            #create Testset of random folktales, 20% of the original set and append to the test_tale_list
            # Ich muss am Ende eine Liste herausgeben, [Accuracy, Precision, Recall, F_score]

            if mode  == "tfidf": # random (key value) pair testset
                target = copy.deepcopy(tale_word_dict)

                for (tale_type,tales) in target.items():
                    n = (len(target[tale_type].keys()) // 5)
                    while n > 1:

                        ran_elem = tales.pop(random.choice(list(tales.keys())))
                        test_tale_list.append((tale_type,ran_elem))
                        n -= 1

            # for tfidf I create 8 documents (each for the respective category) and use the other 20% as queries to categorize
                for supertype in target:
                    for wordc in (target[supertype]).values():
                        for elem in wordc:
                            if elem not in test_tfidf_dict[supertype].keys():
                                test_tfidf_dict[supertype][elem] = 0
                            test_tfidf_dict[supertype][elem] += wordc[elem]

                #print("Length of Testtaleslist:" + str(len(test_tale_list)))
                if binary:
                    res = classifier_tfidf(test_tfidf_dict, test_tale_list, True)
                else:
                    res = classifier_tfidf(test_tfidf_dict, test_tale_list, False)
                classification_matrix.clear()
                target.clear()


            elif mode == "classify":

                for supertype in tale_word_dict:
                    for (id, ta) in (tale_word_dict[supertype]).items():
                        test_classify_dict_list[supertype].append((id,ta[2]))

                for tale_type in test_classify_dict_list:
                    target = test_classify_dict_list[tale_type]
                    n = (len(test_classify_dict_list[tale_type]) // 5)
                    while n > 1:
                        test_tale_list.append((tale_type, target.pop(random.randrange(0, len(target)))))
                        n -= 1
                #print("Length of Testtaleslist:" + str(len(test_tale_list)))
                if binary:
                    res = classifier_intervall(test_classify_dict_list, test_tale_list,True)
                else:
                    res = classifier_intervall(test_classify_dict_list, test_tale_list,False)

            elif mode == "classify_MT":


                for supertype in tale_word_dict:
                    for (id, ta) in (tale_word_dict[supertype]).items():
                        test_classify_dict_list[supertype].append((id,ta[2]))




                for tale_type in test_classify_dict_list:
                    target = test_classify_dict_list[tale_type]
                    n = (len(test_classify_dict_list[tale_type]) // 5)
                    while n > 1:
                        test_tale_list.append((tale_type, target.pop(random.randrange(0, len(target)))))
                        n -= 1
                print("Length of Testtaleslist:" + str(len(test_tale_list)))
                only_MT = defaultdict(dict)

                only_MT["Tales of Magic"] = test_classify_dict_list["Tales of Magic"]

                res = classifier_intervall(only_MT, test_tale_list)


            resultlist[0] += res[0]
            resultlist[1] += res[1]
            resultlist[2] += res[2]
            resultlist[3] += res[3]
            r += 1
        print([resultlist[0]/r,resultlist[1]/r,resultlist[2]/r,resultlist[3]/r])

def classifier_tfidf(taleslist, tales, binary):

    idf_dict = defaultdict(float)
    tf_dict = defaultdict(defaultdict)
    talequery = defaultdict()
    cosinuslist = []
    talestuff = defaultdict()
    classcount = defaultdict(dict)
    classcount.clear()

    for tal in taleslist:
        for (w, va) in taleslist[tal].items():
            idf_dict[w] += 1.0
            tf_dict[tal][w] = (va/len(taleslist[tal].values()))

    for word in idf_dict:
        idf_dict[word] = 1.0 + math.log10(len(taleslist) / idf_dict[word])

    for tale in tales:
        if tale[0] not in classcount.keys():
            classcount[tale[0]] = 0
        else:
            classcount[tale[0]] += 1

        for taletype in tf_dict:

            for words in (tale[1]):
                talestuff[words] = (1+(math.log10(1)))*((tale[1][words])/len((tale[1])))
                try:
                    talequery[words] = tf_dict[taletype][words] * idf_dict[words]
                except KeyError:
                    talequery[words] = 0

            tale_vector = numpy.array(list(talestuff.values()))
            taleslist_vector = numpy.array(list(talequery.values()))
            norm1 = numpy.linalg.norm(tale_vector)
            norm2 = numpy.linalg.norm(taleslist_vector)

            cosinus = (numpy.dot(tale_vector,taleslist_vector)) / (norm1* norm2)
            #print(cosinus)
            cosinuslist.append((taletype,cosinus))
        talestuff.clear()
        talequery.clear()

        cosines = sorted(cosinuslist,key=lambda x: x[1],reverse=True)
        result = cosines[0]


        cosinuslist = []


        if tale[0] in classification_matrix[result[0]].keys():
            classification_matrix[result[0]][tale[0]] += 1
        else:
            classification_matrix[result[0]][tale[0]] = 1

    if binary:
        return  Eval_Binary_Classification(classification_matrix,classcount)

    else:
        return Eval_Multi_Classification(classification_matrix, classcount)

def classifier_intervall(taleslist, testtales, binary):

    mittelwert_dict = defaultdict()
    classification_matrix.clear()

    for (id,tales) in taleslist.items():
        if id in mittelwert_dict.keys():
            mittelwert_dict[id] += len(tales[1])
        else:
            mittelwert_dict[id] = 1

    for mittelwert in mittelwert_dict.keys():
        mittelwert_dict[mittelwert] = len(taleslist[mittelwert]) / mittelwert_dict[mittelwert]

    # Categorization based on simple average score of the Testtale
    testtales_class_count = defaultdict()
    for (klass, tales) in testtales:
        testtales_class_count[klass] = len(tales)
        distancelist = []
        for (id,wert) in mittelwert_dict.items():
            distancelist.append((id,abs(wert-(len(tales[1])))))
        result = (sorted(distancelist, key=lambda x: x[1], reverse=False))[0]

        if klass in classification_matrix[result[0]].keys():
            classification_matrix[result[0]][klass] += 1
        else:
            classification_matrix[result[0]][klass] = 1
    if binary:
        return Eval_Binary_Classification(classification_matrix,testtales_class_count)
    else:
        return Eval_Multi_Classification(classification_matrix,testtales_class_count)

#Class Matrix hast to look like:
#   Predicted Cat   Actual Cat    How many predicted were of this orig cat
# { Animal Tale: { Magical Tale : 50,
#                  Animal Tale : 20},
#   Magical Tale: { Tales of the Stupid Ogre: 5,
#                   Magical Tale:99}
# }
#And classcount has to have all classes with the number of tales per class
def Eval_Multi_Classification(ClassMatrix,class_count):
    overall_matrix = defaultdict(dict)
    accuracycount = [0,0]
    for predicted_class in ClassMatrix.keys():
        tp_fp = 0
        for actual_class in ClassMatrix[predicted_class].keys():
            tp_fp += ClassMatrix[predicted_class][actual_class]

        precision = 0
        try:
            precision = (ClassMatrix[predicted_class][predicted_class]) / tp_fp
            overall_matrix[predicted_class]["Precision"] = precision
            accuracycount[0] += ClassMatrix[predicted_class][predicted_class]
        except KeyError:
           # print("Precision KeyError " + actual_class)
            overall_matrix[predicted_class]["Precision"] = 0

        accuracycount[1] += tp_fp

        try:
            recall_fn_counter = 0
            for pred_class in ClassMatrix.keys():
                if pred_class == predicted_class:
                    pass
                else:
                    try:
                        recall_fn_counter += ClassMatrix[pred_class][predicted_class]
                    except:
                        recall_fn_counter += 0
            try:
                recall = (ClassMatrix[predicted_class][predicted_class]) / (recall_fn_counter + ClassMatrix[predicted_class][predicted_class])
            except ZeroDivisionError:
                recall = 1


        except KeyError:
            recall = 0 # Passiert bei Anecdotes and Jokes
            #print("Recall KeyError " + actual_class)
        overall_matrix[predicted_class]["Recall"] = recall

        try:
            f_score = 2 * (precision*recall) / (precision+recall)
        except ZeroDivisionError: # passiert bei Anecdotes and Jokes
            f_score = 0
            #print("F-Score KeyError " + actual_class)
        overall_matrix[predicted_class]["F-Score"] = f_score

    accuracy = accuracycount[0] / accuracycount[1]
    average_recall = 0
    average_precision = 0
    average_f_score = 0
    for cat in overall_matrix.keys():
        average_f_score += overall_matrix[cat]["F-Score"]
        average_precision += overall_matrix[cat]["Precision"]
        average_recall += overall_matrix[cat]["Recall"]

    average_recall = average_recall / len(overall_matrix.keys())
    average_precision = average_precision / len(overall_matrix.keys())
    average_f_score = average_f_score / len(overall_matrix.keys())

    return (accuracy,average_precision,average_recall,average_f_score)

def Eval_Binary_Classification(ClassMatrix, class_count):

    overall_matrix = defaultdict(dict)
    accuracycount = [0, 0]
    mainclass = "Tales of Magic"
    tp_fp = 0
    precision = 0
    recall_fn_counter = 0

    for predicted_class in ClassMatrix.keys():  # TODO laut theresa darüber schreiben, dass ich nicht genug daten von manchen cats habe und damit dierser Teil durch die anderen Überdeckt wird (mainly MT und AT)

        if predicted_class == mainclass:
            for actual_class in ClassMatrix[predicted_class].keys():
                tp_fp += ClassMatrix[predicted_class][actual_class]




        if predicted_class == mainclass:
            pass
        else:
            for actual_class in ClassMatrix[predicted_class].keys():
                accuracycount[0] += ClassMatrix[predicted_class][actual_class]
                if actual_class == mainclass:
                    pass

                else:
                    accuracycount[1] += ClassMatrix[predicted_class][actual_class]

            try:
                recall_fn_counter += ClassMatrix[predicted_class][mainclass]
            except:
                recall_fn_counter += 0

    try:
        recall = (ClassMatrix[mainclass][mainclass]) / (recall_fn_counter+ ClassMatrix[mainclass][mainclass])

    except KeyError:
        recall = 0  # Passiert bei Anecdotes and Jokes
    overall_matrix[mainclass]["Recall"] = recall
    try:
        precision = (ClassMatrix[mainclass][mainclass]) / tp_fp
        overall_matrix[mainclass]["Precision"] = precision


    except KeyError:
        # print("Precision KeyError " + actual_class)
        overall_matrix[mainclass]["Precision"] = 0
    try:
        f_score = 2 * (precision * recall) / (precision + recall)
    except ZeroDivisionError:  # passiert bei Anecdotes and Jokes
        f_score = 0
        # print("F-Score KeyError " + actual_class)
    overall_matrix[mainclass]["F-Score"] = f_score



    accuracy = (ClassMatrix[mainclass][mainclass] + (accuracycount[1])) / (accuracycount[0]+tp_fp)
    tp_fp = 0
    accuracycount = [0,0]
    average_f_score = overall_matrix[mainclass]["F-Score"]
    average_precision = overall_matrix[mainclass]["Precision"]
    average_recall = overall_matrix[mainclass]["Recall"]


    return (accuracy, average_precision, average_recall, average_f_score)

def averag_doc2vec(evaldatatuplelist):
    leng = len(evaldatatuplelist)
    eval_results = [0,0,0,0]
    for (acc,pre,re,fs) in evaldatatuplelist:
        eval_results[0] += acc
        eval_results[1] += pre
        eval_results[2] += re
        eval_results[3] += fs

    eval_results[0] = eval_results[0] / leng
    eval_results[1] = eval_results[1] / leng
    eval_results[2] = eval_results[2] / leng
    eval_results[3] = eval_results[3] / leng

    return eval_results

def main(modus):

    if modus == "load":
        load_docs()

    elif modus == "doc2vec":
        datatuplelist = []
        print("Start Classification of Set 1:")
        n = 0
        while n <= 50:
            datatuplelist.append(Doc2Vec("trainset_1.txt",False))
            n+=1
        print(averag_doc2vec(datatuplelist))
        classification_matrix.clear()
        datatuplelist = []
        n = 0
        while n <= 50:
            datatuplelist.append(Doc2Vec("trainset_1.txt", True))
            n += 1
        print(averag_doc2vec(datatuplelist))
        classification_matrix.clear()

        print("Start Classification of Set 2:")
        datatuplelist = []
        n = 0
        while n <= 50:
            datatuplelist.append(Doc2Vec("trainset_2.txt",False))
            n += 1
        print(averag_doc2vec(datatuplelist))
        classification_matrix.clear()

        datatuplelist = []
        n = 0
        while n <= 50:
            datatuplelist.append(Doc2Vec("trainset_2.txt", True))
            n += 1
            print(n)
        print(averag_doc2vec(datatuplelist))
        classification_matrix.clear()

        print("Start Classification of Set 3:")
        datatuplelist = []
        n = 0
        while n <= 50:
            datatuplelist.append(Doc2Vec("trainset_3.txt",False))
            n += 1
            print(n)
        print(averag_doc2vec(datatuplelist))
        classification_matrix.clear()

        datatuplelist = []
        n = 0
        while n <= 50:
            datatuplelist.append(Doc2Vec("trainset_3.txt", True))
            n += 1
            print(n)
        print(averag_doc2vec(datatuplelist))
        classification_matrix.clear()

    elif modus == "wordlen":
        print("Start Classification of Set 1 Binary:")
        classify("trainset_1.txt","classify",True)
        classification_matrix.clear()
        print("Start Classification of Set 1 Multi:")
        classify("trainset_1.txt", "classify", False)
        classification_matrix.clear()
        print("")
        print("Start Classification of Set 2 Binary:")
        classify("trainset_2.txt", "classify",True)
        classification_matrix.clear()

        print("Start Classification of Set 2 Multi:")
        classify("trainset_2.txt", "classify", False)
        print("")
        classification_matrix.clear()
        print("Start Classification of Set 3 Binary:")
        classify("trainset_3.txt", "classify",True)
        classification_matrix.clear()


        print("Start Classification of Set 3 Multi:")
        classify("trainset_3.txt", "classify", False)
        classification_matrix.clear()

    elif modus == "tfidf":
        print("Start Classification of Set 1 Binary:")
        classify("trainset_word_per_tale_count_1.txt", "tfidf", True)
        classification_matrix.clear()

        print("Start Classification of Set 1 Multi:")
        classify("trainset_word_per_tale_count_1.txt", "tfidf", False)
        classification_matrix.clear()

        print("Start Classification of Set 2 Binary:")
        classify("trainset_word_per_tale_count_2.txt", "tfidf", True)
        classification_matrix.clear()

        print("Start Classification of Set 2 Multi:")
        classify("trainset_word_per_tale_count_2.txt", "tfidf", False)
        classification_matrix.clear()

        print("Start Classification of Set 3 Binary:")
        classify("trainset_word_per_tale_count_3.txt", "tfidf", True)
        classification_matrix.clear()

        print("Start Classification of Set 3 Multi:")
        classify("trainset_word_per_tale_count_3.txt", "tfidf", False)
        classification_matrix.clear()


    else:
        classification_matrix.clear()
        evaluation.clear()

# To use this program simply open the terminal and run the file with one the following arguments :
# "load" (must be done first, but only once), "doc2vec", "doclen" or "tfidf"
if __name__ == '__main__':

    if (sys.argv[1]) == "load":
        main("load")

    elif (sys.argv[1]) =="doc2vec":
        main("doc2vec")

    elif (sys.argv[1]) == "doclen":
        main("wordlen")

    else:
        main("tfidf")