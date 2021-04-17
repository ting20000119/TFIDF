from __future__ import division, unicode_literals
from pprint import pprint
from Parser import Parser
import util
import math
import sys
import argparse
import os
import time
import datetime
import operator
import nltk
class VectorSpace:
    """ A algebraic model for representing text documents as vectors of identifiers. 
    A document is represented as a vector. Each dimension of the vector corresponds to a 
    separate term. If a term occurs in the document, then the value in the vector is non-zero.
    """

    #Collection of document term vectors
    documentVectors = []

    #Mapping of vector index to keyword
    vectorKeywordIndex=[]

    #Tidies terms
    parser=None

    
    idf =[]
    count = 0 
   
    def __init__(self, documents=[]):
        self.documentVectors=[]
        self.parser = Parser()
        if(len(documents)>0):
            self.build(documents)

    def build(self,documents):
        """ Create the vector space for the passed document strings """
        self.vectorKeywordIndex = self.getVectorKeywordIndex(documents)
        self.idf = [0]*len(self.vectorKeywordIndex)
        self.documentVectors = [self.makeVector(document) for document in documents]
        if self.count ==0:
            for n in range(len(self.vectorKeywordIndex)):
                for doc in range(len(self.documentVectors)):
                    if self.documentVectors[doc][n]>0:
                        self.idf[n]+=1
            self.count+=1  


    def getVectorKeywordIndex(self, documentList):
        """ create the keyword associated to the position of the elements within the document vectors """

        #Mapped documents into a single word string	
        vocabularyString = " ".join(documentList)

        vocabularyList = self.parser.tokenise(vocabularyString)
        #Remove common words which have no search value
        vocabularyList = self.parser.removeStopWords(vocabularyList)
        uniqueVocabularyList = util.removeDuplicates(vocabularyList)

        vectorIndex={}
        offset=0
        #Associate a position with the keywords which maps to the dimension on the vector used to represent this word
        for word in uniqueVocabularyList:
            vectorIndex[word]=offset
            offset+=1
        return vectorIndex  #(keyword:position)


    def makeVector(self, wordString):
        """ @pre: unique(vectorIndex) """

        #Initialise vector with 0's
        vector = [0] * len(self.vectorKeywordIndex)
        wordList = self.parser.tokenise(wordString)
        wordList = self.parser.removeStopWords(wordList)
        for word in wordList:
            vector[self.vectorKeywordIndex[word]] += 1 #Use simple Term Count Model
            
        return vector

    def buildQueryVector(self, termList,r="non"):
        """ convert query string into a term vector """
        if r == "relevance":
            tags = set(['VB','VBD','VBG','VBN','VBP','VBZ','NN','NNS','NNP','NNPS'])
            pos_tags =nltk.pos_tag(termList)
            termList =[word for word,pos in pos_tags if pos in tags] 
            query = self.makeVector(" ".join(termList))
        else:
            query = self.makeVector(" ".join(termList))
        return query
            

    def Sort(self,ratings,flag):
        L = {} #創dictionary存放 
        for num, i in enumerate(ratings):  # == for i in range(len(rating)): L[i] = rating[i] #enumerate 把list加編號 (num : 編號 i:第幾個list取值) 
            L[num]=i #一個一個rating放進diction
        L = sorted(L.items(), key=lambda item: item[1], reverse=flag)
        #https://blog.csdn.net/u014662865/article/details/81807112
        #print(L)
        #前五個
        L = L[:5]
        dictdata = {}
        for l in L:
            dictdata[l[0]] = l[1]
        #print(dictdata)
        return dictdata
    
    def tfidf(self,queryVector,flag):
        for i in range(len(queryVector)):
            if self.idf[i] > 0:
                queryVector[i] = queryVector[i]*math.log10( float( 7034/self.idf[i] ))
        if flag == "cos":
            ratings = [util.cosine(queryVector, documentVector)
                      for documentVector in self.documentVectors]
        elif flag == "el":
            ratings = [util.Euclidean(queryVector, documentVector)
                   for documentVector in self.documentVectors]
        return ratings

    def search1and2(self,searchList,compare,flag): 
        """ search for documents that match based on a list of terms """
        queryVector = self.buildQueryVector(searchList)
        if compare == "cos":
            ratings = [util.cosine(queryVector, documentVector) for documentVector in self.documentVectors]
        elif compare == "el":
            ratings = [util.Euclidean(queryVector, documentVector) for documentVector in self.documentVectors]
        return self.Sort(ratings,flag)
    
    def search3(self,searchList):
        queryVector = self.buildQueryVector(searchList)
        for document in self.documentVectors:
            for i in range(len(document)):
                if self.idf[i] > 0:
                    document[i] = document[i]*math.log10( float( 7034/self.idf[i] ))
        ratings = self.tfidf(queryVector,"cos")
        return self.Sort(ratings,True)
    
    def search4(self,searchList):
        queryVector = self.buildQueryVector(searchList)
        ratings = self.tfidf(queryVector,"el")
        return self.Sort(ratings,False)

    def search5(self,searchList,NewsID):
        #origin query 
        queryVector = self.buildQueryVector(searchList) 
        ratings = self.tfidf(queryVector,"cos")
        idx = 0
        maxvalue =0
        for num in range(len(ratings)):
            if ratings[num] >maxvalue:
                maxvalue = ratings[num]
                idx = num     
        files = open('./EnglishNews/'+ NewsID[idx] + ".txt")
        #print(files)
        #feedback
        feedbackVector = self.buildQueryVector(files,"relevance")
        for i in range(len(feedbackVector)):
            feedbackVector[i] = ((0.5) * feedbackVector[i]) + queryVector[i]
        ratings = self.tfidf(feedbackVector,"cos")
        return self.Sort(ratings,True)
        
       
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", nargs='+', required=True) 
    args = parser.parse_args()
    keyword = args.query[0]    
    if keyword != "":
        EnglishNews = []
        NewsID = {}
        count=0
        for name in os.listdir('./EnglishNews'):
            files = open('./EnglishNews/'+name)          
            NewsID[count]=name.replace(".txt", "")
            count +=1
            
            text = ""
            for l in files:
                text += l
                text += " "
            
            EnglishNews.append(text)

        vectorSpace = VectorSpace(EnglishNews)
        #1
        print("----------------------------------") 
        print("TF Weighting + Cosine Similarity\n")
        print("NewsID  Score")
        print("------  ------")
        start = datetime.datetime.now()
        d1 = vectorSpace.search1and2([str(keyword)],"cos",True)
        d1 = sorted( d1.items(), key=lambda item: item[1], reverse=True)
        for n in d1:
            print(NewsID[int(n[0])], n[1])
        end = datetime.datetime.now()
        print("Data Size: 7034")
        print("\n")
        print("Execution Time：", end - start)
        #2
        print("----------------------------------")
        print("TF Weighting + Euclidean Distance\n")
        print("NewsID  Score")
        print("------  ------")
        start = datetime.datetime.now()
        d2 = vectorSpace.search1and2([str(keyword)],"el",False)
        d2 = sorted(d2.items(), key=lambda item: item[1], reverse=False)
        for n in d2:
            print(NewsID[int(n[0])], n[1])
        end = datetime.datetime.now()
        print("Data Size: 7034")
        print("\n")
        print("Execution Time：", end - start)
        #3
        print("----------------------------------")
        print("TF-IDF Weighting + Cosine Similarity\n")
        print("NewsID  Score")
        print("------  ------")
        start = datetime.datetime.now()
        d3=vectorSpace.search3([str(keyword)])
        d3=sorted(d3.items(), key=lambda item: item[1], reverse=True)
        for n in d3:
            print(NewsID[int(n[0])], n[1])
        end = datetime.datetime.now()
        print("Data Size: 7034")
        print("\n")
        print("Execution Time：", end - start)
        #4
        print("----------------------------------")
        print("TF-IDF Weighting + Euclidean Distance\n")
        print("NewsID  Score")
        print("------  ------")
        start = datetime.datetime.now()
        d4=vectorSpace.search4([str(keyword)])
        d4=sorted(d4.items(), key=lambda item: item[1], reverse=False)
        for n in d4:
            print(NewsID[int(n[0])], n[1])
        end = datetime.datetime.now()
        print("Data Size: 7034")
        print("\n")
        print("Execution Time：", end - start)
        ##############
        #2
        print("----------------------------------")
        print("Relevance Feedback\n")
        print("NewsID  Score")
        print("------  ------")
        start = datetime.datetime.now()
        d5 = vectorSpace.search5([str(keyword)],NewsID)
        d5=sorted(d5.items(), key=lambda item: item[1], reverse=True)
        for n in d5:
            print(NewsID[int(n[0])], n[1])
        end = datetime.datetime.now()
        print("Data Size: 7034")
        print("\n")
        print("Execution Time：", end - start)



