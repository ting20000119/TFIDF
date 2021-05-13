# Retrieval Program
 Develop a retrieval program that is able to retrieve the relevant news to the given query from a set of 7,034 English News 
 
執行方法：My execute command will always be “python main.py --query <query>”

e.g.：python main.py --query "Trump Biden Taiwan China"

執行後一次會跑出五題的結果：
(1) TF Weighting + Cosine Similarity top 5 results and scores
(2) TF Weighting + Euclidean Distance top 5 results and scores
(3) TF-IDF Weighting + Cosine Similarity top 5 results and scores
(4) TF-IDF Weighting + Euclidean Distance top 5 results and scores
(5) Relevance Feedback top 5 results and scores

我的python版本為：Python 3.8.3

套件模組：
future  0.18.2
nltk   3.5
argparse 1.4.0
scipy   1.5.0

