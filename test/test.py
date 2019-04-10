import sys
sys.path.append("../src")

from retrieval_answer import top5results_invidx
from retrieval_answer import top5results_emb

print ("test tfidf")
print ("question:Who is White's daughter")
print (top5results_invidx("Who is White's daughter"))
print ("test embedded")
print ("question:Who is White's daughter")
print (top5results_emb("Who is White's daughter"))
print ("question:which company verify contents of the leaked information")
print (top5results_emb("which company verify contents of the leaked information"))