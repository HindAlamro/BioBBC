import torch
from flair.data import Sentence
from flair.models import SequenceTagger

from seqeval.metrics import accuracy_score
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score
from seqeval.scheme import IOB2
import pickle





dataset='s800'
class Trained_model():
    def __init__(self, data):
        self.model=SequenceTagger.load(f'models/{dataset}/best-model.pt')
        self.tagger=data
        self.tagger.load_state_dict(self.model.state_dict())
         
            
    def get_model(self):
        return self.tagger


##---Read saved model
with open(f'models/{dataset}/tagger.pickle', 'rb') as pkl:
    tag3 = pickle.load(pkl)
    

##---define a model
model=Trained_model(tag3).get_model()
model.eval()


#----------Read testing file--------------#
import csv
class SentenceFetch(object):
  
    def __init__(self, data):
        self.data = data
        self.sentences = []
        self.tags = []
        self.sent = []
        self.tag = []
        self.poss=[]
        self.pos=[]

        # make tsv file readable
        with open(self.data) as tsv_f:
            reader = csv.reader(tsv_f,delimiter= ' ',quoting=csv.QUOTE_NONE)
            for row in reader:
                #print(row)
                if len(row) == 0:
                    if len(self.sent) != len(self.tag):
                        break
                    self.sentences.append(self.sent)
                    self.tags.append(self.tag)
                    self.poss.append(self.pos)
                    self.sent = []
                    self.tag = []
                    self.pos=[]
                else:
                    self.sent.append(row[0])
                    self.pos.append(row[1]) 
                    self.tag.append(row[2][0])

    def getSentences(self):
        return self.sentences

    def getTags(self):
        return self.tags
    
    def getPos(self):
        return self.poss


df='Datasets/s800/test.txt'
sent = SentenceFetch(df).getSentences()
tag = SentenceFetch(df).getTags()
poss = SentenceFetch(df).getPos()



#-------Predict---------##
sentences = [Sentence(text) for text in sent]
model.predict(sentences)

##------compare--------##
alls=[]
allt=[]

# transfer entity labels to token level
for sentence1 in sentences:
    subs=[]
    subt=[]
    # now go through all tokens and print label
    for token in sentence1:
        subs.append(token.text)
        if (token.has_label('ner')):
            subt.append(token.tag)
        else:
            subt.append('O')
    alls.append(subs)
    allt.append(subt)



print(classification_report(tag,allt,digits=4,  scheme=IOB2))