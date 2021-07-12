# Write your code here
from lxml import etree
from nltk import tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
import string
import pandas as pd

def data(path):
    xml_path = path
    tree = etree.parse(xml_path)
    return tree

def extract(root, header, textlist):
    corpus = root[0]
    for i in corpus:
        header.append(i[0].text)
        textlist.append(i[1].text)

def lemma(text):
    lemmatizer = WordNetLemmatizer()
    for i,word in enumerate(text):
        text[i]=lemmatizer.lemmatize(word)
    return text

def remove(text):
    stp=stopwords.words('english')
    punc=list(string.punctuation)
    rmv=""
    for i,word in enumerate(text):
        if(word not in stp):
            if(word not in punc):
                if(pos_tag([word])[0][1]=="NN"):
                    rmv+=word+" "
    return rmv

def token(text):
    result = tokenize.word_tokenize(text)
    result=lemma(result)
    nouns=remove(result)
    return nouns

def most(nouns,num):
    #return most common
    freq={}
    for word in nouns:
        freq.setdefault(word,0)
        freq[word]+=1
    freq=sorted(freq.items(), key= lambda x:(x[1],x[0]),reverse=True)
    topkey = ""
    for i, key in enumerate(freq):
        if (i < num):
            topkey+=key[0]
            topkey+=" "
        else:
            break
    return topkey
def got(arr,name,top):
    #CONVERT
    matrix=pd.DataFrame(arr)
    matrix.sort_values(by = 0, ascending=True).reset_index()
    df2 = matrix.transpose()
    df2["name"]=name
    #print(df2)
    #SORTING
    df2=df2.sort_values(by=[0,"name"], ascending=[False,False]).reset_index()

    #TOP
    result=""
    for i in range(0, top):
        result += df2.iloc[i]["name"]+" "
    return result

def main():
    #INITIALIZE
    tree = data("news.xml")
    header = []
    text = []
    extract(tree.getroot(), header, text)
    #LOWER -> REDUCE -> NOUNS
    newtext=[]
    for i in text:
        newtext.append(token(i.lower()))
    #TF-IDF
    vectorizer = TfidfVectorizer(input='content', use_idf=True, lowercase=True, analyzer='word')
    weight_matrix = vectorizer.fit_transform(newtext)
    terms = vectorizer.get_feature_names()
    #FIND THE KEYWORDS
    keyword=[]
    top=5
    for i,K in enumerate(newtext):
        keyword.append(got(weight_matrix[i].toarray(),terms,top))
    #PRINT RESULT
    for i,head in enumerate(header):
        print(head + ":")
        print(keyword[i])

if __name__ == "__main__":
    main()
