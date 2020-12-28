# word_concreteness_predict


# 參考
[基于词语抽象度的汉语隐喻识别](http://gb.oversea.cnki.net/KCMS/detail/detail.aspx?filename=1015549527.nh&dbcode=CMFD&dbname=CMFDREF)

[word2vec 模型實作](https://medium.com/pyladies-taiwan/%E8%87%AA%E7%84%B6%E8%AA%9E%E8%A8%80%E8%99%95%E7%90%86%E5%85%A5%E9%96%80-word2vec%E5%B0%8F%E5%AF%A6%E4%BD%9C-f8832d9677c8)

# Dataset
### word2vec
搜狗實驗室的新聞文本 [下載連結](http://www.sogou.com/labs/resource/cs.php)
### concreteness predict model
Concreteness ratings for 40 thousand English lemmas [link](http://crr.ugent.be/archives/1330)

上述資料分別經過簡體轉繁體，及英文翻中文

# Traning
### Word2vec model
先將搜狗實驗室的新聞文本透過結巴斷詞，並訓練word2vec模型，詳細說明可以參考[word2vec 模型實作](https://medium.com/pyladies-taiwan/%E8%87%AA%E7%84%B6%E8%AA%9E%E8%A8%80%E8%99%95%E7%90%86%E5%85%A5%E9%96%80-word2vec%E5%B0%8F%E5%AF%A6%E4%BD%9C-f8832d9677c8)
```
fileTrainRead = []
with open("corpus.txt",encoding="utf-8") as fileTrainRaw:
    for line in fileTrainRaw:
        fileTrainRead.append(HanziConv.toTraditional(line)) # 簡轉繁

# 斷詞
fileTrainSeg=[]
for i in range(len(fileTrainRead)):
    fileTrainSeg.append([' '.join(list(jieba.cut(fileTrainRead[i][9:-11],cut_all=False)))])
# 因為會跑很久，檢核是否資料有持續在跑
    if i % 50000 == 0 :
        print(i)

# 將jieba的斷詞產出存檔
fileSegWordDonePath ='corpusSegDone.txt'
with open(fileSegWordDonePath,'wb') as fW:
    for i in range(len(fileTrainSeg)):
        fW.write(fileTrainSeg[i][0].encode('utf-8'))
        fW.write(b'\n')

from gensim.models import Word2Vec
from gensim.test.utils import common_texts, get_tmpfile

# Settings
seed = 666
sg = 0
window_size = 5
vector_size = 200
min_count = 1
workers = 4
epochs = 5
batch_words = 10000

train_data = word2vec.LineSentence('corpusSegDone.txt')
model = word2vec.Word2Vec(
    train_data,
    min_count=min_count,
    size=vector_size,
    workers=workers,
    iter=epochs,
    window=window_size,
    sg=sg,
    seed=seed,
    batch_words=batch_words
)

model.save('word2vec.model')
```

### Filtering concreteness labels 
將抽象詞標註資料的抽象程度中位數及標準差設定門檻，定義為抽象詞或非抽象詞
```
df_label = pd.read_excel("concreteness_labels.xlsx")

#decide if row is concrete or not
def concrete_judge(row):
    if row['Conc.Medium'] >= 4.7 and row["Conc.Stantard deviation"]<=1:
        return 1
    if row['Conc.Medium'] <= 3.0:
        return 0
    return None

df_label["is_concrete"]=df_label.apply(lambda row: concrete_judge(row), axis=1)
df_label=df_label.dropna()
```

### Concreteness model traning
將過濾後的抽象詞轉為word vector並訓練logistic model，詳細可以參考[基于词语抽象度的汉语隐喻识别](http://gb.oversea.cnki.net/KCMS/detail/detail.aspx?filename=1015549527.nh&dbcode=CMFD&dbname=CMFDREF) 第二部分
```
# convert word to word vector
x_df=pd.DataFrame()
y_df=pd.DataFrame()
for index,row in df_label.iterrows():
    if index%300 ==0:
        print(index)
    try:
        word_vector=pd.DataFrame([word2vec_model[row["Word_ch"]]])
        x_df=x_df.append(word_vector)
        y_df=y_df.append(pd.DataFrame({row["is_concrete"]}))
    except KeyError:
        print("word {} not in word2vec_model".format(row["Word_ch"]))
        pass

x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.25, random_state=0)

# all parameters not specified are set to their defaults
logisticRegr = LogisticRegression()

logisticRegr.fit(x_train, np.ravel(y_train))

#save model
with open('logisticRefrModel.pickle', 'wb') as f:
    pickle.dump(logisticRegr, f)
```

### Predict
```
def predict(word):
    y_result=None
    try:
        word_vector=pd.DataFrame([word2vec_model[word]])
        y_result=logisticRegr.predict(word_vector)
    except KeyError:
        print("word '{}' not in dict".format(word))
        return None
    return int(y_result)

predict("蘋果")
predict("測試")
```
output:

1

0


