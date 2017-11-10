
# Topic Modeling with Pyspark

   - **Supported by TingLin**
   - **Kansas State University**

# Table of Contents
  - **Extracting, transforming and selecting features**
    
- **Feature Extractors**
     - [TF-IDF](#TF-IDF)
     - [Word2Vec](#Word2Vec)
     - [CountVectorizer](#CountVectorizer)
- **Feature Transformers**
     - [Tokenizer](#Tokenizer)
     - [StopWordsRemover](#StopWordsRemover)
     - [nn-gram](#nn-gram)
     - [Binarizer](#Binarizer)
     - [PCA](#PCA)
     - [PolynomialExpansion](#PolynomialExpansion)
     - [Discrete Cosine Transform (DCT)](#Discrete Cosine Transform)
     - [StringIndexer](#StringIndexer)
     - [IndexToString](#IndexToString)
     - [OneHotEncoder](#OneHotEncoder)
     - [VectorIndexer](#VectorIndexer)
     - [Interaction](#Interaction)
     - [Normalizer](#Normalizer)
     - [StandardScaler](#StandardScaler)
     - [MinMaxScaler](#MinMaxScaler)
     - [MaxAbsScaler](#MaxAbsScaler)
     - [Bucketizer](#Bucketizer)
     - [ElementwiseProduct](#ElementwiseProduct)
     - [SQLTransformer](#SQLTransformer)
     - [VectorAssembler](#VectorAssembler)
     - [QuantileDiscretizer](#QuantileDiscretizer)
- **Feature Selectors**
     - [VectorSlicer](#VectorSlicer)
     - [RFormula](#RFormula)
     - [ChiSqSelector](#ChiSqSelector)
- **Clustering**
     - [LDA](#LDA)
- **LDA Topic Modeling with csv file**
     - [LDA Topic Modeling with csv file](#LDA Topic Modeling with csv file)
- ** Visualization**
     - [Visualization](#Visualization)


```python
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, CountVectorizer
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.ml.linalg import Vectors, SparseVector
from pyspark.ml.clustering import LDA, BisectingKMeans
from pyspark.sql.functions import monotonically_increasing_id
import re
```


```python
from pyspark.sql import SQLContext, Row
from pyspark.ml.feature import CountVectorizer
from pyspark.mllib.clustering import LDA, LDAModel
from pyspark.mllib.linalg import Vector, Vectors
from pyspark.ml.feature import StopWordsRemover
```


```python
from pyspark.sql import SQLContext
from pyspark import SparkContext
sc = SparkContext.getOrCreate()
sqlContext = SQLContext(sc)

```

# Visualization


```python
# Load Data
rawdata = sqlContext.read.load("data/airlines2.csv", format="csv", header=True)
rawdata = rawdata.fillna({'review': ''})                               # Replace nulls with blank string
rawdata = rawdata.withColumn("uid", monotonically_increasing_id())     # Create Unique ID
rawdata = rawdata.withColumn("year_month", rawdata.date.substr(1,7))   # Generate YYYY-MM variable
 
# Show rawdata (as DataFrame)
rawdata.show(10)
```

    +-----+---------------+---------+--------+------+--------+-----+-----------+--------------------+---+----------+
    |   id|        airline|     date|location|rating|   cabin|value|recommended|              review|uid|year_month|
    +-----+---------------+---------+--------+------+--------+-----+-----------+--------------------+---+----------+
    |10001|Delta Air Lines|21-Jun-14|Thailand|     7| Economy|    4|        YES|Flew Mar 30 NRT t...|  0|   21-Jun-|
    |10002|Delta Air Lines|19-Jun-14|     USA|     0| Economy|    2|         NO|Flight 2463 leavi...|  1|   19-Jun-|
    |10003|Delta Air Lines|18-Jun-14|     USA|     0| Economy|    1|         NO|Delta Website fro...|  2|   18-Jun-|
    |10004|Delta Air Lines|17-Jun-14|     USA|     9|Business|    4|        YES|"I just returned ...|  3|   17-Jun-|
    |10005|Delta Air Lines|17-Jun-14| Ecuador|     7| Economy|    3|        YES|"Round-trip fligh...|  4|   17-Jun-|
    |10006|Delta Air Lines|17-Jun-14|     USA|     9|Business|    5|        YES|Narita - Bangkok ...|  5|   17-Jun-|
    |10007|Delta Air Lines|14-Jun-14|      UK|     0| Economy|    1|         NO|Flight from NY La...|  6|   14-Jun-|
    |10008|Delta Air Lines|14-Jun-14|     USA|     0| Economy|    1|         NO|Originally I had ...|  7|   14-Jun-|
    |10009|Delta Air Lines|13-Jun-14|     USA|     4|Business|    2|         NO|We flew paid busi...|  8|   13-Jun-|
    |10010|Delta Air Lines|13-Jun-14|      UK|     9| Economy|    3|        YES|"I flew from Heat...|  9|   13-Jun-|
    +-----+---------------+---------+--------+------+--------+-----+-----------+--------------------+---+----------+
    only showing top 10 rows
    


- unique id and words would be selected when doing topic modeling


```python
def cleanup_text(record):
    text  = record[8]
    uid   = record[9]
    words = text.split()  
    # Default list of Stopwords
    stopwords_core = ['a', u'about', u'above', u'after', u'again', u'against', u'all', u'am', u'an', u'and', u'any', u'are', u'arent', u'as', u'at', 
    u'be', u'because', u'been', u'before', u'being', u'below', u'between', u'both', u'but', u'by', 
    u'can', 'cant', 'come', u'could', 'couldnt', 
    u'd', u'did', u'didn', u'do', u'does', u'doesnt', u'doing', u'dont', u'down', u'during', 
    u'each', 
    u'few', 'finally', u'for', u'from', u'further', 
    u'had', u'hadnt', u'has', u'hasnt', u'have', u'havent', u'having', u'he', u'her', u'here', u'hers', u'herself', u'him', u'himself', u'his', u'how', 
    u'i', u'if', u'in', u'into', u'is', u'isnt', u'it', u'its', u'itself', 
    u'just', 
    u'll', 
    u'm', u'me', u'might', u'more', u'most', u'must', u'my', u'myself', 
    u'no', u'nor', u'not', u'now', 
    u'o', u'of', u'off', u'on', u'once', u'only', u'or', u'other', u'our', u'ours', u'ourselves', u'out', u'over', u'own', 
    u'r', u're', 
    u's', 'said', u'same', u'she', u'should', u'shouldnt', u'so', u'some', u'such', 
    u't', u'than', u'that', 'thats', u'the', u'their', u'theirs', u'them', u'themselves', u'then', u'there', u'these', u'they', u'this', u'those', u'through', u'to', u'too', 
    u'under', u'until', u'up', 
    u'very', 
    u'was', u'wasnt', u'we', u'were', u'werent', u'what', u'when', u'where', u'which', u'while', u'who', u'whom', u'why', u'will', u'with', u'wont', u'would', 
    u'y', u'you', u'your', u'yours', u'yourself', u'yourselves']
    
    # Custom List of Stopwords - Add your own here
    stopwords_custom = ['']
    stopwords = stopwords_core + stopwords_custom
    stopwords = [word.lower() for word in stopwords]    
    
    text_out = [re.sub('[^a-zA-Z0-9]','',word) for word in words]                                       # Remove special characters
    text_out = [word.lower() for word in text_out if len(word)>2 and word.lower() not in stopwords]     # Remove stopwords and words under X length
    return text_out

udf_cleantext = udf(cleanup_text , ArrayType(StringType()))
clean_text = rawdata.withColumn("words", udf_cleantext(struct([rawdata[x] for x in rawdata.columns])))

# tokenizer = Tokenizer(inputCol="description", outputCol="words")
# wordsData = tokenizer.transform(text)
```

- split review into words and then clean the words, finally add words as a new column on rawdata


```python
# Show first row of clean_text
clean_text.take(1)
```




    [Row(id=u'10001', airline=u'Delta Air Lines', date=u'21-Jun-14', location=u'Thailand', rating=u'7', cabin=u'Economy', value=u'4', recommended=u'YES', review=u'Flew Mar 30 NRT to BKK. All flights were great. Flight was on-time and the in-flight entertainment was great. Apart from the meals - some Thai passengers cannot eat beef so the flight crews tried to ask other passengers who could eat beef and changed the meals around. We feel disappointed with their food services.', uid=0, year_month=u'21-Jun-', words=[u'flew', u'mar', u'nrt', u'bkk', u'flights', u'great', u'flight', u'ontime', u'inflight', u'entertainment', u'great', u'apart', u'meals', u'thai', u'passengers', u'cannot', u'eat', u'beef', u'flight', u'crews', u'tried', u'ask', u'passengers', u'eat', u'beef', u'changed', u'meals', u'around', u'feel', u'disappointed', u'food', u'services'])]




```python
# Term Frequency Vectorization  - Option 2 (CountVectorizer)    : 
vectorizer = CountVectorizer(inputCol="words", outputCol="Features", vocabSize = 1000)
vectorizer = vectorizer.fit(clean_text)
featurizedData = vectorizer.transform(clean_text)

vocablist = vectorizer.vocabulary
vocab_broadcast = sc.broadcast(vocablist)

idf = IDF(inputCol="Features", outputCol="features")
idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData)

```


```python
rescaledData.take(1)
```




    [Row(id=u'10001', airline=u'Delta Air Lines', date=u'21-Jun-14', location=u'Thailand', rating=u'7', cabin=u'Economy', value=u'4', recommended=u'YES', review=u'Flew Mar 30 NRT to BKK. All flights were great. Flight was on-time and the in-flight entertainment was great. Apart from the meals - some Thai passengers cannot eat beef so the flight crews tried to ask other passengers who could eat beef and changed the meals around. We feel disappointed with their food services.', uid=0, year_month=u'21-Jun-', words=[u'flew', u'mar', u'nrt', u'bkk', u'flights', u'great', u'flight', u'ontime', u'inflight', u'entertainment', u'great', u'apart', u'meals', u'thai', u'passengers', u'cannot', u'eat', u'beef', u'flight', u'crews', u'tried', u'ask', u'passengers', u'eat', u'beef', u'changed', u'meals', u'around', u'feel', u'disappointed', u'food', u'services'], features=SparseVector(1000, {0: 0.4099, 3: 1.0601, 11: 1.2624, 25: 1.3913, 32: 3.4155, 46: 1.8131, 56: 4.3116, 97: 2.3469, 113: 2.5063, 201: 2.8577, 213: 2.9304, 249: 6.1442, 332: 3.3172, 346: 3.3454, 369: 3.435, 395: 3.4667, 490: 3.7227, 509: 3.8097, 537: 3.6819, 621: 8.2563, 693: 4.1281, 846: 8.6716}))]



-New column as features is added to the rescaleddata


```python
countVectors = vectorizer.transform(rescaledData).select("uid", "features")
from pyspark.mllib.feature import IDF
frequencyVectors = countVectors.rdd.map(lambda vector: vector[1])
from pyspark.mllib.linalg import Vectors
frequencyDenseVectors = frequencyVectors.map(lambda vector: Vectors.dense(vector))
idf = IDF().fit(frequencyDenseVectors)
tfidf = idf.transform(frequencyDenseVectors)
corpus = tfidf.map(lambda x: [1, x]).cache()
```


```python
countVectors.take(1)
```




    [Row(uid=0, features=SparseVector(1000, {0: 2.0, 3: 1.0, 11: 1.0, 25: 1.0, 32: 2.0, 46: 1.0, 56: 2.0, 97: 1.0, 113: 1.0, 201: 1.0, 213: 1.0, 249: 2.0, 332: 1.0, 346: 1.0, 369: 1.0, 395: 1.0, 490: 1.0, 509: 1.0, 537: 1.0, 621: 2.0, 693: 1.0, 846: 2.0}))]




```python
# find the probability for each vectors
frequencyVectors.take(1)
```




    [SparseVector(1000, {0: 2.0, 3: 1.0, 11: 1.0, 25: 1.0, 32: 2.0, 46: 1.0, 56: 2.0, 97: 1.0, 113: 1.0, 201: 1.0, 213: 1.0, 249: 2.0, 332: 1.0, 346: 1.0, 369: 1.0, 395: 1.0, 490: 1.0, 509: 1.0, 537: 1.0, 621: 2.0, 693: 1.0, 846: 2.0})]




```python
corpus.take(1)
```




    [[1,
      DenseVector([0.4099, 0.0, 0.0, 1.0601, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.2624, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.3913, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.4155, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.8131, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.3116, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.3469, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.5063, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.8577, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.9304, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6.1442, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.3172, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.3454, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.435, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.4667, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.7227, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.8097, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.6819, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8.2563, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.1281, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8.6716, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])]]




```python
ldaModel = LDA.train(corpus, k = 15, maxIterations=100, optimizer="online", docConcentration=2.0, topicConcentration=3.0)
```

- Build Latent Dirichlet Allocation model for clustering
- Note: LDA does not perform well with the EMLDAOptimizer which is used by default. In the case of EMLDAOptimizer we have significant bies to the most popular hashtags. I used the OnlineLDAOptimizer instead. The Optimizer implements the Online variational Bayes LDA algorithm, which processes a subset of the corpus on each iteration, and updates the term-topic distribution adaptively.


```python
topicIndices = ldaModel.describeTopics(maxTermsPerTopic=5)
```

- each topic has maximun 5 terms


```python
vocablist = vectorizer.vocabulary
```

- create vocabulary list


```python
topicsRDD = sc.parallelize(topicIndices)
```


```python
termsRDD.take(5)
```




    [(u'lax', 0.016897424756377493, 0),
     (u'deltas', 0.008325115753216757, 0),
     (u'delta', 0.007506281781445614, 0),
     (u'sydney', 0.005419189092402393, 0),
     (u'777', 0.004502947895167999, 0)]



- each terms and its probability with its topic number


```python
import operator
termsRDD = topicsRDD.map(lambda topic: (zip(operator.itemgetter(*topic[0])(vocablist), topic[1])))
indexedTermsRDD = termsRDD.zipWithIndex()
termsRDD = indexedTermsRDD.flatMap(lambda term: [(t[0], t[1], term[1]) for t in term[0]])
termDF = termsRDD.toDF(['term', 'probability', 'topicId'])
rawJson = termDF.toJSON().collect()

```


```python
from IPython.core.display import display, HTML
from IPython.display import Javascript

s = ""
for line in rawJson:
    s += (str(line) +',')
stringJson = s[:-1]
```

- prepare the data and transform it into JSON format.


```python
html_code = """
<!DOCTYPE html>
<meta charset="utf-8">
<style>

circle {
  fill: rgb(31, 119, 180);
  fill-opacity: 0.5;
  stroke: rgb(31, 119, 180);
  stroke-width: 1px;
}

.leaf circle {
  fill: #ff7f0e;
  fill-opacity: 1;
}

text {
  font: 14px sans-serif;
}

</style>
<body>
<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/3.5.5/d3.min.js"></script>

<script>

var json = {
 "name": "data",
 "children": [
  {
     "name": "topics",
     "children": [
      %s
     ]
    }
   ]
};

var r = 1500,
    format = d3.format(",d"),
    fill = d3.scale.category20c();

var bubble = d3.layout.pack()
    .sort(null)
    .size([r, r])
    .padding(1.5);

var vis = d3.select("body").append("svg")
    .attr("width", r)
    .attr("height", r)
    .attr("class", "bubble");

  
var node = vis.selectAll("g.node")
    .data(bubble.nodes(classes(json))
    .filter(function(d) { return !d.children; }))
    .enter().append("g")
    .attr("class", "node")
    .attr("transform", function(d) { return "translate(" + d.x + "," + d.y + ")"; })
    color = d3.scale.category20();
  
  node.append("title")
      .text(function(d) { return d.className + ": " + format(d.value); });

  node.append("circle")
      .attr("r", function(d) { return d.r; })
      .style("fill", function(d) {return color(d.topicName);});

var text = node.append("text")
    .attr("text-anchor", "middle")
    .attr("dy", ".3em")
    .text(function(d) { return d.className.substring(0, d.r / 3)});
  
  text.append("tspan")
      .attr("dy", "1.2em")
      .attr("x", 0)
      .text(function(d) {return Math.ceil(d.value * 10000) /10000; });

// Returns a flattened hierarchy containing all leaf nodes under the root.
function classes(root) {
  var classes = [];

  function recurse(term, node) {
    if (node.children) node.children.forEach(function(child) { recurse(node.term, child); });
    else classes.push({topicName: node.topicId, className: node.term, value: node.probability});
  }

  recurse(null, root);
  return {children: classes};
}

</script>""" % stringJson
```

- prepare the data and transform it into JSON format


```python
stringJson
```




    '{"term":"lax","probability":0.016897424756377493,"topicId":0},{"term":"deltas","probability":0.008325115753216757,"topicId":0},{"term":"delta","probability":0.007506281781445614,"topicId":0},{"term":"sydney","probability":0.005419189092402393,"topicId":0},{"term":"777","probability":0.004502947895167999,"topicId":0},{"term":"charlotte","probability":0.004254761625426125,"topicId":1},{"term":"due","probability":0.0040716864678084184,"topicId":1},{"term":"flights","probability":0.003907312136767623,"topicId":1},{"term":"policy","probability":0.003791183509959711,"topicId":1},{"term":"united","probability":0.003470584635726847,"topicId":1},{"term":"water","probability":0.011875345997707046,"topicId":2},{"term":"crew","probability":0.0043589782279327425,"topicId":2},{"term":"bottle","probability":0.0041667955655610095,"topicId":2},{"term":"seats","probability":0.003674392688385272,"topicId":2},{"term":"dtw","probability":0.0034379131174090746,"topicId":2},{"term":"gate","probability":0.01263174996166642,"topicId":3},{"term":"minutes","probability":0.011402435672042067,"topicId":3},{"term":"hours","probability":0.011138747301920021,"topicId":3},{"term":"delayed","probability":0.010697938133956699,"topicId":3},{"term":"plane","probability":0.00967783098280554,"topicId":3},{"term":"nov","probability":0.004366374267701612,"topicId":4},{"term":"flights","probability":0.003716774732554013,"topicId":4},{"term":"delta","probability":0.00360125872042232,"topicId":4},{"term":"passengers","probability":0.0032400843154457796,"topicId":4},{"term":"plane","probability":0.003226957072244385,"topicId":4},{"term":"economy","probability":0.014995624839142054,"topicId":5},{"term":"seat","probability":0.013953316626647184,"topicId":5},{"term":"delta","probability":0.011423472098133745,"topicId":5},{"term":"extra","probability":0.0105864636175475,"topicId":5},{"term":"entertainment","probability":0.009050172555658095,"topicId":5},{"term":"service","probability":0.0032978153604987467,"topicId":6},{"term":"crew","probability":0.00321722468864023,"topicId":6},{"term":"time","probability":0.0030083758510464583,"topicId":6},{"term":"london","probability":0.0029751762643115964,"topicId":6},{"term":"delta","probability":0.0028740522422701443,"topicId":6},{"term":"boarding","probability":0.011819586851965574,"topicId":7},{"term":"group","probability":0.011141400227797698,"topicId":7},{"term":"toronto","probability":0.009965280579860248,"topicId":7},{"term":"vegas","probability":0.008542280235787375,"topicId":7},{"term":"southwest","probability":0.008118836212038896,"topicId":7},{"term":"carry","probability":0.009457529191389962,"topicId":8},{"term":"bag","probability":0.007474781623951644,"topicId":8},{"term":"fit","probability":0.006780564805784119,"topicId":8},{"term":"overhead","probability":0.005523947896016676,"topicId":8},{"term":"check","probability":0.004819558410180046,"topicId":8},{"term":"philadelphia","probability":0.016405592447497,"topicId":9},{"term":"atlanta","probability":0.008165252479741482,"topicId":9},{"term":"told","probability":0.007211299960378772,"topicId":9},{"term":"seattle","probability":0.0072031822702206694,"topicId":9},{"term":"asked","probability":0.006799926811648974,"topicId":9},{"term":"class","probability":0.018291153730777148,"topicId":10},{"term":"business","probability":0.014305298111971048,"topicId":10},{"term":"good","probability":0.01178196674252696,"topicId":10},{"term":"excellent","probability":0.009680978228934031,"topicId":10},{"term":"first","probability":0.009397997014770449,"topicId":10},{"term":"min","probability":0.00473933571977625,"topicId":11},{"term":"staff","probability":0.0038121448621697323,"topicId":11},{"term":"flights","probability":0.00354433394032275,"topicId":11},{"term":"pretzels","probability":0.003516747398316574,"topicId":11},{"term":"tampa","probability":0.0032099787082924118,"topicId":11},{"term":"continental","probability":0.00405809610460256,"topicId":12},{"term":"united","probability":0.0040042841343161766,"topicId":12},{"term":"feb","probability":0.003924498734536032,"topicId":12},{"term":"new","probability":0.003898982109672334,"topicId":12},{"term":"newark","probability":0.0036504790901874255,"topicId":12},{"term":"phoenix","probability":0.018327807033881382,"topicId":13},{"term":"united","probability":0.008900354012328835,"topicId":13},{"term":"ewr","probability":0.007006470737443336,"topicId":13},{"term":"phx","probability":0.006926567058586085,"topicId":13},{"term":"hnl","probability":0.0068678270275601815,"topicId":13},{"term":"smile","probability":0.009674018070230888,"topicId":14},{"term":"airlines","probability":0.004377891925788065,"topicId":14},{"term":"clean","probability":0.004338753380715281,"topicId":14},{"term":"attendants","probability":0.0043146095442365825,"topicId":14},{"term":"flights","probability":0.0041115348261970075,"topicId":14}'




```python
# visualize data using D3JS framework
# Display the html
display(HTML(html_code))
```



<!DOCTYPE html>
<meta charset="utf-8">
<style>

circle {
  fill: rgb(31, 119, 180);
  fill-opacity: 0.5;
  stroke: rgb(31, 119, 180);
  stroke-width: 1px;
}

.leaf circle {
  fill: #ff7f0e;
  fill-opacity: 1;
}

text {
  font: 14px sans-serif;
}

</style>
<body>
<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/3.5.5/d3.min.js"></script>

<script>

var json = {
 "name": "data",
 "children": [
  {
     "name": "topics",
     "children": [
      {"term":"lax","probability":0.016897424756377493,"topicId":0},{"term":"deltas","probability":0.008325115753216757,"topicId":0},{"term":"delta","probability":0.007506281781445614,"topicId":0},{"term":"sydney","probability":0.005419189092402393,"topicId":0},{"term":"777","probability":0.004502947895167999,"topicId":0},{"term":"charlotte","probability":0.004254761625426125,"topicId":1},{"term":"due","probability":0.0040716864678084184,"topicId":1},{"term":"flights","probability":0.003907312136767623,"topicId":1},{"term":"policy","probability":0.003791183509959711,"topicId":1},{"term":"united","probability":0.003470584635726847,"topicId":1},{"term":"water","probability":0.011875345997707046,"topicId":2},{"term":"crew","probability":0.0043589782279327425,"topicId":2},{"term":"bottle","probability":0.0041667955655610095,"topicId":2},{"term":"seats","probability":0.003674392688385272,"topicId":2},{"term":"dtw","probability":0.0034379131174090746,"topicId":2},{"term":"gate","probability":0.01263174996166642,"topicId":3},{"term":"minutes","probability":0.011402435672042067,"topicId":3},{"term":"hours","probability":0.011138747301920021,"topicId":3},{"term":"delayed","probability":0.010697938133956699,"topicId":3},{"term":"plane","probability":0.00967783098280554,"topicId":3},{"term":"nov","probability":0.004366374267701612,"topicId":4},{"term":"flights","probability":0.003716774732554013,"topicId":4},{"term":"delta","probability":0.00360125872042232,"topicId":4},{"term":"passengers","probability":0.0032400843154457796,"topicId":4},{"term":"plane","probability":0.003226957072244385,"topicId":4},{"term":"economy","probability":0.014995624839142054,"topicId":5},{"term":"seat","probability":0.013953316626647184,"topicId":5},{"term":"delta","probability":0.011423472098133745,"topicId":5},{"term":"extra","probability":0.0105864636175475,"topicId":5},{"term":"entertainment","probability":0.009050172555658095,"topicId":5},{"term":"service","probability":0.0032978153604987467,"topicId":6},{"term":"crew","probability":0.00321722468864023,"topicId":6},{"term":"time","probability":0.0030083758510464583,"topicId":6},{"term":"london","probability":0.0029751762643115964,"topicId":6},{"term":"delta","probability":0.0028740522422701443,"topicId":6},{"term":"boarding","probability":0.011819586851965574,"topicId":7},{"term":"group","probability":0.011141400227797698,"topicId":7},{"term":"toronto","probability":0.009965280579860248,"topicId":7},{"term":"vegas","probability":0.008542280235787375,"topicId":7},{"term":"southwest","probability":0.008118836212038896,"topicId":7},{"term":"carry","probability":0.009457529191389962,"topicId":8},{"term":"bag","probability":0.007474781623951644,"topicId":8},{"term":"fit","probability":0.006780564805784119,"topicId":8},{"term":"overhead","probability":0.005523947896016676,"topicId":8},{"term":"check","probability":0.004819558410180046,"topicId":8},{"term":"philadelphia","probability":0.016405592447497,"topicId":9},{"term":"atlanta","probability":0.008165252479741482,"topicId":9},{"term":"told","probability":0.007211299960378772,"topicId":9},{"term":"seattle","probability":0.0072031822702206694,"topicId":9},{"term":"asked","probability":0.006799926811648974,"topicId":9},{"term":"class","probability":0.018291153730777148,"topicId":10},{"term":"business","probability":0.014305298111971048,"topicId":10},{"term":"good","probability":0.01178196674252696,"topicId":10},{"term":"excellent","probability":0.009680978228934031,"topicId":10},{"term":"first","probability":0.009397997014770449,"topicId":10},{"term":"min","probability":0.00473933571977625,"topicId":11},{"term":"staff","probability":0.0038121448621697323,"topicId":11},{"term":"flights","probability":0.00354433394032275,"topicId":11},{"term":"pretzels","probability":0.003516747398316574,"topicId":11},{"term":"tampa","probability":0.0032099787082924118,"topicId":11},{"term":"continental","probability":0.00405809610460256,"topicId":12},{"term":"united","probability":0.0040042841343161766,"topicId":12},{"term":"feb","probability":0.003924498734536032,"topicId":12},{"term":"new","probability":0.003898982109672334,"topicId":12},{"term":"newark","probability":0.0036504790901874255,"topicId":12},{"term":"phoenix","probability":0.018327807033881382,"topicId":13},{"term":"united","probability":0.008900354012328835,"topicId":13},{"term":"ewr","probability":0.007006470737443336,"topicId":13},{"term":"phx","probability":0.006926567058586085,"topicId":13},{"term":"hnl","probability":0.0068678270275601815,"topicId":13},{"term":"smile","probability":0.009674018070230888,"topicId":14},{"term":"airlines","probability":0.004377891925788065,"topicId":14},{"term":"clean","probability":0.004338753380715281,"topicId":14},{"term":"attendants","probability":0.0043146095442365825,"topicId":14},{"term":"flights","probability":0.0041115348261970075,"topicId":14}
     ]
    }
   ]
};

var r = 1500,
    format = d3.format(",d"),
    fill = d3.scale.category20c();

var bubble = d3.layout.pack()
    .sort(null)
    .size([r, r])
    .padding(1.5);

var vis = d3.select("body").append("svg")
    .attr("width", r)
    .attr("height", r)
    .attr("class", "bubble");

  
var node = vis.selectAll("g.node")
    .data(bubble.nodes(classes(json))
    .filter(function(d) { return !d.children; }))
    .enter().append("g")
    .attr("class", "node")
    .attr("transform", function(d) { return "translate(" + d.x + "," + d.y + ")"; })
    color = d3.scale.category20();
  
  node.append("title")
      .text(function(d) { return d.className + ": " + format(d.value); });

  node.append("circle")
      .attr("r", function(d) { return d.r; })
      .style("fill", function(d) {return color(d.topicName);});

var text = node.append("text")
    .attr("text-anchor", "middle")
    .attr("dy", ".3em")
    .text(function(d) { return d.className.substring(0, d.r / 3)});
  
  text.append("tspan")
      .attr("dy", "1.2em")
      .attr("x", 0)
      .text(function(d) {return Math.ceil(d.value * 10000) /10000; });

// Returns a flattened hierarchy containing all leaf nodes under the root.
function classes(root) {
  var classes = [];

  function recurse(term, node) {
    if (node.children) node.children.forEach(function(child) { recurse(node.term, child); });
    else classes.push({topicName: node.topicId, className: node.term, value: node.probability});
  }

  recurse(null, root);
  return {children: classes};
}

</script>


- D3 (Data-Driven Documents or D3.js) is a JavaScript library for visualizing data using web standards. D3 helps you bring data to life using SVG, Canvas and HTML. D3 combines powerful visualization and interaction techniques with a data-driven approach to DOM manipulation, giving you the full capabilities of modern browsers and the freedom to design the right visual interface for your data.
- download d3.js, and put it at the same location as this files



```python

```
