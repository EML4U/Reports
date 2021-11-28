# EML4U AP3.4 - Adaptation von Einbettungen bei Anderungen im Eingaberaum

In diesem Arbeitspaket wurde untersucht, wie eine **Neuberechnung von Einbettungen durch inkrementelle Updates vermieden** werden kann.

Im hier vorgestellten Beispiel wurden die Daten aus dem AP 2.7 Beispiel wiederverwendet.
Zunächst erfolgte das Training eines initialen Modells mit 50% der verfügbaren Texte.
Für anschließende **inkrementelle Updates des Modells** wurden nur diejenigen Daten verwendet, die in AP 2.7 keine Überschneidung der Einbettungen zwischen 1-Stern und 5-Stern-Datensätzen ergaben.
Dies entspricht einer Datenbereinigung.
Abschließend wurden die Methoden aus AP 2.5 und AP 2.7 auf dem neuen Modell angewendet.
Die resultierenden Erklärungen für 1-Stern Daten entsprechen den in AP 2.7 berechneten Erklärungen.
Neu berechnete Erklärungen für 5-Stern-Daten weisen durch die eingeschränkten Eingabedaten und entsprechende Polygonverschiebungen Unterschiede auf.

<!--
(AP3.4) Adaptation von Einbettungen bei Anderungen im Eingaberaum ¨ (UPB-DICE 2PM) Sofern Daten bereinigt werden, kann dieses Auswirkungen auf die niedrigdimensionale Einbettung als
auch davon abgeleitete NLG-Erklarungskomponenten haben. Es wird untersucht, wie in diesem Fall ¨
durch inkrementelle Updates eine Neuberechnung der Einbettung und Komponenten vermieden werden kann.
-->


```python
# Reload modules every time before executing the Python code typed
%load_ext autoreload
%autoreload 2

# Import from parent directory
import sys; sys.path.insert(0, '..')

# Configure data storage
from yaml import safe_load
import classes.io
io = classes.io.Io(safe_load(open('../config.yaml', 'r'))['DATA_DIRECTORY'])

# Additional imports
import pickle
import classes.doc_to_vec

import classes.reduction

import numpy as np

from classes.geometry import Geometry
from classes.clustering import Clustering

from gensim.utils import simple_preprocess
from collections import Counter

import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

import numpy as np
```

## Inkrementelles Trainieren des Modells


```python
# Load model with 2x 5,000 entries
dataset_id = 'amazon-movie-reviews-5000'
vector_size=50
epochs=50
details = 'dim' + str(vector_size) + '-epochs' + str(epochs)
model_path = io.get_path(dataset_id, io.DATATYPE_EMBEDDINGS, io.DESCRIPTOR_DOC_TO_VEC, details, io.FILETYPE_EXTENSION_MODEL)
doc2vec = classes.doc_to_vec.DocToVec()
doc2vec.load_model(model_path)
```


```python
# Load indexes from wp 2.7 (documents with clear drift)
with open('/tmp/indexes_a_not_b.pickle', 'rb') as handle:
    indexes_a_not_b = pickle.load(handle)
with open('/tmp/indexes_b_not_a.pickle', 'rb') as handle:
    indexes_b_not_a = pickle.load(handle)

# Load texts
dataset_id = 'amazon-movie-reviews-10000'
texts = io.load_data_pair(dataset_id, io.DATATYPE_TEXT)

# Get texts for indexes
texts_a_not_b = {}
for text_tuple in texts.get_a().items():
    if(text_tuple[0] in indexes_a_not_b):
        texts_a_not_b[text_tuple[0]] = text_tuple[1]
print(len(texts_a_not_b))
texts_b_not_a = {}
for text_tuple in texts.get_b().items():
    if(text_tuple[0] in indexes_b_not_a):
        texts_b_not_a[text_tuple[0]] = text_tuple[1]
print(len(texts_b_not_a))

# Train model
texts_list = list(texts_a_not_b.values()) + list(texts_b_not_a.values())
tagged_docs = [doc2vec.get_tagged_document(doc, i) for i, doc in enumerate(texts_list)]
doc2vec.get_model().train(documents=tagged_docs, total_examples=len(tagged_docs), epochs=epochs)
```

    Loaded /home/eml4u/EML4U/data/explanation/data/amazon-movie-reviews-10000/text.pickle
    2398
    1696


## Einbetten der Texte


```python
dataset_id = 'amazon-movie-reviews-10000'
datatype_id = io.DATATYPE_EMBEDDINGS
descriptor = io.DESCRIPTOR_DOC_TO_VEC
details = 'dim' + str(vector_size) + '-epochs' + str(epochs) + '-increment'

if not io.exists(dataset_id, datatype_id, descriptor, details):
    embeddings = classes.data_pair.DataPair()
    
    # Test with less input
    #for element_id, txt in texts_a_not_b.items():
    #    embeddings.add_a(element_id, doc2vec.embedd(txt))
    #for element_id, txt in texts_b_not_a.items():
    #    embeddings.add_b(element_id, doc2vec.embedd(txt))
    #
    #texts_5000 = io.load_data_pair(dataset_id, io.DATATYPE_TEXT)
    #for element_id, txt in texts_5000.get_a().items():
    #    embeddings.add_a(element_id, doc2vec.embedd(txt))
    #for element_id, txt in texts_5000.get_b().items():
    #    embeddings.add_b(element_id, doc2vec.embedd(txt))
    
    for element_id, txt in texts.get_a().items():
        embeddings.add_a(element_id, doc2vec.embedd(txt))
    for element_id, txt in texts.get_b().items():
        embeddings.add_b(element_id, doc2vec.embedd(txt))

    embeddings.set_runtime();
    io.save_pickle(embeddings, dataset_id, datatype_id, descriptor, details)
else:
    embeddings = io.load_data_pair(dataset_id, datatype_id, descriptor, details)
```

    Wrote: /home/eml4u/EML4U/data/explanation/data/amazon-movie-reviews-10000/doc2vec.dim50-epochs50-increment.embeddings.pickle



```python
def print_info(embeddings):
    print('Items in A and B:', embeddings.get_sizes())
    print('Dimensions:', len(embeddings.get_a_dict_as_lists()[1][0]))
    print('Meta:')
    for key, value in embeddings.get_meta().items():
        print(' ' + key + ':', value)

print_info(embeddings)
```

    Items in A and B: (10000, 10000)
    Dimensions: 50
    Meta:
     creation-date: 2021-11-28 23:45:07
     creation-seconds: 230.860632



```python
# Reduce dimensions
dimension_reduction = classes.reduction.Reduction()

def create(embeddings, method):
    data_pair = classes.data_pair.DataPair()
    
    keys_a, values_a = embeddings.get_a_dict_as_lists()
    keys_b, values_b = embeddings.get_b_dict_as_lists()
    
    pca_a_values, pca_b_values = method(values_a, values_b)
    
    for i, key in enumerate(keys_a):
        data_pair.add_a(key, pca_a_values[i])
    for i, key in enumerate(keys_b):
        data_pair.add_b(key, pca_b_values[i])
        
    data_pair.set_runtime();
    return data_pair

def create_umap(embeddings):
    return create(embeddings, dimension_reduction.umap)

umap = create_umap(embeddings)
```

    UMAP seconds: 24.08634826913476


## Ausführen der Schritte aus AP 2.5


```python
embeddings_a = np.array(list(umap.get_a().values()), dtype=float)
embeddings_b = np.array(list(umap.get_b().values()), dtype=float)
```


```python
geometry = Geometry()

# Detect polygons and get indexes of points
#polylidar_kwargs = dict(lmax=0.275, min_hole_vertices=1000, norm_thresh_min=0.7)
polylidar_kwargs = dict(lmax=0.275, min_hole_vertices=1000)
polygon_indexes_a = geometry.extract_polygon_indexes(embeddings_a, polylidar_kwargs=polylidar_kwargs)
polygon_indexes_b = geometry.extract_polygon_indexes(embeddings_b, polylidar_kwargs=polylidar_kwargs)

# Create polygon objects and substract from each other
polygon_a = geometry.create_polygon(embeddings_a, polygon_indexes_a[0])
polygon_b = geometry.create_polygon(embeddings_b, polygon_indexes_b[0])
polygon_a_not_b = polygon_a - polygon_b
polygon_b_not_a = polygon_b - polygon_a
```


```python
# Get points in distinct parts
points_a_not_b = geometry.get_points_in_polygon(embeddings_a, polygon_a_not_b)
points_b_not_a = geometry.get_points_in_polygon(embeddings_b, polygon_b_not_a)
indexes_a_not_b = geometry.get_indexes_of_points_in_polygon(embeddings_a, list(embeddings.get_a().keys()), polygon_a_not_b)
indexes_b_not_a = geometry.get_indexes_of_points_in_polygon(embeddings_b, list(embeddings.get_b().keys()), polygon_b_not_a)

# Get distinct polygons
clusters_points_a_not_b = Clustering().kmeans(points_a_not_b, indexes_a_not_b)
clusters_points_b_not_a = Clustering().kmeans(points_b_not_a, indexes_b_not_a)
```

## Ausführen der Schritte aus AP 2.7


```python
def get_tokens(indexes, texts):
    tokens = []
    for index in indexes:
        tokens += simple_preprocess(texts[index], deacc=False, min_len=2, max_len=15)

    stopwords = set(STOPWORDS)
    stopwords.add('br')
    
    tokens = [w for w in tokens if w not in stopwords]
    return Counter(tokens)
```


```python
# Get number of points in polygons
print('1-star docs', len(clusters_points_a_not_b[0]), len(clusters_points_a_not_b[1]))
print('5-star docs', len(clusters_points_b_not_a[0]), len(clusters_points_b_not_a[1]))
print()

tokens_a_not_b = get_tokens(indexes_a_not_b, texts.get_a())
tokens_b_not_a = get_tokens(indexes_b_not_a, texts.get_b())
print('1-star words', len(tokens_a_not_b))
print('5-star words', len(tokens_b_not_a))
print()

counter_a_not_b = Counter(tokens_a_not_b)
counter_b_not_a = Counter(tokens_b_not_a)
print('1-star common words', counter_a_not_b.most_common(20))
print('5-star common words', counter_b_not_a.most_common(20))
```

    1-star docs 1996 225
    5-star docs 596 390
    
    1-star words 18523
    5-star words 13268
    
    1-star common words [('movie', 5566), ('film', 2280), ('one', 2108), ('even', 1434), ('bad', 1372), ('time', 1102), ('movies', 1074), ('good', 1053), ('don', 1050), ('see', 960), ('really', 905), ('make', 875), ('people', 854), ('will', 801), ('first', 775), ('made', 745), ('story', 743), ('acting', 727), ('know', 703), ('watch', 666)]
    5-star common words [('film', 1318), ('movie', 1310), ('michael', 978), ('one', 923), ('see', 784), ('will', 632), ('time', 629), ('great', 562), ('samurai', 471), ('love', 459), ('dvd', 440), ('jackson', 435), ('best', 414), ('life', 414), ('well', 406), ('much', 391), ('really', 389), ('even', 386), ('us', 362), ('man', 361)]



```python
def filter_tokens(counter_x, counter_y, factor=5):
    d = {}
    for token in counter_x.keys():
        if(token in counter_y):
            if(counter_x[token] >= counter_y[token] * factor):
                d[token] = counter_x[token]
    return Counter(d)

tokens_1star_0 = get_tokens(clusters_points_a_not_b[0], texts.get_a()) 
tokens_1star_0_filtered = filter_tokens(Counter(tokens_1star_0), counter_b_not_a)

tokens_1star_1 = get_tokens(clusters_points_a_not_b[1], texts.get_a()) 
tokens_1star_1_filtered = filter_tokens(Counter(tokens_1star_1), counter_b_not_a, 1.5)

tokens_5star_0 = get_tokens(clusters_points_b_not_a[0], texts.get_b()) 
tokens_5star_0_filtered = filter_tokens(Counter(tokens_5star_0), counter_a_not_b, 1.5)

tokens_5star_1 = get_tokens(clusters_points_b_not_a[1], texts.get_b()) 
tokens_5star_1_filtered = filter_tokens(Counter(tokens_5star_1), counter_a_not_b)
```


```python
print('1-star words O', len(tokens_1star_0), len(tokens_1star_0_filtered))
print('1-star words 1', len(tokens_1star_1), len(tokens_1star_1_filtered))
print('5-star words 0', len(tokens_5star_0), len(tokens_5star_0_filtered))
print('5-star words 1', len(tokens_5star_1), len(tokens_5star_1_filtered))
```

    1-star words O 17588 1077
    1-star words 1 4299 436
    5-star words 0 6433 432
    5-star words 1 10673 172



```python
print('1-star common words 0', tokens_1star_0.most_common(10))
print('1-star common words 0 filtered', tokens_1star_0_filtered.most_common(10))
print()

print('1-star common words 1', tokens_1star_1.most_common(10))
print('1-star common words 1 filtered', tokens_1star_1_filtered.most_common(10))
print()

print('5-star common words 0', tokens_5star_0.most_common(10))
print('5-star common words 0 filtered', tokens_5star_0_filtered.most_common(10))
print()

print('5-star common words 1', tokens_5star_1.most_common(10))
print('5-star common words 1 filtered', tokens_5star_1_filtered.most_common(10))
```

    1-star common words 0 [('movie', 5274), ('film', 2176), ('one', 1920), ('bad', 1331), ('even', 1330), ('time', 1019), ('movies', 1001), ('good', 985), ('don', 965), ('see', 888)]
    1-star common words 0 filtered [('bad', 1331), ('movies', 1001), ('acting', 722), ('plot', 631), ('worst', 548), ('money', 513), ('nothing', 488), ('horror', 446), ('waste', 423), ('minutes', 406)]
    
    1-star common words 1 [('dvd', 392), ('movie', 292), ('one', 188), ('will', 135), ('amazon', 114), ('disc', 107), ('film', 104), ('even', 104), ('ray', 96), ('blu', 94)]
    1-star common words 1 filtered [('amazon', 114), ('disc', 107), ('ray', 96), ('blu', 94), ('copy', 86), ('product', 79), ('player', 76), ('edition', 72), ('digital', 67), ('release', 59)]
    
    5-star common words 0 [('michael', 963), ('see', 532), ('movie', 464), ('jackson', 424), ('will', 403), ('one', 378), ('dvd', 360), ('film', 307), ('potty', 295), ('great', 291)]
    5-star common words 0 filtered [('michael', 963), ('jackson', 424), ('potty', 295), ('mj', 288), ('music', 283), ('concert', 246), ('song', 167), ('amazing', 145), ('songs', 144), ('loved', 138)]
    
    5-star common words 1 [('film', 1011), ('movie', 846), ('one', 545), ('samurai', 468), ('time', 365), ('story', 328), ('nash', 316), ('life', 278), ('great', 271), ('well', 269)]
    5-star common words 1 filtered [('samurai', 468), ('nash', 316), ('cruise', 228), ('japan', 163), ('japanese', 161), ('sister', 160), ('george', 127), ('bond', 105), ('culture', 84), ('howard', 84)]



```python
def print_wordcould(counts):
    font_path='/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf' #  fc-list | grep 'NotoSans-Bold'
    wordcloud = WordCloud(background_color="white", font_path=font_path, colormap='Dark2', width=1200, height=800).generate_from_frequencies(counts)
    plt.imshow(wordcloud)
    plt.axis("off")
```


```python
print_wordcould(tokens_1star_0_filtered)
```


    
![png](output_20_0.png)
    



```python
print_wordcould(tokens_1star_1_filtered)
```


    
![png](output_21_0.png)
    



```python
print_wordcould(tokens_5star_0_filtered)
```


    
![png](output_22_0.png)
    



```python
print_wordcould(tokens_5star_1_filtered)
```


    
![png](output_23_0.png)
    


Diese Arbeit wurde vom Bundesministerium für Bildung und Forschung (BMBF) im Rahmen des Projekts [EML4U](https://dice-research.org/EML4U) unter der Kennziffer 01IS19080B gefördert.
