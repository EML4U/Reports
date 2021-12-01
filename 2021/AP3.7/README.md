# EML4U AP3.7 - Strategien zur Konfliktlösung

In diesem Arbeitspaket wurde untersucht, ob sich **2-dimensionale Einbettungen** und bereits **generierte Erklärungen zur Konfliktlösung** eignen.
Es ging die Annahme voraus, dass sich Einbettungen, die sich geometrisch nah an Erkärungen konträrem Labeling befinden, zur Konflikterkennung verwendet werden können.
Dies konnte **für einen Teil bestätigt** werden (siehe 5-Stern Ergebnisse in Teil 1) (siehe 5-Stern Ergebnisse in Teil 1), teilweise ist dieses Vorgehen nicht erfolgreich (siehe 1-Stern Ergebnisse in Teil 1).

Weiterhin wurde geprüft, ob durchschnittliche Einbettungen auf Texten basieren, die neutral formuliert sind und sich daher für eine Konfliklösung in Form von einfachem Löschen bzw. dem Ausschluss für weitere Trainings eignen. Dies konnte nicht bestätigt werden (siehe Teil 2).
Also vielleicht welche mit Überlappung.

<!--
(AP3.7) Strategien zur Konfliktlosung ¨ (UPB-DICE 2PM) Die Wahl der Strategie muss auf der Basis
geeigneter Interaktion geschehen. Im NLG Bereich bietet es sich an, diese Interaktionen zum Teil auf
naturlicher Sprache zu basieren.
-->

## Konfiguration: Anzahl der zu betrachdenen Dokumente
<!--
## Initiale Konfiguration von Nutzern## Initiale Konfiguration von Nutzern

## Initiale Nutzerinteraktion## Initiale Nutzerinteraktion

Der lmax Parameter wird zur Polygonerkennung auf den 2D-Punktwolken der Einbettungen verwendet.
Innerhalb der verwendeten [Polylidar3D](https://jeremybyu.github.io/polylidar/python_api/polylidar.Polylidar3D.html#polylidar.Polylidar3D.__init__) Bibliothek wird der Parameter als maximale Länge von Polygon-Kantenlängen verwendet.Der lmax Parameter wird zur Polygonerkennung auf den 2D-Punktwolken der Einbettungen verwendet.
Innerhalb der verwendeten [Polylidar3D](https://jeremybyu.github.io/polylidar/python_api/polylidar.Polylidar3D.html#polylidar.Polylidar3D.__init__) Bibliothek wird der Parameter als maximale Länge von Polygon-Kantenlängen verwendet.

lmax (float, optional, default=1.0) – Maximum triangle edge length of a triangle in a planar segment. Filters ‘big’ triangles.
-->




```python
number_of_checks = 10
```

## Laden der Daten


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
import numpy as np

from classes.geometry import Geometry



from shapely.geometry import Point, Polygon

from classes.clustering import Clustering

from collections import defaultdict

#from gensim.utils import simple_preprocess
#from collections import Counter

#import matplotlib.pyplot as plt
#from wordcloud import WordCloud, STOPWORDS

#import pickle
```


```python
dataset_id = 'amazon-movie-reviews-10000'
embeddings = io.load_data_pair(dataset_id, io.DATATYPE_EMBEDDINGS, io.DESCRIPTOR_DOC_TO_VEC, 'dim50-epochs50-umap')
texts = io.load_data_pair(dataset_id, io.DATATYPE_TEXT)
```

    Loaded /home/eml4u/EML4U/data/explanation/data/amazon-movie-reviews-10000/doc2vec.dim50-epochs50-umap.embeddings.pickle
    Loaded /home/eml4u/EML4U/data/explanation/data/amazon-movie-reviews-10000/text.pickle



```python
embeddings_a = np.array(list(embeddings.get_a().values()), dtype=float)
embeddings_b = np.array(list(embeddings.get_b().values()), dtype=float)
```

## Geometrische Operationen


```python
geometry = Geometry()

# Detect polygons and get indexes of points
polylidar_kwargs = dict(lmax=0.275, min_hole_vertices=1000)
polygon_indexes_a = geometry.extract_polygon_indexes(embeddings_a, polylidar_kwargs=polylidar_kwargs)
polygon_indexes_b = geometry.extract_polygon_indexes(embeddings_b, polylidar_kwargs=polylidar_kwargs)

# Create polygon objects and substract from each other
polygon_a = geometry.create_polygon(embeddings_a, polygon_indexes_a[0])
polygon_b = geometry.create_polygon(embeddings_b, polygon_indexes_b[0])
polygon_a_not_b = polygon_a - polygon_b
polygon_b_not_a = polygon_b - polygon_a

# Get distinct indexes of points inside distinct polygons
points_a_not_b = geometry.get_points_in_polygon(embeddings_a, polygon_a_not_b)
points_b_not_a = geometry.get_points_in_polygon(embeddings_b, polygon_b_not_a)
indexes_a_not_b = geometry.get_indexes_of_points_in_polygon(embeddings_a, list(embeddings.get_a().keys()), polygon_a_not_b)
indexes_b_not_a = geometry.get_indexes_of_points_in_polygon(embeddings_b, list(embeddings.get_b().keys()), polygon_b_not_a)
clusters_points_a_not_b = Clustering().kmeans(points_a_not_b, indexes_a_not_b)
clusters_points_b_not_a = Clustering().kmeans(points_b_not_a, indexes_b_not_a)
```

### Teil 1: Reviews mit konträr liegenden Einbettungen


```python
# Get centroids of individual polygons
centroids_a = []
centroids_b = []
for cluster in clusters_points_a_not_b:
    coords = []
    for index in cluster:
        coords.append(embeddings.get_a().get(index))
    centroids_a.append(Polygon(coords).centroid)
for cluster in clusters_points_b_not_a:
    coords = []
    for index in cluster:
        coords.append(embeddings.get_b().get(index))
    centroids_b.append(Polygon(coords).centroid)
```


```python
# Get reviews near to available explanations to opposite rating
distances_a = defaultdict(int)
for item in embeddings.get_a().items():
    min_distance = 0
    for centroid in centroids_b:
        distance = Point(item[1]).distance(centroid)
        if(min_distance == 0 or distance < min_distance):
            distances_a[item[0]] = distance
            
distances_b = defaultdict(int)
for item in embeddings.get_b().items():
    min_distance = 0
    for centroid in centroids_a:
        distance = Point(item[1]).distance(centroid)
        if(min_distance == 0 or distance < min_distance):
            distances_b[item[0]] = distance
```


```python
# Print results
print("1-star with small distance to 5-star explanations")
i = 0
for w in sorted(distances_a, key=distances_a.get, reverse=False):
    i += 1
    if i > number_of_checks:
        break
    else:
        print(w, distances_a[w])
        print(texts.get_a().get(w))

print()
print("5-star with small distance to 1-star explanations")
i = 0
for w in sorted(distances_b, key=distances_b.get, reverse=False):
    i += 1
    if i > number_of_checks:
        break
    else:
        print(w, distances_b[w])
        print(texts.get_b().get(w))
```

    1-star with small distance to 5-star explanations
    82068 0.028468974991068027
    I should have picked up on this from the reviews, and the award from Kids First, but I didn't.  My low rating is for adults, or even older kids.  Other  reviewers liked it, so it might be fine for kids 8  or 9 and under,  but the card tricks are really lame for anyone else.  Be advised that this DVD is FOR KIDS ONLY.
    105218 0.07694256693890572
    I bought this video because I saw how many good reviews it got. Well I put it in and was like "what the heck is this??" Its extremely repetative, and very boring. It just shows the same scenes over and over and over. My kids have no interest in it whatsoever. I wish I had never bought this. If you really want a good childrens video check out "Horatio the lion" volumes 1 and 2. They are just great!! My kids love them so much. He teaches so many things.
    45690 0.09208701554407508
    har har this movie suuuux! me whar the frenchman sais that this was the biggest load of bum poo i have ever seen im my life. CROISSANT! it was very smelly and i yelled very much at it for its dumbness. In my countreee we have ze great JERRY LOUVIS HOR HOR HOR HOR le borscht! If this movie had a wood worm named PICO sing once r twice and the story was ab&Uuml;t searching the world fora FREAKING FAIRY i would think better but you AMERICHANZ think that boys with simple and egg plant shaped heads are fuuun to watch then i say get your boxers up the foxes nose n do a lil dance for me HAR HAR HAR HAR HAR PHHHT AHHH!
    62667 0.0972837879709833
    I love movies like The Philadelphia Story, or His Girl Friday, or Stage door I even love Guess Who's Coming to Dinner, they all have fantastic dialog!  As far as musicals I found Seven Brides for Seven Brothers, Meet me in St Louis and even White Christmas  ++ completely charming. I own most of these and re-watch them.<br /><br />  With that being said I just saw this movie Guys and Dolls at an outdoor venue and the dialogue was dreadful, the dance scenes were so cheesy, the story line was so unbelievably boring and the characters mostly unlikable.  I found myself praying that this would be over soon so I could get my ride home so I could go to sleep.  No coolness factor, not sweet lovable charm, slow on the scenes and unlikable characters.. This movie just grated on my nerves.  What more can I say but don't buy this.  I know I'm going to be a lone reviewer but, I had to say my truth.
    110558 0.12736388099457416
    not sure why, but this movie just did not hold my child's attention.  he enjoys baby mozart etc., baby songs, and usually sits still, enthralled by a movie when i put one on.  this one just did not hold his attention.
    94322 0.1930632139422304
    Strange, scary, and not musical at all.  The images are scary - grandmas flying through the air - baby cribs under water surrounded by giant fish looking like they'll eat the babies.  Music/songs are completely un-tuneful and instantly forgetful.  Images have little if anything to do with the songs.  I want my money back on this one.
    105506 0.20401358801069797
    Don't waste your money and time on this garbage.  If you want the Stooges go for the volume collections that are currently out.  You will be receiving value for your buck and the opportunity to relive or perhaps meet the genius of this comedy group.  For us baby-boomers it brings back fond memories of the silver screen and watching these get replayed on the little screen in our home.
    98848 0.21176094411799934
    I'm not sure if this movie could get anymore boring but if you like to watch grass grow or paint dry then you'll enjoy this movie.  They talk sooooo quietly that I had my TV turned all the way up and even then I could barely hear them speaking. I'm sure they were trying to pull off something "Artsy" but good grief this was bad!  Way too many unneeded scenes and clips that were just thrown in with long drawn out music.
    17178 0.22072580372426234
    I don't really mind the cheap animation. It's more the cliched, near offensive parts. If you want your kids to watch movies with the phrase "junk in the trunk" in it then watch this. For my kids, I'll skip it. It's not as crude as more modern "kids" movies but there is still plenty for me not to like.<br /><br />Yes, I am happily prudish with what I want my kids watching.
    15491 0.23424984293214182
    Watered-down, dumbed-down, sanitized music. I know that's Baby E's m.o. for the most part but come on. Not only is the music bad, the choice of images is also boring. I will not subject my toddler to this again. Yikes.
    
    5-star with small distance to 1-star explanations
    300 0.037588030398303184
    I found the movie to be compelling interesting and totally unlike any other movie ive seen, that is the reason I read the book, I didnt like the book I found it to be very very dry, its stupid to say that you dont like the movie, what did you expect. If your a fan of a book its your job not to like the movie, Thats why I anticipate that both Ender's Game and the Hitchhikers Guide to the galaxy are both going to be terrible. Moral to the story Anticipate the worse that way you wont be disapointed.
    4977 0.05356741208870148
    This is one of my favorite films, and it isn't the horror part of it.  My wife hates horror films, but I had to get her to watch it.  She really enjoyed it.  The story and characters have been described before so I won't.  But what I had to talk about is that this movie doesn't descend into the depths of humanity. It goes for the goodness of people.  Sounds a little strange, what I'm trying to say is that it isn't like the 'Night of the Living Dead' with the man hiding his family in the cellar and fighting off the other people.  The people here are neighbors and they act like it.  They try to help each other, even if it cost them their lives.  The people are real, with faults and all, but there is no greedy or cowardly idiot like almost every horror film has.  It was one of the first After Dark Horrorfest movies and it's why I try to pick them up every year.  Every year there is a lot of crap, but there are also a couple of gems like this, such as 'Lake Mungo' or 'The Abandoned'.
    9437 0.09581567176400405
    I am absolutely NOT a fan of the horror genre - on more than one Halloween outing with friends I have run, sobbing, out of a haunted house; my brother gets a kick out of jumping out from behind things to hear me scream; and at night, I still jump into bed from 3 feet away...just in case something's under there. Yet Saw has made it onto the list of my all-time favorite films.<br /><br />Saw's psychological "damned-if-you-do, damned-if-you-don't" concept intrigued me, so I rented it on DVD one night. I shut and locked my bedroom door, turned on every light in the room, and barricaded myself with blankets to see if the product lived up to the hype...and it did.<br /><br />On one hand, I was terrified (afterward, I watched TV for a while to ward off the nightmares)...but I was awed by the film's brilliance. Solid acting, creative cinematography, and the most disturbing and mind-blowing plot twists imaginable.<br /><br />I was so impressed, I saw each sequel on its opening day...from the seat in the rear corner of the theater. No one can sneak up on you from there.
    5475 0.10039337611824661
    Very hilarious but disgusting in the same time :P<br />Anna Faris is great at acting and she can really change her whole reality character!<br /><br />  About the movie, if you are in the mood of laughing then what r u waiting for?! Just watch it and have fun!
    3456 0.1452419676799336
    I just can't believe all of the negative reviews here on this film that go on and on saying that it just isn't possible that something like this could ever happen!<br /><br />Listen to me now:  Star Wars.  Star Trek.  Wolfman.  Dracula.  Spiderman.  Waterworld.  Terminator.  Narnia.  And on and on and on.<br /><br />Get it?  How can you trash a movie like this for being so unbelievable when you can just ADORE  films like I have written above and a thousand other just like them.<br /><br />One word:  Liberalism.<br /><br />Nuff' said.
    2172 0.16243894485910662
    I don't give it 5 stars .. i give it freaking 10. I absolutely love this movie and it shocked the hell out of me. I though it was going to be a regular vampire killing vampires movie and you know, nothing much but blood and fights which seemed fine by me. But then I watched the movie and of course it has blood and fights (oh so good!) but there is also the complexity of "D" and the other characters .. they're all so interesting in their own way, that just made me care about the plot within the movie much more. THE ENDING ROCKS !!! I abso freaking lutely loved the ending ... just watch it and you'll know what I mean .. dammit i wanted to hug himm lol
    15090 0.1728330521801776
    Imagine Leah Adler Spielberg telling her son, who is working on the movie "Duel," that it's time to put away the camera and go to bed. "Steven, I'm telling you for the last time..." Actually, when filming Duel, Steven Spielberg was closer to 25 than 12, the age at which Emily Hagins directed her first film, "Pathogen."<br /><br />"Zombie Girl: The Movie" is an award-winning documentary about a tween who loved movies--so much so that she wanted to direct her own. With her mother, Megan, handling the boom microphone (a mike duct-taped onto the end of a paint roller extension) and helping with transportation and other important film-making chores, such as shopping for and making props, and applying stage makeup to the actors, Hagins managed to make her movie. Dad Jerry appears as a researcher in the film and had a few film-making tasks for which he was responsible. Hagins even received a $1000 grant, which was of more interest to Mom, who had been financing, than to Emily.<br /><br />"Zombie Girl: The Movie" features interviews with the Hagins family, as well as the cast and crew of "Pathogen," and Emily's mentors. Her big break was director Peter Jackson (Lord of the Rings) writing to a friend in Austin (where the Hagins live), telling him to assist the girl.<br /><br />In between scenes of production, we see family members carving pumpkins, cooking and making music together. Megan and Emily have a few disagreements, most over "artistic vision," and Megan must walk the tightrope stretched between responsible mom and film tech.<br /><br />Emily Hagins wrote "Pathogen" when she was ten years old. When filming began, she was twelve. She and her cast scheduled filming around homework, school holidays and events, and family activities. The bulk of filming occurred on weekends and during school vacations. Inevitably, shooting fell behind and the project took much longer to complete than was expected.<br /><br />"Zombie Girl: The Movie follows" Emily through early stages of film-making to opening night; the Alamo Drafthouse, where the film was screened, sold out. In a funny scene the director greeted her audience and introduced her film with aplomb, making it clear that she didn't really want to talk about continuity.<br /><br />Throughout "Zombie Girl: The Movie", adults involved in film-making and criticism discuss the technological changes that have allowed teenagers to become filmmakers, including their positive and negative aspects. Also included on the DVD are a number of extras, including interviews and the entire feature-length film, "Pathogen" (this is the first film I've seen where the "making of" was the feature and the film was a "bonus"). For info on all of Emily's movies, visit cheesynuggets.com.<br /><br />Bottom Line: Would I buy/rent/stream "Zombie Girl: The Movie?" Buy! There's a certain teenage Chloë I need to send a copy to--she wants to be a filmmaker, too. (Zombie Girl: The Movie on DVD hits the streets November 9.)
    14427 0.21278641068630788
    Really good movie! It's been awhile since I've read the book, but it seemed to follow the story well, and Kristen Stewart did a fantastic job with her role of not speaking but communicating through her actions instead. I highly recommend this movie! A lot of people only know Kristen Stewart through the Twilight movies, but don't pay attention to her other movies, which is sad. She's a really good actress! If you like Indie movies like I do, then watch this movie!
    4759 0.2210386599355529
    Before watching any Russell Crowe movie, I thought of him as a pretentious jerk.  After watching this movie, (my first RC movie), I fell in love.  So much so that I had to rent  Cinderella Man and Gladiator.  An exceptional actor who took the story of John Nash and made us understand the reality of Schizophrenia.  You would be crazy not to watch this.
    3425 0.22160676743800203
    Laugh? I thought I'd die. This classic slice of Reagan-era paranoia pits a bunch of heroic, 2nd-Amendment lovin' schoolkids against the Red Army. Makes Rambo: First Blood Part II look like Saving Private Ryan.


## Teil 2: Reviews mit 1 und 5-Stern Bewetungen, deren 2-dimensionale Einbettungen durchschnittlich sind


```python
intersection = polygon_a.intersection(polygon_b)
centroid = intersection.centroid

display(polygon_a)
display(polygon_b)
display(intersection)
```


    
![svg](output_14_0.svg)
    



    
![svg](output_14_1.svg)
    



    
![svg](output_14_2.svg)
    



```python
distances_mid_a = defaultdict(int)
for item in embeddings.get_a().items():
    distances_mid_a[item[0]] = Point(item[1]).distance(centroid)

distances_mid_b = defaultdict(int)
for item in embeddings.get_b().items():
    distances_mid_b[item[0]] = Point(item[1]).distance(centroid)
```


```python
# Print results
print("1-star with small distance to overall centroid")
i = 0
for w in sorted(distances_mid_a, key=distances_mid_a.get, reverse=False):
    i += 1
    if i > number_of_checks:
        break
    else:
        print(w, distances_a[w])
        print(texts.get_a().get(w))
        
print()
print("5-star with small distance to overall centroid")
i = 0
for w in sorted(distances_mid_b, key=distances_mid_b.get, reverse=False):
    i += 1
    if i > number_of_checks:
        break
    else:
        print(w, distances_mid_b[w])
        print(texts.get_b().get(w))
```

    1-star with small distance to overall centroid
    81299 2.0895845402958164
    This is a lousy remake of a very fanous French movie. Do not waste your time with this but instead get the original movie: Le Pere Noel est une Ordure. You will have real fun
    24715 2.1359719070689263
    This was one of the most boring movies I ever saw!  It makes the word love truly just a "four letter word".<br /><br />I even brought it to work and let my coworkers borrow it to get their opinion.  Only one out of six said it was worth watching.  I gave the movie to them!
    39889 2.0695883996319924
    I am so sorry that I bought this film. I found it at a swap meet for $2.00. This was the worst $2.00 that I've ever spent. It looks like they just picked up a camera and started saying lines and flashing gang signs.
    51309 2.1562495334227356
    this movie was dull.boring.stupid.
    80630 2.1061669166554555
    Please, if you loved this movie, have your head examined.
    50342 1.995945177206548
    This is just too slow! If you want something energetic, pass this one. I started to fall asleep...
    69548 2.116468948561548
    The film was really badly done , it was so boring ! I love traveloque films , but this film must have been done 30 years ago.<br />The area is beautiful , and someone with talent could have made the film interesting .I wish that the BBC had done the film , or even a French Production .<br />Plus the guy who commended on the films couldn't even pronounce the names of the towns correctly<br />Waste of time and money!Try and find a good book area of the area .
    68287 1.9799321637448881
    I was really looking forward to this, based on other reviews.  Once again, I learn that bad reviews are usually the correct ones. How easily duped people are....they believe what they read, if not what they've seen, and the cancer spreads.<br /><br />A thin gruel  of Victorian repression trying to achieve psychological Gothic horror....absolutely vacuous performances by the gals;  a red-headed kid who looks yanked out of a Disney fishin' hole tale; prop dept. headmistress, governesses, and police; more empty metaphors than a Harry Chapin song; a 70s softcore porn soundtrack of light classics mixed with pseudo-classical processional ditties; laughable devices of "suspense" and "terror"; an overall air of "if you aren't getting this,we're just too subtle and nuanced for ya"<br /><br />Started this one at about 9:30 PM, which is usually a safe hour for me....nonetheless, I dozed off at least twice, and when the thing was over, I barely trudged upstairs to bed, where I promptly conked out.
    81298 2.102197623313796
    The movie had no plot and was just plain dumb and stupid.
    67021 2.149074094299796
    I want the time in my life that Michal Moore took from me.  Truth about Terrorism - This film isn't truth about anything.
    
    5-star with small distance to overall centroid
    12123 0.02414639808519978
    As a dad of a little girl, this is a wonderful movie.  While based on a Grimms fairy tale, this story does not end up with anyone trying to kill anyone else.  (Both Sleeping Beauty and Snow White do talk about killing the heroine, Snow White is graphic in how the Queen wants Snow White killed).<br /><br />The picture and sound quality of this DVD are fantastic.  Disney has either cleaned and remastered especially well, or kept the originals very pristine over the past 50 years.
    13535 0.029337421445666492
    If you into Disney movies this is a good addition to the collection.
    7728 0.03300470152422183
    If you have been to Italy or you just can't afford the trip watch this blu-ray its brilliant. It covers the whole of the Italian Peninsula with high definition camera work and an excellent commentary. After watching this you will feel like you have travelled the length and breadth of the country. Well worth the cost for an arm chair travelogue as good as this. <a href="http://www.amazon.com/gp/product/B004AN0CQA">Magnificent Italia [Blu-ray</a>]
    16873 0.03581117396727206
    it's the best serie I've ever seen. I would die for getting all seasons on DVD.
    12802 0.03713556606052562
    I enjoy the Dan Brown books and love Tom Hanks in his role in them.  I now find my self anxiously waiting for the next Dan Brown/Tom Hanks/Ron Howard movie.  What a great trio!
    1308 0.03728792425253132
    i love this movie in the teather, but in blue ray is more detailed and with an excelent sound you will centanly enjoy
    9983 0.04901165401409326
    John Wayne and Maureen O'hara at their funniest.  This western, is full of great scenes.  This is one of my fall back favorites (when I am not sure what I want to watch I always put in this classic).
    9930 0.06405886103484798
    Once again we are entertained by the best of the best. The Duke, John Wayne, in one of the older movies. But still full of adventure, fun, laughter, action, and Americanism. The Old West is brought to life for us by the last great American Cowboy. The Duke! You can tell they do a good job by the beautiful scenery, grand music, and of course, the best of actors. Prepair to stay on the edge of your seat. Another great John Wayne western. Enjoy.
    3070 0.06420617545184687
    Rubi has looks men admire and women envy. Her ambitions are high, but dangerous. She will use her beauty and cunning desire to get whatever she wants. There is of course Alejandro who is the love of her life, but even he is pushed aside as she aims for her dreams of grandeur. Will she ever come to her sense or keep pushing until she falls over?<br /><br />Barbara Mori excels as Rubi and really brings her to life. She makes you hate her and want her at once. The supporting cast too is excellent, but Rubi is the star just like she would like it. One of the better telenovelas recently made, you will be hooked by the first episode.<br /><br />Great stuff!
    14927 0.06593398653731453
    Add this to Young Mr. Lincoln, The Return of Frank James, and Jesse James as young Henry Fonda classics.  Another actor you'll see in all of them is Eddie Collins. In Drums Along The Mohawk, he plays Christian, the rolly polly roll caller.  Just another funny Collins performance for comic relief.


Diese Arbeit wurde vom Bundesministerium für Bildung und Forschung (BMBF) im Rahmen des Projekts [EML4U](https://dice-research.org/EML4U) unter der Kennziffer 01IS19080B gefördert.
