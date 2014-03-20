summarizer
==========

summarization app


Description: To pursue my interests in Natural Language Processing, I have been developing an algorithm for text-document summarization, more specifically for news articles. My algorithm uses weighted heuristics including sentence position, location, and other title/document similarity measures to rank sentences in order of importance. As a part of my project, I have been experimenting with classifying documents via Naive Bayes and statistical measures through word-occurrence, and tf-idf—with fairly accurate results. I am working on algorithms for clustering these documents, as well. 

Summarization:
There are four heuristics that determine the importance of sentences:
1. Position
2. Sentence Length
3. # of words in common with text title
4. # of words in common with other sentences

These heuristics are calculated in the following was:

Position: sentences that occur first in the article generally provide a more cumulative description of the article.
in a set S of sentences {t0, t1, t2, t3, t4...}, sentences would be assigned a score of: 1.0 - 0.2 * index, and a score of 0 after sentence 5.

Sentence Length: Longer sentences generally contain more information useful to the reader 
the length score for a sentence t, in an article where T is defined to be the the longest sentence in the article, is: length of t / Length of T
*a list of stopwords is used to filter out words that do not give value, such as "and, but, or, like, etc".


# of words in common with text title: Readers depend on article titles to give them a sense of what the article explains. Sentences with more words (*with stopwords filtered out) that are in common with the title are less likely to be unrelated to the main topic of the article, (such as the viewer's opinion -- usually unrelated).
The calculation for this heuristic is fairly intuitive. The score for a sentence t and title T can be calculated like so:
Number of words in sentence t that exist in T / Number of words in sentence t

# of words in common with other sentences: sentences with more words (*with stopwords filtered out) that appear in other sentences are more likely to be describing
information mentioned in more of the article. 
For a set S of sentences, and a sentence t in that set, this score is calculated by an average of the number of words that exist in sentence t AND every other sentence in set S.

The heuristics are calculated, multiplied by their respective waits and added. The last three weights are weighted more heavily than the first two.
I am experimenting with non-linear combinations of these weights, as this may be a better approach than my current method of adding weights.


Classification:
In order to provide a better experience for the user, articles must be classified into categories. 
My categories are: World, Business, Sports, Entertainment, Science, Technology, and Health
I am implementing NLTK's SciKit Naive Bayes classifier, which vectorizes tf-idf scores for words and compares them to a corpus of vectorized documents, which I scraped from 
Google News.


Clustering:
Because I am scraping news articles from many sources, there are often cases in which there are multiple news articles which describe the same event.
Since the reader is only interested in reading one of them (either the most recent, or the best one), I needed to cluster documents that described the same
or similar topics.

The way I do this is I create a vector for the tf-idf scores of each word in a given document.
A variation of the dot-product (called Cosine Similarity) of two document vectors is used to compute their similarity.
Documents whose similarity is above a threshold are grouped together.
