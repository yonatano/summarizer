#encoding=utf-8
from __future__ import unicode_literals, division

from django.template.loader import get_template
from django.template import Template, Context
from django.http import HttpResponse, HttpResponseRedirect
from django.shortcuts import get_object_or_404, render
from htmltotext import convert
import urllib
from summarize import Summary, mergeShortParagraphSentences, levenshtein, returnStringSubForm_define

from collections import defaultdict

from django.core import serializers

import os
import string

import math 

from stokenizer import SentenceTokenizer


from nltk.tokenize import sent_tokenize, word_tokenize
import nltk, json, operator, urllib2, re, unicodedata

from readability.readability import Document
from bs4 import BeautifulSoup
import cookielib, urllib2

from goose import Goose

from sources_scan.models import Article, Categorized_Labeled_Article, Cluster


from datetime import datetime 
from dateutil.parser import parse

from urlparse import urlparse

import feedfinder

from random import shuffle

#classification
from sklearn.svm import LinearSVC
from nltk.classify.scikitlearn import SklearnClassifier

#more classier imports
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import cPickle as pickle
import marshal
import json

from collections import OrderedDict

def getSimilarWords(request):
	# print "WORLD: %s \n\n\n" % getWords('world')
	# print "BUSINESS: %s \n\n\n" % getWords('business')
	# print "TECHNOLOGY: %s \n\n\n" % getWords('technology')
	# print "ENTERTAINMENT: %s \n\n\n" % getWords('entertainment')
	print "SPORTS: %s \n\n\n" % getWords('sports')
	print "SCIENCE: %s \n\n\n" % getWords('science')
	print "HEALTH: %s \n\n\n" % getWords('health')

	return HttpResponse("done")

def getWords(label):
	articles = Categorized_Labeled_Article.objects.all().filter(label=label)
	words = defaultdict(int)

	for article in articles:
		already_used_words = list()

		article_text = article.text
		tagged_text = returnStringSubForm_define(nltk.pos_tag(nltk.word_tokenize(article_text)))


		for word in word_tokenize(tagged_text):
			if(not word in already_used_words):
				words[('%s' % word)] += 1
				already_used_words.append(word)




	words_sorted = sorted(words, key=words.__getitem__, reverse=True)

	return words_sorted



def list_articles(request):
	articles = list()
	already_clustered = list()

	for a in Article.objects.all().order_by('-pub_date'):
		if(a not in already_clustered):
			
			#source_name = re.findall(ur'(?:(?<=http://www.)|(?<=http://)|(?<=www.)|(?<=\.)|(?<=[a-zA-Z]]))(.*)(?=\.com)', a.url, re.UNICODE)
			scheme, netloc, path, params, query, fragment = urlparse(a.url)
			source_name = netloc

			cluster_list = list()

			if(a.cluster is not None):
				#grab all related articles and blacklist them 
				cluster_list = list(Article.objects.filter(cluster=a.cluster))
				cluster_list.remove(a)
				already_clustered.extend(cluster_list)

			articles.append({'id':a.id, 'title':a.title, 'summary':a.summary, 'image':a.image, 'url':a.url, 'date':a.pub_date, 'favicon':a.favicon, 'source_name':source_name, 'category':a.category, 'cluster_list':cluster_list})
		
			
	return render(request, 'articles.html', {'articles':articles})

def list_articles_json(request):


	req_array = list()

	already_clustered = list()

	callback = request.GET.get('callback', '')

	for a in Article.objects.all().order_by('-pub_date'):

		if(a not in already_clustered):
			
			#source_name = re.findall(ur'(?:(?<=http://www.)|(?<=http://)|(?<=www.)|(?<=\.)|(?<=[a-zA-Z]]))(.*)(?=\.com)', a.url, re.UNICODE)
			scheme, netloc, path, params, query, fragment = urlparse(a.url)
			source_name = netloc

			cluster_list = list()
			serialized_clusters = None

			if(a.cluster is not None):
				#grab all related articles and blacklist them 
				shared_cluster = Article.objects.filter(cluster=a.cluster)
				cluster_list = list(shared_cluster)
				cluster_list.remove(a)
				already_clustered.extend(cluster_list)
				serialized_clusters = serializers.serialize('json', shared_cluster, fields=('title', 'text', 'summary', 'url', 'image', 'favicon', 'pub_date'))
				
				#serialized_list = serializers.serialize('json', shared_cluster, fields=('title', 'text', 'summary', 'url', 'image', 'favicon', 'pub_date'))


			req = {}
			req['cluster_list'] = serialized_clusters
			req['pub_date'] = a.pub_date.strftime("%Y-%m-%d %H:%M:%S")
			req['favicon'] = a.favicon
			req['image'] = a.image
			req['url'] = a.url
			req['summary'] = a.summary
			req['text'] = a.text
			req['title'] = a.title
			req_array.append(req)
	
	response = json.dumps(req_array)
	response = callback + response

	




	return HttpResponse(response, mimetype="application/json")
	

def return_corpus(request):
	string = "%s" % Categorized_Labeled_Article.objects.all().__len__()
	for article in Categorized_Labeled_Article.objects.all():
		string += "<br><br> %s : %s" % (article.url, article.label)


	return HttpResponse(string)

#LOAD CLASSIFIER
file_open = open('my_classifier.pickle')
nb_doc_classifier = pickle.load(file_open)
file_open.close()

def scan_sources(request):
	sources_list = list()

	#add sources
	#sources_list.append("http://www.huffingtonpost.com/")
	#sources_list.append("https://news.google.com/")
	sources_list.append("http://www.techcrunch.com/")
	sources_list.append("http://www.forbes.com/")
	sources_list.append("http://www.foxnews.com/")
	sources_list.append("http://www.ew.com/ew/")
	sources_list.append("http://www.theguardian.com/science/")
	sources_list.append("http://www.cnn.com/")
	sources_list.append("http://www.bbc.co.uk/news/")
	sources_list.append("http://www.nbcnews.com/")
	sources_list.append("http://www.usatoday.com/")
	sources_list.append("http://www.pcworld.com/")
	sources_list.append("http://msn.foxsports.com/")
	sources_list.append("http://guardianlv.com/")
	sources_list.append("http://www.bloomberg.com/")
	sources_list.append("http://www.ledger-enquirer.com/")
	sources_list.append("http://www.cnbc.com/")
	sources_list.append("http://phys.org/")
	sources_list.append("http://www.reuters.com/")
	sources_list.append("http://www.telegraph.co.uk/")
	sources_list.append("http://www.latimes.com/")
	sources_list.append("http://www.scienceworldreport.com/")
	sources_list.append("http://www.washingtonpost.com/")
	sources_list.append("http://newswatch.nationalgeographic.com/")
	sources_list.append("http://www.sportingnews.com/")
	sources_list.append("http://www.cbsnews.com/")

	for source in sources_list:
		scan_source(source)

	return HttpResponse('done')

def scan_source(source_url):

	print "scanning_source: %s" % source_url

	source = source_url
	
	# article = Article.objects.create(title="article_title", text="here is my text", summary="summm",url="http://www.article.com/", favicon="url_to_favicon", image="image_", pub_date="2013-11-20")
	# article.save()


	source_extract_rss_feed = "http://pipes.yahoo.com/pipes/pipe.run?_id=cd6d2d4381f3632fcb44c1755fc9e814&_render=json&urlinput=%s" % urllib.quote_plus(source)
	print source_extract_rss_feed
	data = urllib2.urlopen(source_extract_rss_feed)

	json_data = json.load(data)
	json_value = json_data["value"]

	print "items for source: %s \n" % source
	print json_value['items']

	for item in json_value["items"]:
		try:
			article_url = item['link']

			link = urllib2.urlopen(article_url)
			article_url = link.geturl()

			

			if not [x for x, y in enumerate(Article.objects.all()) if (y.url == article_url)]:
					

				print article_url
				g = Goose()
				article = g.extract(url=article_url)
				#"http://www.sportingnews.com/ncaa-football/story/2013-09-17/week-4-exit-poll-johnny-manziel-alabama-oregon-texas-mack-brown-mariota"


				

				#get summary
				summary_data = urllib2.urlopen("http://localhost:3000/summarize/url/%s" % article_url)
				summary_json_data = json.load(summary_data)
				

				if(not(summary_json_data["summary"] == "failed")):

					title = article.title
					text = article.cleaned_text
					summary = summary_json_data["summary"]
					url = article_url
					#favicon = "http://g.etfv.co/%s" % source
					favicon = "http://www.google.com/s2/favicons?domain=%s" % source
					if(article.top_image.src is not None):
						image = article.top_image.src
					else:
						image = 'http://fin6.com/wp-content/uploads/2013/07/62f0ed6fcdec4833da88ca803969b2db1.jpg'
					category = ""
					pubdate = item['pubDate']
				



					#categorize_article
					#######################################

					#tokenize words
					words = dict()
					for word in word_tokenize(text):
						words.setdefault(('%s' % word), 0)	
						words[('%s' % word)] += 1

					#SET CATEGORY
					category = nb_doc_classifier.classify(words)
					########################################
					
					if not [x for x, y in enumerate(Article.objects.all()) if (y.title == title or y.image == image)]:
						#Save in database
						article = Article.objects.create(title=title, text=text, summary=summary, url=url, favicon=favicon, image=image, category=category, pub_date=parse(pubdate))
						article.save()

					else:
						print "ARTICLE ALREADY EXISTS"
				else:
					print "ARTICLE ALREADY EXISTS"
		except (urllib2.HTTPError, UnicodeDecodeError, AttributeError):
			print "error: %s" % (article_url)




	return HttpResponse("Success")


def clusterArticles(request):
	# for a in Article.objects.all():
	# 	processClustersForArticle(a)
	processClustersForArticle()

	return HttpResponse('Done')


def train_classifier(request):
	featureset_list = list()


	#for every article, create a featureset and add it to the featuresetlist

	for article in Categorized_Labeled_Article.objects.all().exclude(text=""):
		words = dict()

		url = article.url
		article_text = article.text
		#tagged_text = returnStringSubForm_define(nltk.pos_tag(nltk.word_tokenize(article_text)))
		stop_words = {'a', 'about', 'above', 'above', 'across', 'after', 'afterwards', 'again', 'against', 'all', 'almost', 'alone', 'along', 'already', 'also','although','always','am','among', 'amongst', 'amoungst', 'amount',  'an', 'and', 'another', 'any','anyhow','anyone','anything','anyway', 'anywhere', 'are', 'around', 'as',  'at', 'back','be','became', 'because','become','becomes', 'becoming', 'been', 'before', 'beforehand', 'behind', 'being', 'below', 'beside', 'besides', 'between', 'beyond', 'bill', 'both', 'bottom','but', 'by', 'call', 'can', 'cannot', 'cant', 'co', 'con', 'could', 'couldnt', 'cry', 'de', 'describe', 'detail', 'do', 'done', 'down', 'due', 'during', 'each', 'eg', 'eight', 'either', 'eleven','else', 'elsewhere', 'empty', 'enough', 'etc', 'even', 'ever', 'every', 'everyone', 'everything', 'everywhere', 'except', 'few', 'fifteen', 'fify', 'fill', 'find', 'fire', 'first', 'five', 'for', 'former', 'formerly', 'forty', 'found', 'four', 'from', 'front', 'full', 'further', 'get', 'give', 'go', 'had', 'has', 'hasnt', 'have', 'he', 'hence', 'her', 'here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'however', 'hundred', 'ie', 'if', 'in', 'inc', 'indeed', 'interest', 'into', 'is', 'it', 'its', 'itself', 'keep', 'last', 'latter', 'latterly', 'least', 'less', 'ltd', 'made', 'many', 'may', 'me', 'meanwhile', 'might', 'mill', 'mine', 'more', 'moreover', 'most', 'mostly', 'move', 'much', 'must', 'my', 'myself', 'name', 'namely', 'neither', 'never', 'nevertheless', 'next', 'nine', 'no', 'nobody', 'none', 'noone', 'nor', 'not', 'nothing', 'now', 'nowhere', 'of', 'off', 'often', 'on', 'once', 'one', 'only', 'onto', 'or', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'own','part', 'per', 'perhaps', 'please', 'put', 'rather', 're', 'same', 'see', 'seem', 'seemed', 'seeming', 'seems', 'serious', 'several', 'she', 'should', 'show', 'side', 'since', 'sincere', 'six', 'sixty', 'so', 'some', 'somehow', 'someone', 'something', 'sometime', 'sometimes', 'somewhere', 'still', 'such', 'system', 'take', 'ten', 'than', 'that', 'the', 'their', 'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'therefore', 'therein', 'thereupon', 'these', 'they', 'thickv', 'thin', 'third', 'this', 'those', 'though', 'three', 'through', 'throughout', 'thru', 'thus', 'to', 'together', 'too', 'top', 'toward', 'towards', 'twelve', 'twenty', 'two', 'un', 'under', 'until', 'up', 'upon', 'us', 'very', 'via', 'was', 'we', 'well', 'were', 'what', 'whatever', 'when', 'whence', 'whenever', 'where', 'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while', 'whither', 'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'will', 'with', 'within', 'without', 'would', 'yet', 'you', 'your', 'yours', 'yourself', 'yourselves', 'the'}

		for word in word_tokenize(article_text):
			if(not (word in stop_words)):
				words.setdefault(('%s' % word), 0)	
				words[('%s' % word)] += 1

		featureset = (words, article.label)
		featureset_list.append(featureset)		



	#shuffle list
	shuffle(featureset_list)

	#split training data into train and test
	lenovertwo = math.trunc(featureset_list.__len__() * 0.5)
	train_set = featureset_list[lenovertwo:]
	test_set = featureset_list[:lenovertwo]

	#create and train classifier SVM Classifier
	# classif_svm = SklearnClassifier(LinearSVC())
	# classif_svm.train(train_set)

	#naive_bayes
	classif_nb = nltk.NaiveBayesClassifier.train(featureset_list)

	#pipeline
	# pipeline = Pipeline([('tfidf', TfidfTransformer()), ('chi2', SelectKBest(chi2, k=1000)), ('nb', MultinomialNB())])
	# classif_pl = SklearnClassifier(pipeline)
	# classif_pl.train(train_set)

	#get score
	# score_svm = nltk.classify.accuracy(classif_svm, test_set)
	score_nb = nltk.classify.accuracy(classif_nb, test_set)
	# score_pl = nltk.classify.accuracy(classif_pl, test_set)

	# scores = "SVM: %s <br> NB: %s <br> PL: %s <br>" % (score_svm, score_nb, score_pl)


	#save model
	f = open('my_classifier.pickle', 'wb')
	pickle.dump(classif_nb, f, 1)
	f.close()


	return HttpResponse("%s" % score_nb)

# def categorize_tfidf(request, article_url):

# 	#load model
# 	f = open('my_classifier.pickle')
# 	classif = pickle.load(f)
# 	f.close()
# 	print "loaded model"

# 	#categorize incoming article
# 	g = Goose()
# 	article = g.extract(url=article_url)


# 	corpus = Categorized_Labeled_Article.objects.all()
# 	number_of_documents = corpus.__len__()

# 	words = dict()

# 	document_text = article.cleaned_text
# 	document_words = word_tokenize(document_text)

# 	#filter out punctuation
# 	document_words = filter(lambda x: (re.search(ur'^[a-zA-Z]+$', x, re.UNICODE) is not None), document_words) 

# 	number_of_words = document_words.__len__()

# 	print "num_words: %i" % number_of_words

# 	document_words_without_repeats = [x.lower() for x in list(OrderedDict.fromkeys(document_words))]

# 	for word in document_words_without_repeats:
# 		word = word.lower()
# 		term_frequency = document_words.count("%s" % word) / number_of_words
# 		words[('%s' % word)] = term_frequency




# 	#now that we have the tfs, let's find the idfs for all the words:


# 	for word in document_words_without_repeats:
# 		#idf = log(number of documents total / number of documents with word inside)
# 		number_of_documents_with_word = [doc for doc in corpus if ("%s" % word).lower() in ("%s" % doc.text).lower()].__len__()
# 		#print "number_of_documents_with_word= %s: %i" % (word, number_of_documents_with_word)
# 		inverse_doc_frequency = math.log(number_of_documents / (number_of_documents_with_word+1))

# 		#replace the term_frequency score of every word with it's tf-idf: term_frequency * inverse_doc_frequency
# 		words[('%s' % word)] *= inverse_doc_frequency

# 	featureset = words
	

# 	classification = classif.classify(featureset)

# 	output = ""
# 	output += "PREDICTED: %s <br>" % classification
# 	output += "<br><br> %s" % featureset

# 	return HttpResponse(output)


def categorize(request, article_url):
	
	#load model
	f = open('my_classifier.pickle')
	classif = pickle.load(f)
	f.close()
	print "loaded model"

	#categorize incoming article
	g = Goose()
	article = g.extract(url=article_url)

	#get list of words
	words = dict()

	
	article_text = article.cleaned_text
		
	for word in word_tokenize(article_text):
		words.setdefault(('%s' % word), 0)	
		words[('%s' % word)] += 1

	print "got words!"

	


	classified = classif.classify(words)

	output = ""
	output += "PREDICTED: %s <br>" % classified
	output += "<br><br> %s" % article_text

	return HttpResponse(output)

def categorize_tfidf(request, article_url):
	from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
	from sklearn.feature_extraction import DictVectorizer

	#categorize incoming article
	# g = Goose()
	# article = g.extract(url=article_url)

	
	# article_text = article.cleaned_text

	raw_text = []
	raw_labels = []

	documents = Categorized_Labeled_Article.objects.all()

	#split training data into train and test
	

	# for document in documents:
	# 	print "%s %s %d" % (document.label, document.text.split(" ")[1], document.text is None)
	# 	raw_text.append({'text':document.text, 'label':document.label})

	# 	if(document.label == 'world'):
	# 		raw_labels.append(0)
	# 	if(document.label == 'business'):
	# 		raw_labels.append(1)
	# 	if(document.label == 'technology'):
	# 		raw_labels.append(2)
	# 	if(document.label == 'sports'):
	# 		raw_labels.append(3)
	# 	if(document.label == 'science'):
	# 		raw_labels.append(4)
	# 	if(document.label == 'health'):
	# 		raw_labels.append(5)
	# 	if(document.label == 'entertainment'):
	# 		raw_labels.append(6)
		
	print raw_text.__len__()
	print raw_labels.__len__()

	data = [({'first_word-postag': 'Bras\xc3\xadlia PROP',
   'last_word-postag': 'Bras\xc3\xadlia PROP',
   'partial_path': 'vp fcl',
   'path': 'vp\xc2\xa1fcl!np',
   'phrase': u'np',
   'phrase_left-sibling': 'NULL',
   'phrase_parent': u'fcl',
   'phrase_right-sibling': u'np',
   'position': 0,
   'pred_context_left': u'hoje',
   'pred_context_left_postag': u'ADV',
   'pred_context_right': u'um',
   'pred_context_right_postag': u'ART',
   'pred_form': u'revela',
   'pred_lemma': u'revelar',
   'punct_left': 'NULL',
   'punct_right': 'NULL',
   'subcat': 'fcl=np np vp np pu',
   'voice': 0},
  'NULL')]

	samples, labels = zip(*data)

	v = DictVectorizer()
	X = v.fit_transform(samples)
	print X

	#split into training and testing
	lenovertwo = math.trunc(raw_text.__len__() * 0.5)

	train_set_text = raw_text#[lenovertwo:]
	#test_set_text = raw_text[:lenovertwo]

	train_set_labels= raw_labels#[lenovertwo:]
	#test_set_labels = raw_labels[:lenovertwo]


	#stop_words = {'a', 'about', 'above', 'above', 'across', 'after', 'afterwards', 'again', 'against', 'all', 'almost', 'alone', 'along', 'already', 'also','although','always','am','among', 'amongst', 'amoungst', 'amount',  'an', 'and', 'another', 'any','anyhow','anyone','anything','anyway', 'anywhere', 'are', 'around', 'as',  'at', 'back','be','became', 'because','become','becomes', 'becoming', 'been', 'before', 'beforehand', 'behind', 'being', 'below', 'beside', 'besides', 'between', 'beyond', 'bill', 'both', 'bottom','but', 'by', 'call', 'can', 'cannot', 'cant', 'co', 'con', 'could', 'couldnt', 'cry', 'de', 'describe', 'detail', 'do', 'done', 'down', 'due', 'during', 'each', 'eg', 'eight', 'either', 'eleven','else', 'elsewhere', 'empty', 'enough', 'etc', 'even', 'ever', 'every', 'everyone', 'everything', 'everywhere', 'except', 'few', 'fifteen', 'fify', 'fill', 'find', 'fire', 'first', 'five', 'for', 'former', 'formerly', 'forty', 'found', 'four', 'from', 'front', 'full', 'further', 'get', 'give', 'go', 'had', 'has', 'hasnt', 'have', 'he', 'hence', 'her', 'here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'however', 'hundred', 'ie', 'if', 'in', 'inc', 'indeed', 'interest', 'into', 'is', 'it', 'its', 'itself', 'keep', 'last', 'latter', 'latterly', 'least', 'less', 'ltd', 'made', 'many', 'may', 'me', 'meanwhile', 'might', 'mill', 'mine', 'more', 'moreover', 'most', 'mostly', 'move', 'much', 'must', 'my', 'myself', 'name', 'namely', 'neither', 'never', 'nevertheless', 'next', 'nine', 'no', 'nobody', 'none', 'noone', 'nor', 'not', 'nothing', 'now', 'nowhere', 'of', 'off', 'often', 'on', 'once', 'one', 'only', 'onto', 'or', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'own','part', 'per', 'perhaps', 'please', 'put', 'rather', 're', 'same', 'see', 'seem', 'seemed', 'seeming', 'seems', 'serious', 'several', 'she', 'should', 'show', 'side', 'since', 'sincere', 'six', 'sixty', 'so', 'some', 'somehow', 'someone', 'something', 'sometime', 'sometimes', 'somewhere', 'still', 'such', 'system', 'take', 'ten', 'than', 'that', 'the', 'their', 'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'therefore', 'therein', 'thereupon', 'these', 'they', 'thickv', 'thin', 'third', 'this', 'those', 'though', 'three', 'through', 'throughout', 'thru', 'thus', 'to', 'together', 'too', 'top', 'toward', 'towards', 'twelve', 'twenty', 'two', 'un', 'under', 'until', 'up', 'upon', 'us', 'very', 'via', 'was', 'we', 'well', 'were', 'what', 'whatever', 'when', 'whence', 'whenever', 'where', 'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while', 'whither', 'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'will', 'with', 'within', 'without', 'would', 'yet', 'you', 'your', 'yours', 'yourself', 'yourselves', 'the'}

	#vectorizer = TfidfVectorizer(min_df=2, stop_words='english')
	vectorizer2 = DictVectorizer()
	
	#vectorizer = CountVectorizer(stop_words='english')
	#vectorizer = CountVectorizer(ngram_range=(3, 3), stop_words='english')


	x_transformed = vectorizer2.fit_transform(raw_text)
	
	print "-- FINISHED VECTORIZING TRAINING DATA -- "

	# from sklearn import svm
	# clf_svm = svm.SVC()
	# clf_svm.fit(x_transformed.toarray(), raw_labels)  

	# from sklearn.naive_bayes import GaussianNB
	# clf_gnb = GaussianNB()
	# clf_gnb.fit(x_transformed.toarray(), raw_labels)

	from sklearn.naive_bayes import MultinomialNB
	clf_mnb = MultinomialNB()
	clf_mnb.fit(x_transformed.toarray(), raw_labels)

	# from sklearn.neighbors.nearest_centroid import NearestCentroid
	# import numpy as np
	# clf_ne = NearestCentroid()
	# clf_ne.fit(x_transformed.toarray(), raw_labels)

	# from sklearn.ensemble import RandomForestClassifier
	# clf_ens = RandomForestClassifier(n_estimators=10)
	# clf_ens = clf_ens.fit(x_transformed.toarray(), raw_labels)

	# from sklearn.kernel_approximation import RBFSampler
	# from sklearn.linear_model import SGDClassifier
	# clf_ka = SGDClassifier()   
	# clf_ka.fit(x_transformed.toarray(), raw_labels)

	# from sklearn.linear_model import SGDClassifier
	# clf_lm = SGDClassifier(loss="hinge", penalty="l2")
	# clf_lm.fit(x_transformed.toarray(), raw_labels)




	print "-- FINISHED TRAINING CLASSIFIER --"


	#classif_nb = nltk.NaiveBayesClassifier.fit(x_transformed, train_set_labels)
	

	# x_transformed_test = vectorizer.transform(test_set_text)
	# y_labels = test_set_labels


	# score_clf_svm = ('svm', clf_svm.score(x_transformed_test.toarray(), y_labels))
	# score_clf_gnb = ('gnb', clf_gnb.score(x_transformed_test.toarray(), y_labels))
	# score_clf_mnb = ('mnb', clf_mnb.score(x_transformed_test.toarray(), y_labels))
	# score_clf_ne = ('ne', clf_ne.score(x_transformed_test.toarray(), y_labels))
	# score_clf_ens = ('ens', clf_ens.score(x_transformed_test.toarray(), y_labels))
	# score_clf_ka = ('ka', clf_ka.score(x_transformed_test.toarray(), y_labels))
	# score_clf_lm = ('lm', clf_lm.score(x_transformed_test.toarray(), y_labels))

	# report = {score_clf_svm, score_clf_gnb, score_clf_mnb, score_clf_ne, score_clf_ens, score_clf_ka, score_clf_lm}

	classify = clf_mnb.predict(raw_text[0])

	return HttpResponse("%s" % classify)


def compareCosine(a1, a2):
	from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
	from sklearn.metrics.pairwise import cosine_similarity


	
	raw_text = list()
	documents = Article.objects.order_by('id')

	for document in documents:
		raw_text.append(document.text)


	vectorizer = TfidfVectorizer(min_df=2, stop_words='english')

	#vectorize corpus
	vectorized_corpus = vectorizer.fit_transform(raw_text)


	#vectorize inputs one and two
	vectorized_articleObject1 = vectorizer.transform({a1.text})
	vectorized_articleObject2 = vectorizer.transform({a2.text})


	similarity_scores = cosine_similarity(vectorized_articleObject1, vectorized_articleObject2)
	print similarity_scores

def processClustersForArticle():
	from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
	from sklearn.metrics.pairwise import cosine_similarity


	
	raw_text = list()
	documents = Article.objects.order_by('id')

	for document in documents:
		raw_text.append(document.text)


	vectorizer = TfidfVectorizer(min_df=2, stop_words='english')

	#vectorize corpus
	vectorized_corpus = vectorizer.fit_transform(raw_text)

	for articleObject in Article.objects.all():

		#vectorize inputs one and two
		vectorized_articleObject = vectorizer.transform({articleObject.text})


		similarity_scores = cosine_similarity(vectorized_articleObject, vectorized_corpus)
		
		

		filtered_documents = list()
		for index, score in enumerate(similarity_scores[0]):
			if score >= 0.4:
				filtered_documents.append(documents[index])
				print documents[index].title

		

		for doc in filtered_documents:
			#Check if article already belongs to a cluster. If it does, add the input article
			#to the cluster and break
			if doc.cluster is not None:
				articleObject.cluster = doc.cluster
				articleObject.save()
				break
			#If it isn't part of a cluster, create a new cluster for the articleObject and 
			if articleObject.cluster is None:
				#create new cluster
				c = Cluster.objects.create()
				c.save()

				articleObject.cluster = c
				doc.cluster = c

				articleObject.save()
				doc.save()

			else:
				#cluster already exists, add article with ID to cluster
				doc.cluster = articleObject.cluster
				doc.save()
			

	return True

def getAverageTermFrequency(request):
	

	featureset_list = list()

	corpus = Categorized_Labeled_Article.objects.all()
	number_of_documents = corpus.__len__()


	#Loop through every document
	#Loop through every word
	#TF = # occurences of word in document / # words in document
	#to add word into dict: take current result, add it, divide by two = new result (avg)
	for document in corpus:
		words = dict()

		document_text = document.text
		document_words = word_tokenize(document_text)

		#filter out punctuation
		document_words = filter(lambda x: (re.search(ur'^[a-zA-Z]+$', x, re.UNICODE) is not None), document_words) 

		number_of_words = document_words.__len__()

		print "num_words: %i" % number_of_words

		document_words_without_repeats = [x.lower() for x in list(OrderedDict.fromkeys(document_words))]

		for word in document_words_without_repeats:
			word = word.lower()
			term_frequency = document_words.count("%s" % word) / number_of_words
			words[('%s' % word)] = term_frequency




		#now that we have the tfs, let's find the idfs for all the words:


		for word in document_words_without_repeats:
			#idf = log(number of documents total / number of documents with word inside)
			number_of_documents_with_word = [doc for doc in corpus if ("%s" % word).lower() in ("%s" % doc.text).lower()].__len__()
			#print "number_of_documents_with_word= %s: %i" % (word, number_of_documents_with_word)
			inverse_doc_frequency = math.log(number_of_documents / number_of_documents_with_word)

			#replace the term_frequency score of every word with it's tf-idf: term_frequency * inverse_doc_frequency
			words[('%s' % word)] *= inverse_doc_frequency

		featureset = (words, document.label)
		featureset_list.append(featureset)


	#train nb and save

	#naive_bayes
	classif_nb = nltk.NaiveBayesClassifier.train(featureset_list)

	#save model
	f = open('my_classifier.pickle', 'wb')
	pickle.dump(classif_nb, f, 1)
	f.close()

	return HttpResponse("TRAINED ON TF-IDF YEHA")



def scikittfidf(request):
	from sklearn.feature_extraction.text import TfidfVectorizer

	raw_text = list()
	raw_labels = list()

	documents = Categorized_Labeled_Article.objects.all()

	#split training data into train and test
	

	for document in documents:
		raw_text.append(document.text)
		raw_labels.append(document.label)



	lenovertwo = math.trunc(raw_text.__len__() * 0.5)

	train_set_text = raw_text[lenovertwo:]
	test_set_text = raw_text[:lenovertwo]

	train_set_labels= raw_labels[lenovertwo:]
	test_set_labels = raw_labels[:lenovertwo]


	
	vectorizer = TfidfVectorizer()

	x_transformed = vectorizer.fit_transform(train_set_text)

	from sklearn import svm
	clf = svm.SVC()
	clf.fit(x_transformed, train_set_labels)  

	#classif_nb = nltk.NaiveBayesClassifier.fit(x_transformed, train_set_labels)

	x_transformed_acc = vectorizer.fit_transform(test_set_text)
	accuracy = clf.accuracy(x_transformed_acc, test_set_labels)
	
	
	



	return HttpResponse("%s" % accuracy)







def article_categorization(request):
	#world
	source_world = "https://news.google.com/news/section?pz=1&cf=all&topic=w&siidp=35b7c7ee176b70bbb385de5db257e1bfd149&ict=ln"
	scrape_category(source_world, 'world')
	
	#business
	source_business = "https://news.google.com/news/section?pz=1&cf=all&topic=b&siidp=d1a1eaa3f888022796f5e782882603704a93&ict=ln"
	scrape_category(source_business, 'business')

	#tech
	source_tech = "https://news.google.com/news/section?pz=1&cf=all&topic=tc&siidp=ad5d8f4895bdda868ce2be7be0b7d5ca1b6d&ict=ln"
	scrape_category(source_tech, 'technology')

	#science
	source_sci = "https://news.google.com/news/section?pz=1&cf=all&topic=snc&siidp=ad5d8f4895bdda868ce2be7be0b7d5ca1b6d&ict=ln"
	scrape_category(source_sci, 'science')

	#entertainment
	source_ent = "https://news.google.com/news/section?pz=1&cf=all&topic=e&siidp=ad5d8f4895bdda868ce2be7be0b7d5ca1b6d&ict=ln"
	scrape_category(source_ent, 'entertainment')

	#health
	source_health = "https://news.google.com/news/section?pz=1&cf=all&topic=m&siidp=ad5d8f4895bdda868ce2be7be0b7d5ca1b6d&ict=ln"
	scrape_category(source_health, 'health')

	#sports
	source_sports = "https://news.google.com/news/section?pz=1&cf=all&topic=s&siidp=ad5d8f4895bdda868ce2be7be0b7d5ca1b6d&ict=ln"
	scrape_category(source_sports, 'sports')


	count = Categorized_Labeled_Article.objects.all().count()
	print count
	return HttpResponse(count)


def scrape_category(url, c_label):
	extract_feed_world = "http://pipes.yahoo.com/pipes/pipe.run?_id=a625f9823d9b5c4858865b107dcc2516&_render=json&urlinput1=%s" % urllib.quote_plus(url)
	data_world = urllib2.urlopen(extract_feed_world)
	json_data_world = json.load(data_world)

	for item in json_data_world['value']['items']:
		# link = urllib2.urlopen(item['link'])
		# link = link.geturl()
		if not [x for x, y in enumerate(Categorized_Labeled_Article.objects.all()) if (y.url == item['link'])]:
			try:
				cj = cookielib.CookieJar()
				opener = urllib2.build_opener(urllib2.HTTPCookieProcessor(cj))
				request = urllib2.Request(item['link'])
				response = opener.open(request)

				url = response.geturl()
			
				g = Goose()
				article = g.extract(url=url)

				readable_article = article.cleaned_text

				#Save in database
				article = Categorized_Labeled_Article.objects.create(text=readable_article,label=c_label,url=item['link'])
				article.save()
				print article.label

			except (urllib2.HTTPError, UnicodeDecodeError, AttributeError, IOError):
				print "error %s" % item['link']


