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


#import feeds
from feeds.models import Feed
#Django Q object
from django.db.models import Q

def getCustomFeed(request, feed_id):
	feed = Feed.objects.filter(id=feed_id)[0]
	tags = json.loads(feed.tags)

	print tags

	feed_articles = Article.objects.filter(reduce(operator.or_, (Q(text__contains=x) for x in tags)))

	response = serializers.serialize('json', feed_articles, fields=('title', 'text', 'summary', 'url', 'image', 'favicon', 'pub_date'))

	return HttpResponse(response)

def getBasicFeedWithCategory(request, category):


	req_array = list()

	already_clustered = list()

	callback = request.GET.get('callback', '')

	for a in Article.objects.filter(category=category).order_by('-pub_date'):

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
			req['category'] = a.category
			req['pub_date'] = a.pub_date.strftime("%Y-%m-%d %H:%M:%S")
			req['favicon'] = a.favicon
			req['image'] = a.image
			req['url'] = a.url
			req['summary'] = a.summary
			req['text'] = a.text
			req['title'] = a.title
			req_array.append(req)
	
	response = json.dumps({'articles':req_array, 'size':len(req_array)})
	response = callback + response

	




	return HttpResponse(response, mimetype="application/json")
	