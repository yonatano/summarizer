# -*- coding: utf-8 -*-
from __future__ import division

from django.template.loader import get_template
from django.template import Template, Context
from django.http import HttpResponse, HttpResponseRedirect
from django.shortcuts import get_object_or_404, render
from htmltotext import convert
import urllib
from summarize import Summary, mergeShortParagraphSentences, levenshtein

import os
import string


from stokenizer import SentenceTokenizer


from nltk.tokenize import sent_tokenize, word_tokenize
import nltk, json, operator, urllib2, re, unicodedata

from readability.readability import Document
from bs4 import BeautifulSoup
import cookielib, urllib2

from goose import Goose


def mainPage(request):


	return render(request, 'homepage.html', {})


def summarize_url(request, link):
	

	# url = "http://techcrunch.com/2013/07/18/never-forget-a-face-with-numbersnap/"
	# scraped_content_home = BeautifulSoup(urllib2.urlopen(url).read())
	# scraped_content_divs = scraped_content_home.find_all("div",recursive=True)

	# data = urllib2.urlopen("http://www.diffbot.com/api/article?token=59489a840a862ca754300d9fcd42295b&url=" + link)

	# #raw_json_data = open(data)
	# json_data = json.load(data)
	# json_text = json_data["text"]
	# json_title = json_data["title"]

	# print json_text
		

	#html = urllib2.urlopen(link).read()
	
	#readable_article = Document(html).summary()
	#readable_title = Document(html).short_title()

	article_sentences = list()

	link = urllib2.urlopen(link)
	link = link.geturl()

	g = Goose()
	article = g.extract(url=link)

	readable_title = article.title
	readable_article = article.cleaned_text



	#if goose fails, let's use readability
	if(len(readable_article) <= 800):
		html = urllib2.urlopen(link).read()
		
		readable_article = Document(html).summary()
		readable_title = Document(html).short_title()

		soup = BeautifulSoup(readable_article)

		for p in soup("p", text=True):
			article_sentences.append("%s" % p)

		my_sentences = mergeShortParagraphSentences(article_sentences)


	else:

		sent_original = sent_tokenize(readable_article)
		article_sentences = sent_original
		my_sentences = mergeShortParagraphSentences(article_sentences)



	



	if(my_sentences.__len__() <= 1):
		callback = request.GET.get('callback', '')
		req = {}
		req['url'] = link
		req['summary'] = "failed"
		req['title'] = readable_title
		
		response = json.dumps(req)
		response = callback + response
		return HttpResponse(response, mimetype="application/json")




	

	if isinstance(readable_title, unicode):
		title = unicodedata.normalize('NFKD', readable_title).encode('ascii','ignore')
	else:
		title = readable_title

	for index, article in enumerate(article_sentences):
		if isinstance(article, unicode):
			article = unicodedata.normalize('NFKD', article).encode('ascii','ignore')
		# article = re.sub('  ', ' ', article)
		# article = re.sub(' +', ' ', article)
		# article = re.sub(u'\n', ' ', article)
		# article = re.sub(' \.', '.', article)
		article = re.sub(title, '', article)
		article = article.strip()
		regex = re.compile(r'([^\(]*)\([^\)]*\) *(.*)')
		article = regex.sub('', article) #remove parenthesis
		article = convert(article)
		# if(article.count('\'') == 1):
		# 	article = re.sub(u'\'', '', article)

		if ("correspondent" in article.lower()):#("click" in article.lower()) or ("tips" in article.lower() and "tricks" in article.lower()) ):
			article_sentences.pop(index)

		# upperC = re.findall(r'[A-Z]{3,10}', article, re.UNICODE)

		# if(upperC is not None and len(tuple(upperC)) > 1):
		# 	article_sentences.pop(index)

		# if(len(article) > 50 and not ("." in article)):
		# 	article_sentences.pop(index)


		# for s in article_sentences:
		# 	if(not(s is article) and levenshtein(s, article) <= 50):
		# 		article_sentences.pop(index)


		




	






	#sentences = sent_tokenize(article_composed)
	#sentences = mergeShortParagraphSentences(article_sentences_final)


	# sentences_list_final = list()

	# for s in sentences:
	# 	for sen in sent_tokenize(s):
	# 		sentences_list_final.append(sen)

	sentences_final_list = list()

	sentences_final_list = my_sentences
	# for s in my_sentences:
	# 	for sen in sent_tokenize(s):
	# 		sentences_final_list.append(sen)


	summ = Summary(sentences_final_list, title, "en")

	summary = ""
	if(summ.setUpScoresAndSentencesDictList()):
		if(summ.filterSentences()):
			if(summ.scoreSentences()):
				summary = summ.createSummary()
	
	callback = request.GET.get('callback', '')
	req = {}
	req['url'] = link
	req['summary'] = summary
	req['title'] = title
	
	response = json.dumps(req)
	response = callback + response





	return HttpResponse(response, mimetype="application/json")
	

	#return render(request, 'summary_document.html', {'summary_response': summary, 'summary_title': title, 'summary_score': original_final_length_ratio})







def summarize_text(request, text):
	
	print text
	print ""
	print ""
	title = ""
	sentences = sent_tokenize(text)

	
	summ = Summary(sentences, title, "en")

	summary = ""
	if(summ.setUpScoresAndSentencesDictList()):
		if(summ.filterSentences()):
			if(summ.scoreSentences()):
				summary = summ.createSummary()
	
	

	callback = request.GET.get('callback', '')
	req = {}
	req['title'] = title
	req['summary'] = summary
	response = json.dumps(req)
	response = callback + response

	
    
	return HttpResponse(response, mimetype="application/json")
	



def getWordList(request):
 	html = urllib2.urlopen('http://www.freevocabulary.com/').read()
	soup = BeautifulSoup(html)
	soup.prettify()

	text = ""
	for p in soup("p",text=True):
		if (" v. " in p or " n. " in p or " adv. " in p or " adj. " in p):
				#first word is word
				#second word is POS
				#third word is definition
				tokenization = word_tokenize(p)

				word = tokenization[0]
				pos = tokenization[1]
				definition = ""

				for d in range(tokenization.__len__() - 3):
					definition += tokenization[d+2] + " "

				text += "INSERT INTO `vocabulary`(`word`, `definition`) VALUES (\"%s\",\"%s\");<br>" % (word, definition)


	return HttpResponse(text)


def getSources(request, link):



	#response_ENGLISH_world = constructUrlBullshit("http://pipes.yahoo.com/pipes/pipe.run?_id=cd6d2d4381f3632fcb44c1755fc9e814&_render=json&urlinput1=%s" % urllib.quote_plus("https://news.google.com/news/section?cf=all&ned=iw_il&topic=n&siidp=ca6f83ff43e06ffd80cd67a388fb7350095e&ict=ln"))
	#response_ENGLISH_us = constructUrlBullshit("http://pipes.yahoo.com/pipes/pipe.run?_id=cd6d2d4381f3632fcb44c1755fc9e814&_render=json&urlinput1=%s" % urllib.quote_plus("http://news.google.com/news/section?pz=1&cf=all&ned=us&topic=n&siidp=846c0f0cc4a8848f20a9fc96c44547fdeffc&ict=ln"))
	#response_ENGLISH_biz = constructUrlBullshit("http://pipes.yahoo.com/pipes/pipe.run?_id=cd6d2d4381f3632fcb44c1755fc9e814&_render=json&urlinput1=%s" % urllib.quote_plus("http://news.google.com/news/section?pz=1&cf=all&ned=us&topic=b&siidp=846c0f0cc4a8848f20a9fc96c44547fdeffc&ict=ln"))
	#response_ENGLISH_tech = constructUrlBullshit("http://pipes.yahoo.com/pipes/pipe.run?_id=cd6d2d4381f3632fcb44c1755fc9e814&_render=json&urlinput1=%s" % urllib.quote_plus("http://news.google.com/news/section?pz=1&cf=all&ned=us&topic=tc&siidp=846c0f0cc4a8848f20a9fc96c44547fdeffc&ict=ln"))
	#response_ENGLISH_entertainement = constructUrlBullshit("http://pipes.yahoo.com/pipes/pipe.run?_id=cd6d2d4381f3632fcb44c1755fc9e814&_render=json&urlinput1=%s" % urllib.quote_plus("http://news.google.com/news/section?pz=1&cf=all&ned=us&topic=e&siidp=846c0f0cc4a8848f20a9fc96c44547fdeffc&ict=ln"))
	#response_ENGLISH_sports = constructUrlBullshit("http://pipes.yahoo.com/pipes/pipe.run?_id=cd6d2d4381f3632fcb44c1755fc9e814&_render=json&urlinput1=%s" % urllib.quote_plus("http://news.google.com/news/section?pz=1&cf=all&ned=us&topic=s&siidp=846c0f0cc4a8848f20a9fc96c44547fdeffc&ict=ln"))
	#response_ENGLISH_science = constructUrlBullshit("http://pipes.yahoo.com/pipes/pipe.run?_id=cd6d2d4381f3632fcb44c1755fc9e814&_render=json&urlinput1=%s" % urllib.quote_plus("http://news.google.com/news/section?pz=1&cf=all&ned=us&topic=snc&siidp=846c0f0cc4a8848f20a9fc96c44547fdeffc&ict=ln"))
	#response_ENGLISH_health = constructUrlBullshit("http://pipes.yahoo.com/pipes/pipe.run?_id=cd6d2d4381f3632fcb44c1755fc9e814&_render=json&urlinput1=%s" % urllib.quote_plus("http://news.google.com/news/section?pz=1&cf=all&ned=us&topic=m&siidp=846c0f0cc4a8848f20a9fc96c44547fdeffc&ict=ln"))
	#response_ENGLISH = """
	#ENGLISH:
	#world: %s <br><br><br>
	#us: %s <br><br><br>
	#biz: %s <br><br><br>
	#tech: %s <br><br><br>
	#entertainment: %s <br><br><br>
	#sports: %s <br><br><br>
	#science: %s <br><br><br>
	#health: %s <br><br><br>
	#<br><br><br><br><br>
#					   """ % (response_ENGLISH_world, response_ENGLISH_us, response_ENGLISH_biz, response_ENGLISH_tech, response_ENGLISH_entertainement, response_ENGLISH_sports, response_ENGLISH_science, response_ENGLISH_health)

#topics: n w b t e s m h
	responses_total = ""
	topics = ['derp','w','n','b','t','e','s','m','h'];
	# top_stories = "https://news.google.com/news?cf=all&ned=au&siidp=9c05e0ca40298ca2de47bff479689ee3e1a0&ict=ln&edchanged=1&authuser=0"
	# response = constructUrlBullshit("http://pipes.yahoo.com/pipes/pipe.run?_id=cd6d2d4381f3632fcb44c1755fc9e814&_render=json&urlinput1=%s" % urllib.quote_plus(link))
	# responses_total += "%s <br><br><br><br><br>" % response 
	for t in topics:
		link = "https://news.google.com/news/section?cf=all&ned=au&topic=" + t + "&siidp=9c05e0ca40298ca2de47bff479689ee3e1a0&ict=ln"
		response = constructUrlBullshit("http://pipes.yahoo.com/pipes/pipe.run?_id=cd6d2d4381f3632fcb44c1755fc9e814&_render=json&urlinput1=%s" % urllib.quote_plus(link))
		responses_total += "%s <br><br><br><br><br>" % response 

	

	


	

	return HttpResponse(responses_total)

	

	

	






