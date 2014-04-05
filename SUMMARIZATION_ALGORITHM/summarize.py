# -*- coding: utf-8 -*-
from __future__ import division

import os
import string


from stokenizer import SentenceTokenizer


from nltk.tokenize import sent_tokenize, word_tokenize
import nltk, json, operator, urllib2, re, unicodedata

from readability.readability import Document
from BeautifulSoup import BeautifulSoup

from htmltotext import convert




class Summary(object):

	def __init__(self, sentences, title, lang):

		self.summary_sentences = sentences
		self.summary_title = title
		self.summary_sentences_tagged = tagSentences(sentences)
		self.summary_title_tagged = nltk.pos_tag(nltk.word_tokenize(title))
		self.summary_scoresAndSentencesDictList = list()
		self.summary_language = lang
		self.length_original = 0

		for s in sentences:
			self.length_original += len(s)

	#Methods specific to this class
	def getTitle(self):
		return self.summary_title

	def getSentences(self):
		return self.summary_sentences

	def getSentencesTagged(self):
		return self.summary_sentences_tagged

	def getScoresAndSentences(self):
		return self.summary_scoresAndSentencesDictList

	def setUpScoresAndSentencesDictList(self):
		print self.summary_sentences_tagged.__len__()
		print self.summary_sentences.__len__()

		for index, sentence in enumerate(self.summary_sentences):
			#filter the  spam sentences
			if(not("related:" in sentence.lower() or sentence.isupper() )):
				score_sentence_dict = {'score_total':0.0, 'score_position':0.0, 'score_length':0.0, 'score_title':0.0, 'score_sentence_closeness':0.0, 'score_sentence_term_frequency':0.0, 'sentence':sentence, 'sentence_tagged':self.summary_sentences_tagged[index]}
				self.summary_scoresAndSentencesDictList.append(score_sentence_dict)

		return True


	def reportScores(self):
		for s in self.summary_scoresAndSentencesDictList:
			print s
			print " "

	def scoreSentences(self):
		for index, current_sentence in enumerate(self.summary_scoresAndSentencesDictList):

			#Score Position

			#Add variable (starting at 1.0) to each sentence in the order they appear in the text and subtract 0.2 after each run
			#Add 1.0 - index * 0.2
			current_sentence['score_position'] = (current_sentence['score_position'] + 10.0 - index * 2.0) if (index <= 4) else current_sentence['score_position']


			

			#Score relative Length

			#Use getLongestSentenceInList to find the largest LEN count. Divide the current sentence LEN by the this value
			current_sentence['score_length'] = (len(current_sentence['sentence']) / getLongestSentenceInList(self.summary_sentences))



			#FOR BEST ENGLISH SUMMARIZATION TITLE OPT 2 SENT OPT 1

			#Score closeness to title

			#OPTION 1: call getCountCommonToBoth on sentence and title
			#OPTION 2: call getCountCommonToBoth on tag-filtered sentence and title
			closeness_title_option = 1

			if(closeness_title_option == 1):
				if(not (self.summary_title_tagged == " " or current_sentence['sentence_tagged'] == " " or self.summary_title_tagged == "" or current_sentence['sentence_tagged'] == "" or ("he said" in current_sentence['sentence_tagged']) or ("she said" in current_sentence['sentence_tagged']) or ("she says" in current_sentence['sentence_tagged']) or ("he says" in current_sentence['sentence_tagged']))):
					current_sentence['score_title'] = getCountCommonToBoth(returnStringSubForm(self.summary_title_tagged),returnStringSubForm(current_sentence['sentence_tagged']))

			else:
				current_sentence['score_title'] = getCountCommonToBoth(self.summary_title,current_sentence['sentence'])





			#Score closeness to other sentences

			#OPTION 1: call getCountCommmonToBoth on each sentence with current_sentence, and average the results
			#OPTION 2: call GCCTB on each tagged sentence with current_sentence_tagged, and average the results
			closeness_sentences_option = 2

			####AVERAGES CONSTANTS
			sentence_comparison_closeness_average = 0.0
			sentence_comparison_closeness_alpha = 0.5 
			####AVERAGES CONSTANTS /

			if(closeness_sentences_option == 1):

				for comparison_sentence in self.summary_sentences_tagged:

					if(not(returnStringSubFormBare(comparison_sentence) == " " or returnStringSubFormBare(comparison_sentence) == "" or returnStringSubFormBare(current_sentence['sentence_tagged']) == " " or returnStringSubFormBare(current_sentence['sentence_tagged']) == "" or returnStringSubFormBare(current_sentence['sentence_tagged']) == comparison_sentence)):
						sentence_comparison_closeness_average = sentence_comparison_closeness_average * sentence_comparison_closeness_alpha + (1.0 - sentence_comparison_closeness_alpha) * getCountCommonToBoth(returnStringSubFormBare(comparison_sentence), returnStringSubFormBare(current_sentence['sentence_tagged']))
			else:

				for comparison_sentence in self.summary_sentences:

					if(not(current_sentence['sentence'] == comparison_sentence)):
						sentence_comparison_closeness_average = sentence_comparison_closeness_average * sentence_comparison_closeness_alpha + (1.0 - sentence_comparison_closeness_alpha) * getCountCommonToBoth(comparison_sentence, current_sentence['sentence'])



			current_sentence['score_sentence_closeness'] = sentence_comparison_closeness_average


			#Score Term Frequency

			#For every word in the tagged sentence, num of times it is in the sentence / num of times it is in the document
			sentence_termf_average = 0.0
			sentence_termf_alpha = 0.5

			for term in word_tokenize(returnStringSubForm(current_sentence['sentence_tagged'])):
				sentence_termf_average = sentence_termf_average * sentence_termf_alpha + (1-sentence_termf_alpha) * (getOccurencesOfTermInSentence(term, returnStringSubForm(current_sentence['sentence_tagged'])) / getOccurencesOfTermInArrayOfSentences(term, self.summary_sentences_tagged))

			current_sentence['score_sentence_term_frequency'] = sentence_termf_average


		return True

	def filterSentences(self):
		

		for index, sentence in enumerate(self.summary_scoresAndSentencesDictList):
			sentenceWithoutSpaces = re.sub(' ', '', sentence['sentence'])
			if(len(sentenceWithoutSpaces) <= 25 or 'Play Video' in sentence['sentence']):
				#print ("%i   %s" % (len(sentence['sentence']), sentence['sentence']) )
				self.summary_scoresAndSentencesDictList.pop(index)
		return True


	def createSummary(self):


		#Weights
		weight_position = 0.50
		weight_length = 0.20
		weight_title = 0.35
		weight_sentence_closeness = 0.30
		weight_term_frequency = 0.0

		# weight_position = 0.9777095168471347
		# weight_length = 1.0
		# weight_title = 1.2391090313040225
		# weight_sentence_closeness = 1.3832577716822223
		# weight_term_freqeuency = 1.0


		#Add up all of the scores multiplied by their respective weights, and set total
		for sentence in self.summary_scoresAndSentencesDictList:
			sentence['score_total'] = sentence['score_position'] * weight_position + sentence['score_length'] * weight_length + sentence['score_title'] * weight_title + sentence['score_sentence_closeness'] * weight_sentence_closeness + sentence['score_sentence_term_frequency'] * weight_term_frequency


		#Copy scoresAndSentencesDictList
		final_scores_sentences_list = list()

		for sentence in self.summary_scoresAndSentencesDictList:
			final_scores_sentences_list.append(sentence)

		#Sort sentences by ['score_total']
		final_scores_sentences_list.sort(key=operator.itemgetter('score_total'), reverse=True)



		####Number of sentences to add!
		number_of_sentences_to_add = 0#(int) (final_scores_sentences_list.__len__() / 7)

		while (True):
			#length of article with number_of_sentences_to_add sentences
			presumed_length_summary = 0
			length_article = self.length_original

			for s in range(number_of_sentences_to_add):
				presumed_length_summary += len(final_scores_sentences_list[s])

			print "%4.3f / %4.3f = %4.3f" % (presumed_length_summary, length_article, (presumed_length_summary / length_article))
			if((presumed_length_summary / length_article) <= 0.007):
				number_of_sentences_to_add += 1
			else:
				break 

		

		print "NUM OF SEN: %i" % final_scores_sentences_list.__len__()



		final_sentences_to_add_list = list()
		final_score_threshold = final_scores_sentences_list[number_of_sentences_to_add - 1]['score_total']


		# #Update scoresAndSentencesDictList:
		for s in final_scores_sentences_list:
				print "%4.3f   %4.3f   %4.3f   %4.3f   %4.3f   %s" % (s['score_total'],s['score_position'],s['score_length'],s['score_title'],s['score_sentence_closeness'],s['sentence'])


		#FINAL SUMMARY
		final_summary = ""

		#Loop through sentences IN ORDER until 3 with high-enough scores are picked
		

		for sentence in self.summary_scoresAndSentencesDictList:
			if(sentence['score_total'] >= final_score_threshold and number_of_sentences_to_add > 0):
				final_summary += sentence['sentence'] + " "
				number_of_sentences_to_add -= 1




		#Clean up summary:
		#remove "(*)"
		#Check length, if too long shorten to 2 sentences
		#turn " ," to ","

		# if(self.summary_language == "en"):
		# 	regEx = re.compile(r'([^\(]*)\([^\)]*\) *(.*)')
		# 	m = regEx.match(final_summary)
		# 	while m:
  # 				final_summary = m.group(1) + m.group(2)
  # 				m = regEx.match(final_summary)

		
  			#PERMENANT SOLUTION: REMOVE EVERYTHING BEFORE THE "ALL-CAPS + HYPHEN" substring
			m = re.search(ur'^(.{1,60}? [-\xe2\u2014\u2013\u2012\u2010\u2212\u2015\u207B\uFE58\uFE63\uFF0D] {1,2}?)?(.*)', final_summary, re.UNICODE)
			#\u002Du2015\u207B\uFE58\uFE63\uFF0D\\u002D


			if m is not None and m.group(2) is not None and m.group(1) is not None:
			    final_summary = m.group(2)

		
    




		return final_summary



		


	#Test all static methods
	def testAllMethods(self):
		#Test tagSentences
		print "Testing tagSentences:"
		for s in tagSentences(self.summary_sentences):
			print(s)

		#Test returnStringSubFormBare
		print "Testing returnStringSubFormBare on %s:" % self.summary_sentences_tagged[0]
		print returnStringSubFormBare(self.summary_sentences_tagged[0])

		#Test returnStringSubForm
		print "Testing returnStringSubForm on %s:" % self.summary_sentences_tagged[0]
		print returnStringSubFormBare(self.summary_sentences_tagged[0])

		#Testing getLongestSentenceInList
		print "Testing getLongestSentenceInList" 
		print getLongestSentenceInList(self.summary_sentences)

		#Testing getCountCommonToBoth
		print "Testing getCountCommonToBoth on %s AND %s:" % (self.summary_sentences[0], self.summary_sentences[1])
 		print getCountCommonToBoth(self.summary_sentences[0], self.summary_sentences[1])

		#Testing getWordCount
		print "Testing getWordCount on %s" % self.summary_sentences[0]
		print getWordCount(self.summary_sentences[0])


#useful but static methods
	
def returnStringSubFormBare(taggedString):
	subForm = ""
	for tagged_word in taggedString:
			subForm = subForm + tagged_word[0] + " "

	return subForm


def getLongestSentenceInList(sentences):

	count = 0

	for s in sentences:
		if(s.__len__() > count):
			count = s.__len__()

	return count


def returnStringSubForm(taggedString):
	subForm = ""
	
	for tagged_word in taggedString:
		if(tagged_word[1] == 'NN' or tagged_word[1] == 'JJ' or tagged_word[1] == 'NNP' or tagged_word[1] == 'NP'):
			subForm = subForm + tagged_word[0] + " "

	return subForm


def getCountCommonToBoth(s1, s2):
	commonToBoth = 0

	splitS1 = word_tokenize(s1)
	splitS2 = word_tokenize(s2)

	for word1 in splitS1:
		for word2 in splitS2:
			if(word1 == word2):
				commonToBoth += 1

	return commonToBoth


def tagSentences(sentences):
	taggedSentences = list()

	# for index, sentence in enumerate(sentences):
	# 	print "SENTENCE: %s" % sentence
	for index, sentence in enumerate(sentences):
		taggedSentences.append(nltk.pos_tag(nltk.word_tokenize(sentence)))
		
	return taggedSentences


def getWordCount(s):
	count = 0

	tokenizedString = word_tokenize(s)

	for word in tokenizedString:
	 	count += 1

	return count

def getOccurencesOfTermInArrayOfSentences(term, array):
	number_of_times = 0

	for s in array:
		number_of_times += returnStringSubForm(s).count(term)


	return number_of_times

def getOccurencesOfTermInSentence(term, sentence):
	number_of_times = 0

	number_of_times = sentence.count(term)


	return number_of_times




def mergeShortParagraphSentences(sentences):
	#If a first sentence is 1/4 length of the one after it, combine the two
	# s_sentences = sentences
	# for index, p in enumerate(s_sentences):
	# 	s_list = sent_tokenize(p)
	# 	for index2, sen in enumerate(s_list):
	# 		if([index2+1] <= s_list.__len__()):
	# 			if(len(sen) <= 0.25 * len(s_list[index2+1])):
	# 				s_list[index2] = "%s %s" % (sen, s_list[index2+1])
	# 				s_list.pop([index2+1])
	# 	s_sentences[index] = s_list

	# return s_sentences

	s_sentences = sentences

	for index, paragraph in enumerate(s_sentences):
		sentences_in_paragraph = sent_tokenize(paragraph)
		

	return