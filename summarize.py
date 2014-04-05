#encoding=utf-8
from __future__ import division

import os
import string

from stokenizer import SentenceTokenizer


from nltk.tokenize import sent_tokenize, word_tokenize
import nltk, json, operator, urllib2, re, unicodedata

from readability.readability import Document
from bs4 import BeautifulSoup






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
			current_sentence['score_position'] = (current_sentence['score_position'] + 2.5 - index * 0.5) if (index <= 4) else current_sentence['score_position']


			

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



			if(len(current_sentence['sentence_tagged']) > 0):
				current_sentence['score_sentence_closeness'] = sentence_comparison_closeness_average / len(current_sentence['sentence_tagged'])
			else:
				current_sentence['score_sentence_closeness'] = 0


			#Score Term Frequency

			#For every word in the tagged sentence, num of times it is in the sentence / num of times it is in the document
			sentence_termf_average = 0.0
			sentence_termf_alpha = 0.5

			for term in word_tokenize(returnStringSubForm(current_sentence['sentence_tagged'])):
				sentence_termf_average = sentence_termf_average * sentence_termf_alpha + (1-sentence_termf_alpha) * (getOccurencesOfTermInSentence(term, returnStringSubForm(current_sentence['sentence_tagged'])) / getOccurencesOfTermInArrayOfSentences(term, self.summary_sentences_tagged))

			current_sentence['score_sentence_term_frequency'] = sentence_termf_average

			#score tf-idf

		return True

	def filterSentences(self):
		
			#Filter title
		# m = re.search(('[%s] (.*)'%self.summary_title), self.summary_scoresAndSentencesDictList[0]['sentence'])

		# if(m is not None):
		# 	self.summary_scoresAndSentencesDictList[0]['sentence'] = m.group(0)




###################
		# for index, sentence in enumerate(self.summary_scoresAndSentencesDictList):
		# 	sentenceWithoutSpaces = re.sub(' ', '', sentence['sentence'])
		# 	if('Play Video' in sentence['sentence'] or levenshtein(self.summary_title, sentence['sentence']) <= 5):
		# 		#print ("%i   %s" % (len(sentence['sentence']), sentence['sentence']) )
		# 		self.summary_scoresAndSentencesDictList.pop(index)
		# 	if(self.summary_title in sentence):
		# 		sentence = re.sub(self.summary_title, '', sentence)
###################


			#regexWeird = re.compile(u'(\u00a0$)')
			#sentence['sentence'] = regexWeird.sub('', sentence['sentence'])

			# regexWierd = re.findall(ur'(\\u00a0)', sentence['sentence'], re.UNICODE)
			
			# if(regexWierd is not None):
			# 	print regexWierd

			# sentence['sentence'] = sentence['sentence'].replace(u'\u00a0', ' ')
			# sentence['sentence'] = sentence['sentence'].replace(u'\u201c', '"')
			# sentence['sentence'] = sentence['sentence'].replace(u'\u201d', '"')
			# sentence['sentence'] = sentence['sentence'].replace(u'\u2019', '\'')

			
			

			#if(regexWierd is not None):








			# regexTest = re.compile(ur'with$', re.UNICODE)
			# sentence['sentence'] = regexTest.sub('', sentence['sentence'])
			
			# if(m is not None):
			# 	sentence = sentence.rsplit(' ', 1)[0]

		



		return True


	def createSummary(self):


		#Weights
		# weight_position = 1.5 
		# weight_length = 0.20
		# weight_title = 0.35
		# weight_sentence_closeness = 0.25
		# weight_term_frequency = 0.0

		weight_position = 0.5
		weight_length = 0.20
		weight_title = 0.35
		weight_sentence_closeness = 0.5
		weight_term_frequency = 0.0

		# weight_position = 0.9777095168471347
		# weight_length = 1.0
		# weight_title = 1.2391090313040225
		# weight_sentence_closeness = 1.3832577716822223
		# weight_term_freqeuency = 1.0


		#Add up all of the scores multiplied by their respective weights, and set total
		for sentence in self.summary_scoresAndSentencesDictList:
			sentence['score_total'] = sentence['score_position'] * weight_position + sentence['score_length'] * weight_length + sentence['score_title'] * weight_title + sentence['score_sentence_closeness'] * weight_sentence_closeness + sentence['score_sentence_term_frequency'] * weight_term_frequency

			#lower score by 5-10% if contains "", I'm, I.
			if("\"" in sentence['sentence'] or "I" in sentence['sentence'] or "I'm" in sentence['sentence']):
				sentence['score_total'] -= sentence['score_total'] * 0.15

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
				presumed_length_summary += len(final_scores_sentences_list[s]['sentence'])


			print "%4.3f / %4.3f = %4.3f" % (presumed_length_summary, length_article, (presumed_length_summary / length_article))
			#if((presumed_length_summary / length_article) <= 0.25):
			if(presumed_length_summary <= 420 and number_of_sentences_to_add <= final_scores_sentences_list.__len__() - 1):
				number_of_sentences_to_add += 1
			else:
				break 

			# if(presumed_length_summary <= 400):
			#  	number_of_sentences_to_add += 1
			# else:
			# 	break 

		

		print "NUM OF SEN: %i" % final_scores_sentences_list.__len__()



		final_sentences_to_add_list = list()
		final_score_threshold = final_scores_sentences_list[number_of_sentences_to_add - 1]['score_total']


		# #Update scoresAndSentencesDictList:
		for s in final_scores_sentences_list:
				print "%4.3f   %4.3f   %4.3f   %4.3f   %4.3f   %s" % (s['score_total'],s['score_position'],s['score_length'],s['score_title'],s['score_sentence_closeness'],s['sentence'])


		#FINAL SUMMARY
		final_summary = list()

		#Loop through sentences IN ORDER until 3 with high-enough scores are picked
		

		for sentence in self.summary_scoresAndSentencesDictList:
			if(sentence['score_total'] >= final_score_threshold and number_of_sentences_to_add > 0):
				final_summary.append(sentence['sentence'] + " ")
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
			#m = re.search(ur'^(.*)([-\xe2\u2014\u2013\u2012\u2010\u2212\u2015\u207B\uFE58\uFE63\uFF0D\u002D] {1,3}?)?(.*)', final_summary, re.UNICODE)
			
##############################

			# for s in final_summary:
			# 	m = re.search(ur'^((.*)[/ ][-]{1,2} )?(.*)', s, re.UNICODE)


			# 	if m is not None and m.group(2) is not None and m.group(1) is not None and m.group(3) is not None:
			# 		capsSearch = re.search(r'[A-Z]{3,10}', m.group(1), re.UNICODE)

			# 		if capsSearch is not None:
			# 			s = m.group(3)

		FINAL_SUMM = ""

		for s in final_summary:
			FINAL_SUMM += s
#######################

		
    



		#clean up summary
		

		paren = re.findall(ur'([(\u0028][^)\u0029]*[)\u0029])', FINAL_SUMM, re.UNICODE)
				
		if paren is not None:
			for s in paren:
				FINAL_SUMM = re.sub(s, '', FINAL_SUMM)



		FINAL_SUMM = re.sub(' \'', '\'', FINAL_SUMM)

		return FINAL_SUMM



		


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

def returnStringSubForm_define(taggedString):
	subForm = ""
	
	for tagged_word in taggedString:
		if(tagged_word[1] == 'NN'):
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

def levenshtein(a,b):
    "Calculates the Levenshtein distance between a and b."
    n, m = len(a), len(b)
    if n > m:
        # Make sure n <= m, to use O(min(n,m)) space
        a,b = b,a
        n,m = m,n
        
    current = range(n+1)
    for i in range(1,m+1):
        previous, current = current, [i]+[0]*n
        for j in range(1,n+1):
            add, delete = previous[j]+1, current[j-1]+1
            change = previous[j-1]
            if a[j-1] != b[i-1]:
                change = change + 1
            current[j] = min(add, delete, change)
            
    return current[n]



def mergeShortParagraphSentences(sentences):

	# for s in sentences:
	# 	print "PRE: %s" % s

	#filter out all the blank sentences
	sentences = filter(lambda x: (re.search(ur'[a-zA-Z|“|"|”|\u201C|\u201D|.|?|,|\(|\)]', x, re.UNICODE) is not None), sentences) 

	sentencesToAdd = list()


	#if this sentence ends in a letter or a comma and DOES NOT contain capitalized words:
		#this sentence = this sentence + next sentence
		#pop next sentence
		#index -= 1



	# for n in range(sentences.__len__()):
	# 	regex_endsWithLetterOrComma = re.search(ur'[a-zA-Z|,]$', sentences[n].rstrip('\n').strip(), re.UNICODE)
	# 	regex_containsCapitalWords = re.search(ur'([A-Z]{3,10})', sentences[n].rstrip('\n').strip(), re.UNICODE)

	# 	if(regex_containsCapitalWords is not None and word_tokenize(regex_containsCapitalWords.group(0)).__len__() > 2):
	# 		boolean_containsCapitalWords = True
	# 	else:
	# 		boolean_containsCapitalWords = False
		
	# 	print ("%s shouldbecombined:%r") % (sentences[n].rstrip('\n').strip(), regex_endsWithLetterOrComma is not None and boolean_containsCapitalWords is False)

	for n in range(sentences.__len__() - 1):
		regex_matchQuotations = re.findall(ur'([“|"|”|\u201C|\u201D])', sentences[n].rstrip('\n').strip(), re.UNICODE)

		if(regex_matchQuotations is not None and regex_matchQuotations.__len__() % 2 != 0):
			sentences[n+1] = sentences[n] + " " + sentences[n+1]
			sentences[n] = ""
			#sentencesToAdd.append(sentences[n])

	for n in range(sentences.__len__() - 2):
		regex_matchParen = re.findall(ur'([\(\)\u0028\u0029])', sentences[n].rstrip('\n').strip(), re.UNICODE)
		
		if(regex_matchParen is not None and regex_matchParen.__len__() % 2 != 0):
			sentences[n+1] = sentences[n] + " " + sentences[n+1]
			sentences[n] = ""
			#sentencesToAdd.append(sentences[n])
			











	for n in range(sentences.__len__() - 1):

		regex_next_matchPeriods = re.findall(ur'([\.]$)', sentences[n+1].rstrip('\n').strip(), re.UNICODE)

		regex_this_endsWithPeriod = re.findall(ur'[\.]$', sentences[n].rstrip('\n').strip(), re.UNICODE)

		regex_next_startsWithCaps = re.search(ur'^[A-Z](?!.*\.com|.*\.org|.*\.edu|.*\.net|witter|acebook|echcrunch|oogle)', sentences[n+1].rstrip('\n').strip(), re.UNICODE)

		regex_this_endsWithComma = re.search(ur',$', sentences[n].rstrip('\n').strip(), re.UNICODE)

		regex_this_endsWithUrlPattern = re.search(ur'(.com|.net|.edu|.us)$', sentences[n].rstrip('\n').strip(), re.UNICODE)
		regex_next_endsWithUrlPattern = re.search(ur'(.com|.net|.edu|.us)$', sentences[n+1].rstrip('\n').strip(), re.UNICODE)
		
		regex_this_endsWithLetterOrComma = re.search(ur'[a-zA-Z0-9,:\(\)]$', sentences[n].rstrip('\n').strip(), re.UNICODE)
		regex_this_startsWithComma = re.search(ur'^,', sentences[n].rstrip('\n').strip(), re.UNICODE)
		regex_this_containsCapitalWords = re.search(ur'([A-Z]{5,10})', sentences[n].rstrip('\n').strip(), re.UNICODE)

		regex_next_endsWithLetterOrComma = re.search(ur'[a-zA-Z0-9,:\(\)]$', sentences[n+1].rstrip('\n').strip(), re.UNICODE)
		regex_next_startsWithComma = re.search(ur'^[,\.]', sentences[n+1].rstrip('\n').strip(), re.UNICODE)
		regex_next_containsCapitalWords = re.search(ur'([A-Z]{5,10})', sentences[n+1].rstrip('\n').strip(), re.UNICODE)


		# if(regex_this_endsWithLetterOrComma is not None and regex_next_containsCapitalWords is None):
		# 	sentences[n+1] = sentences[n] + " " + sentences[n+1]
		# 	sentences[n] = ""
		# elif(regex_next_startsWithComma is not None and regex_this_endsWithLetterOrComma is not None):
		# 	sentences[n+1] = sentences[n] + " " + sentences[n+1]
		# 	sentences[n] = ""

		#if(regex_this_endsWithPeriod is None or (regex_this_endsWithComma is not None or (regex_this_endsWithLetterOrComma is not None and (regex_next_startsWithCaps is None or (regex_next_startsWithCaps is not None and len(sentences[n+1]) <= 30 ) )) or regex_next_startsWithComma)):
		if(regex_this_endsWithPeriod is None or regex_this_endsWithComma is not None or regex_this_endsWithLetterOrComma is not None or regex_next_startsWithComma is not None):
			 sentences[n+1] = sentences[n] + " " + sentences[n+1]
			 sentences[n] = ""
			#sentencesToAdd.append(sentences[n])




		#THIS ENDS WITH PERIOD

		#OR

		#THIS ENDS WITH COMMA

		#OR 

		#THIS ENDS WITH LETTER AND NEXT DOESN'T START WITH CAPS OR IT DOES START WITH CAPS AND THE LENGTH OF THE NEXT SENTENCE IS <30
		

		#print "SEN: %s this_endsLetterComma: %r next_containsCapitalWords: %r next_startsWithComma: %r WILLBEADDED: %r" % (sentences[n].rstrip('\n').strip(), regex_this_endsWithLetterOrComma is not None, regex_next_containsCapitalWords is not None,  regex_next_startsWithComma is not None, (regex_this_endsWithLetterOrComma is not None and regex_this_endsWithUrlPattern is None and regex_next_startsWithCaps is None) or regex_next_startsWithComma)
		# print ""
		# print ""

	# #get rid of " " sentences
	# for index, s in enumerate(sentences):
	# 	regex_blank_sentence = re.search(r'[a-zA-Z]', s)
	# 	if(regex_blank_sentence is not None):
	# 		sentences.remove(s)



	# for s in sentencesToAdd:
	# 	sentences[sentences.index(s) + 1] = sentences[sentences.index(s)] + " " + sentences[sentences.index(s) + 1]
	# 	sentences[sentences.index(s)] = ""
		


	#clean everything up, get rid of the tags!



		# regex_endsWithLetterOrComma = re.search(ur'[a-zA-Z|,|:]$', sentences[n].rstrip('\n').strip(), re.UNICODE)
	 # 	regex_containsCapitalWords = re.search(ur'([A-Z]{3,10})', sentences[n].rstrip('\n').strip(), re.UNICODE)
		# if(regex_containsCapitalWords is not None and word_tokenize(regex_containsCapitalWords.group(0)).__len__() > 2):
		# 	boolean_containsCapitalWords = True
		# else:
		# 	boolean_containsCapitalWords = False

		# print ("%s shouldBeCombinedWNext:%r \n\n\n" % (sentences[n], regex_endsWithLetterOrComma is not None and boolean_containsCapitalWords is False and sentences[n] is not ""))

		# if(regex_endsWithLetterOrComma is not None and boolean_containsCapitalWords is False and sentences[n] is not ""):
		# 	sentences[n] = sentences[n] + " " + sentences[n+1]
		# 	sentences[n+1] = ""
		# 	length = sentences.__len__()
		# 	n -= 1

	

	for sen in sentencesToAdd:
		sentence_index = 0
		for index, s in enumerate(sentences):
			if(sen in s and index < sentences.__len__() - 1):
				sentence_index = index

		sentences[sentence_index + 1] = sentences[sentence_index] + " " + sentences[sentence_index + 1]
		sentences[sentence_index] = ""




	 

	#get rid of super long sentences: > 400 characters
	sentences = filter(lambda x: (len(x) <= 650 and not(u"Editor’s note" in x)), sentences) 


	final_sentences = list()

	for s in sentences:
		regex_matchPeriods = re.findall(ur'([\.])', s.rstrip('\n').strip(), re.UNICODE)
		regex_matchQuotations = re.findall(ur'([“|"|”|\u201C|\u201D])', s.rstrip('\n').strip(), re.UNICODE)
		# print "SEN: %s %i" % (s, regex_matchPeriods.__len__())
		# print ""
		if(regex_matchQuotations.__len__() == 0 and regex_matchPeriods.__len__() > 1):
			for sen in sent_tokenize(s):
				final_sentences.append(sen)
		else:
			final_sentences.append(s)



	# piece together all the FULL SENTENCES that need to be together
	for n in range(sentences.__len__() - 1):
		regex_next_startsWithBut = re.search(ur'^(But|but)', final_sentences[n+1].rstrip('\n').strip(), re.UNICODE)
		if(regex_next_startsWithBut is not None):
			final_sentences[n+1] = final_sentences[n] + " " + final_sentences[n+1]
			final_sentences[n] = ""


	#filter out all the blank sentences one more time
	final_sentences = filter(lambda x: (re.search(ur'[a-zA-Z|“|"|”|\u201C|\u201D|.|?|,]', x, re.UNICODE) is not None), final_sentences)


	for s in final_sentences:
		print "SEN_FINAL: " + s
		print ""


	
	return sentences[:2] #return just sentences for non-wiki


