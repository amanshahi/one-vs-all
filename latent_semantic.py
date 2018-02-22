import os
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from collections import *
transformer = TfidfTransformer(smooth_idf=False)
path = './q2data/train/'
categories = sorted(list(os.listdir(path)))
train = 0.8
# train, training_set, validating_set = 0.8, [], []
# term_frequencies = [defaultdict(dict) for i in xrange(len(categories))]
doc_words = [defaultdict(dict) for i in xrange(len(categories))]
no_of_files = [len(os.listdir(path + categories[i])) for i in range(len(categories))]
word_doc, validating_doc, validating_class = [], [], []
rank, cnt = {}, 0
allWords, original_class = [], []
print 'Categories : ', categories

def read_files():
	global cnt
	for i in xrange(len(categories)):
		new_path1 = path + categories[i] + '/'
		docs = sorted(list(os.listdir(new_path1)))
		for j in xrange(len(docs)):
			file = new_path1 + docs[j]
			with open(file) as fd:
				data = fd.read()
			words = data.split()
			for k in xrange(len(words)): 
				words[k] = words[k].lower()
				try:rank[words[k]]
				except:
					rank[words[k]] = cnt
					allWords.append(words[k])
					cnt+=1
				# try:term_frequencies[int(categories[i])][words[k]][docs[j]] += 1
				# except:term_frequencies[int(categories[i])][words[k]][docs[j]] = 1
				try: doc_words[int(categories[i])][docs[j]][words[k]] += 1
				except:doc_words[int(categories[i])][docs[j]][words[k]] = 1

def word_doc_matrix():
	for i in xrange(len(categories)):
		new_path1 = path + categories[i] + '/'
		docs = sorted(list(os.listdir(new_path1)))
		train_only_for = int(train * len(docs))
		for j in xrange(len(docs)):
			temp = []
			print j
			for k in xrange(cnt):
				try:temp.append(doc_words[int(categories[i])][docs[j]][allWords[k]])
				except:temp.append(0)
			# temp.append(docs[j])
			# temp.append(categories[i])
			if j <= train_only_for:
				word_doc.append(temp)
				original_class.append(i)
			else:
				validating_doc.append(temp)
				validating_class.append(i)


def tfidf():
	# with open('./words_doc','w') as file: file.write(str(word_doc))
	tfidf_matrix = transformer.fit_transform(word_doc)
	print tfidf_matrix.toarray()

	U,sigma,VT = np.linalg.svd(tfidf_matrix.toarray(), full_matrices=False)
	k = 25
	new_words_doc = np.dot(U,np.dot(np.diag(sigma),VT))
	print new_words_doc
	sigma[-k:] = 0
	new_words_doc = np.dot(U,np.dot(np.diag(sigma),VT))
	print new_words_doc


def multi_class_perceptron():
	distinct_words = len(allWords)
	W = np.array([[0 for i in xrange(distinct_words)] for j in range(len(categories))])
	misclassified = 10
	while misclassified != 0:
		misclassified = 0
		for i in xrange(len(word_doc)):
			for j in xrange(len(categories)):
				temp_class = np.dot(W[j], word_doc[i])
				if j == 0: 
					argmax = temp_class
					classify = j
				elif argmax < temp_class: 
					argmax = temp_class
					classify = j
			if classify != original_class[i]:
				misclassified += 1
				W[classify] -= word_doc[i]
				W[original_class[i]] += word_doc[i]
		print misclassified


def find_class(k, j):
	if original_class[k] != j: return -1
	return 1

def one_versus_all_perceptron():
	number_of_classes = len(categories)
	W = [ np.array([0 for j in range(len(word_doc[0]))]) for i in range(number_of_classes) ] 
	B = [0 for i in range(number_of_classes)]
	VOTINGS = [[W[i], B[i], 1] for i in range(number_of_classes)]
	number_of_epochs = 1
	for i in range(number_of_epochs):
		for j in range(number_of_classes):
			for k in range(len(word_doc)):
				print 'Learning for class = ', j, 'Word vector = ', k
				temporary = find_class(k, j)
				if (np.dot(np.array(W[j])*np.array(word_doc[k])) + B[j])*temporary <= 0:
					W[j] = W[j] + word_doc[k]*temporary
					B[j] = B[j] + temporary
					VOTINGS[j].append([W[j], B[j], 1])
				else:
					VOTINGS[j][-1][2] += 1
	return VOTINGS

def one_V_all_model():
	misclassified = 0
	example = OneVsRestClassifier(LinearSVC(random_state = 0)).fit(word_doc, original_class)
	# print original_class
	predicted = example.predict(word_doc)
	for i in xrange(len(predicted)): print predicted[i],original_class[i]
	predicted = example.predict(validating_doc)
	for i in xrange(len(predicted)): 
		if predicted[i] != validating_class[i]:misclassified+=1
	print misclassified, len(predicted)



def relavent_docs(given_doc, orig_class):
	cosines, topk = [], 10
	countings = [[0,i] for i in range(len(categories))]
	# print 'Just entered!'
	for i in xrange(len(word_doc)):
		# print 'Running on doc',i
		similarity = cosine_similarity([word_doc[i]], [given_doc])
		cosines.append([similarity,i])
	cosines = sorted(cosines)[::-1]
	relavent_documents = cosines[:topk]
	docs = [i[1] for i in relavent_documents]
	predicted_categories = [original_class[i[1]] for i in relavent_documents]
	for i in predicted_categories: countings[i][0] += 1
	countings = sorted(countings)
	# print 'Predicted category = ', countings[-1][1]
	# print docs
	# print predicted_categories

	if countings[-1][1] != orig_class: return 1
	return 0

read_files()
word_doc_matrix()

# print original_class
# print validating_class

word_doc = np.array(word_doc)	
validating_doc = np.array(validating_doc)
# print 'Yeh toh ho gya'		
# tfidf()

# multi_class_perceptron()
# one_V_all_model()


# misclassified = 0
# for i in range(len(validating_doc)):
# 	print 'Running for ',i,' out of ',len(validating_doc)
# 	misclassified += relavent_docs(validating_doc[i], validating_class[i])
# print misclassified, len(validating_doc)


one_versus_all_perceptron()

# relavent_docs(word_doc[1])
print 'Total number of distinct words = ', cnt
