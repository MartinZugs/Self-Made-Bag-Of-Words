import collections
import numpy
import random
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

#All of this between this and the next comment is just opening the text files
#and splitting them
positive = open("rt-polarity.pos.txt","r")
negative = open("rt-polarity.neg.txt", "r")

#Arrays of all the reviews with one location in the array is one review
pos_reviews = ([])
neg_reviews = ([])

#Filling the array
with open("rt-polarity.pos.txt") as f:
    for line in f:
        pos_reviews.append(line)

with open("rt-polarity.neg.txt") as f:
    for line in f:
        neg_reviews.append(line)

pos_content = positive.read()
neg_content = negative.read()

positive.close()
negative.close()

pos_words = pos_content.split()
neg_words = neg_content.split()

#Get a set of all of the unique words in each text file
unique_pos_words = set(pos_words)
unique_neg_words = set(neg_words)
total_unique_words = unique_pos_words.union(unique_neg_words)

#Count of all unique words in each text file
pos_word_count = collections.Counter(pos_words)
neg_word_count = collections.Counter(neg_words)
tot_word_count = collections.Counter(pos_words + neg_words)

#Get rid of all words that appear less than 5 times and more than 5000
print("Getting rid of words that occur more than 4700 times or less than 10 times")
print("Total count of unique words before: " + str(len(total_unique_words)))
for x in tot_word_count:
    if(tot_word_count[x] < 10):
        total_unique_words.remove(x)
    elif(tot_word_count[x] > 4700):
        total_unique_words.remove(x)
    else:
        pass

print("Total count of unique words after: " + str(len(total_unique_words)))
print()

#Make one full dataset where each column is a unique word and each row is the number
#of occurences of that word in the dataset. One column will be a label as well.
master = ([])

for x in pos_reviews:
    master.append([x, "1"])

for x in neg_reviews:
    master.append([x, "-1"])

#This is the master data file, each one of the lists contains the words and their label
random.shuffle(master)

#Now to get separate the data into test and training, 80% train 20% test
print("Separating into 80% training and 20% test")
length = len(master)
eighty = round(length*0.8)
train = master[:eighty]
test = master[eighty:]
print("Master size: " + str(len(master)))
print("Train size: " + str(len(train)))
print("Test size: " + str(len(test)))
print()

#Now to make a vectorizer for the training data
train_vectorized_data = numpy.zeros(shape=(len(train),len(total_unique_words)))
print("Shape of vectorized numpy training array: " + str(numpy.shape(train_vectorized_data)))
unique_words_list = list(total_unique_words)
train_labels=[]

for x in range(len(train_vectorized_data)):
    review_counter = collections.Counter(train[x][0].split())
    train_labels.append(train[x][1])
    tlist = []
    
    for y in review_counter:
        if(tot_word_count[y] < 10):
            tlist.append(y)
        elif(tot_word_count[y] > 4700):
            tlist.append(y)
        else:
            pass

    for w in tlist:
        del review_counter[w]

    for e in review_counter:
        train_vectorized_data[x][unique_words_list.index(e)] = review_counter[e]

#Now to make a vectorizer for the test data
test_vectorized_data = numpy.zeros(shape=(len(test),len(total_unique_words)))
print("Shape of vectorized numpy test array: " + str(numpy.shape(test_vectorized_data)))
unique_words_list = list(total_unique_words)
test_labels=[]

for x in range(len(test_vectorized_data)):
    review_counter = collections.Counter(test[x][0].split())
    test_labels.append(test[x][1])
    tlist = []
    
    for y in review_counter:
        if(tot_word_count[y] < 10):
            tlist.append(y)
        elif(tot_word_count[y] > 4700):
            tlist.append(y)
        else:
            pass

    for w in tlist:
        del review_counter[w]

    for e in review_counter:
        test_vectorized_data[x][unique_words_list.index(e)] = review_counter[e]



#Now to run logistic regression on that vectorized data, just for kicks
log_reg = LogisticRegression(solver='liblinear')
log_reg.fit(train_vectorized_data, train_labels)

score = log_reg.score(test_vectorized_data, test_labels)
print()
print("Accuracy on test set using logistic regression: " + str(score))

predictions = log_reg.predict(test_vectorized_data)

cm = metrics.confusion_matrix(test_labels, predictions)
print(cm)
print()

#Now train logistic regression using n-fold cross validation
print("Doing logistic regression with n-fold cross validation")
cf_train_data = []
cf_train_labels = []
cf_test_data = []
cf_test_labels = []
results = []
total = 0
t=[]
t_l=[]
length = len(train)
ten = round(length*0.1)
train_1 = train_vectorized_data[:ten]
train_2 = train_vectorized_data[ten:ten*2]
train_3 = train_vectorized_data[ten*2:ten*3]
train_4 = train_vectorized_data[ten*3:ten*4]
train_5 = train_vectorized_data[ten*4:ten*5]
train_6 = train_vectorized_data[ten*5:ten*6]
train_7 = train_vectorized_data[ten*6:ten*7]
train_8 = train_vectorized_data[ten*7:ten*8]
train_9 = train_vectorized_data[ten*8:ten*9]
train_10 = train_vectorized_data[ten*9:]
t.append(train_1)
t.append(train_2)
t.append(train_3)
t.append(train_4)
t.append(train_5)
t.append(train_6)
t.append(train_7)
t.append(train_8)
t.append(train_9)
t.append(train_10)

train_labels_1 = train_labels[:ten]
train_labels_2 = train_labels[ten:ten*2]
train_labels_3 = train_labels[ten*2:ten*3]
train_labels_4 = train_labels[ten*3:ten*4]
train_labels_5 = train_labels[ten*4:ten*5]
train_labels_6 = train_labels[ten*5:ten*6]
train_labels_7 = train_labels[ten*6:ten*7]
train_labels_8 = train_labels[ten*7:ten*8]
train_labels_9 = train_labels[ten*8:ten*9]
train_labels_10 = train_labels[ten*9:]
t_l.append(train_labels_1)
t_l.append(train_labels_2)
t_l.append(train_labels_3)
t_l.append(train_labels_4)
t_l.append(train_labels_5)
t_l.append(train_labels_6)
t_l.append(train_labels_7)
t_l.append(train_labels_8)
t_l.append(train_labels_9)
t_l.append(train_labels_10)

for q in range(10):
    for e in range(10):
        if q == e:
            for k in t[e]:
                cf_test_data.append(k)
            for i in t_l[e]:
                cf_test_labels.append(i)
        else:
            for k in t[e]:
                cf_train_data.append(k)
            for i in t_l[e]:
                cf_train_labels.append(i)

    log_reg_2 = LogisticRegression(solver='liblinear', max_iter = 400)
    log_reg_2.fit(cf_train_data, cf_train_labels)
    score = log_reg_2.score(cf_test_data, cf_test_labels)
    results.append(score)
    cf_train_data = []
    cf_train_labels = []
    cf_test_data = []
    cf_test_labels = []
    

for u in results:
    total = total + u

print("Average percent accuracy over all n-fold validation is: " + str(total/len(results)))
print()

#Now do some fancy grid search over some logistic regression with the n-fold cross validation!
print("Now doing grid search with n-fold validation to find the best hyper parameters!")
c_values = [0.001, 0.01, 0.1, 1, 20, 30, 400]
penalties = ['l1','l2']
best = 0

for q in range(10):
    for e in range(10):
        if q == e:
            for k in t[e]:
                cf_test_data.append(k)
            for i in t_l[e]:
                cf_test_labels.append(i)
        else:
            for k in t[e]:
                cf_train_data.append(k)
            for i in t_l[e]:
                cf_train_labels.append(i)

    for solve in penalties:

        for c in c_values:
            
            log_reg_t = LogisticRegression(solver='liblinear', C=c, max_iter=500, penalty=solve)
            log_reg_t.fit(cf_train_data, cf_train_labels)

            score = log_reg_t.score(cf_test_data, cf_test_labels)
            #print("Accuracy on test set using logistic regression where penalty is " + solve + " and C value is " + str(c) + ": " + str(score))

            if score > best:
                best = score
                best_penalty = solve
                best_c = c
            else:
                pass

            predictions = log_reg_t.predict(cf_test_data)

            cm = metrics.confusion_matrix(cf_test_labels, predictions)
            #print(cm)
            #print()


    cf_train_data = []
    cf_train_labels = []
    cf_test_data = []
    cf_test_labels = []

print("Best overall is: penalty is: " + best_penalty + " with a C value of: " + str(best_c) + " which gives an accuracy of: " + str(best))
print()

#And now for the moment of truth!
print("And now to test on the untouched test data!!")

log_reg_final = LogisticRegression(solver='liblinear', max_iter=500, C=best_c, penalty=best_penalty)
log_reg_final.fit(train_vectorized_data, train_labels)

score = log_reg_final.score(test_vectorized_data, test_labels)
print("Accuracy on test set using logistic regression: " + str(score))

predictions = log_reg_final.predict(test_vectorized_data)

cm_final = metrics.confusion_matrix(test_labels, predictions)
print(cm_final)
print()

        

    
