import math
import nltk
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords

nltk.download('movie_reviews')

def read_words(filename):
    words = []
    with open(filename, 'r') as infile:
        lines = infile.readlines()
        for line in lines:
            words.append(line.strip())

    return words

def makeBagOfWords(reviewLst):
    '''
    Extract words for each review
    Throw out those that are not alphabetic as well as short frequent English words
    Add the result to bag
    At the end, bag will be the list of words in a certain category of review.
    Use NLTK functions
    '''
                  
    bag = []
    for review in reviewLst:
        words = movie_reviews.words(review)  #list of words in a review 
        words = [word for word in words if word.isalpha()] #remove items witn non-alpha chars
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word not in stop_words] #stop words removed
        bag = bag + words
    return bag

def word_counts(words):
  counts = {}
  for word in words:
    if word in counts:
      counts[word] += 1
    else:
      counts[word] = 1

  return counts

def word_likelihoods(words, vocab):
    counts = word_counts(words)
    likelihoods = {}
    denominator = len(words) + len(vocab)
    for word in vocab:
        if word in counts:
            likelihoods[word] = (counts[word] + 1) / denominator
        else:
            likelihoods[word] = 1 / denominator
    return likelihoods

def test_review(review, priors, likelihoods):
    classes = priors.keys()
 
    sums = {}
    for c in classes:
        sums[c] = math.log(priors[c])
        for word in review:
            if word in likelihoods[c]:
                sums[c] += math.log(likelihoods[c][word])
 
    max_class = max(classes, key= lambda c: sums[c])
    return max_class


def main():
    '''
    Training
    '''
    classes = ["neg", "pos"]
    priors = {"neg": 0.5, "pos": 0.5}
 
    pos_words_train = read_words('pos.txt')
    neg_words_train = read_words('neg.txt')
    V = set(pos_words_train + neg_words_train)
   
    pos_likelihoods = word_likelihoods(pos_words_train, V)
    neg_likelihoods = word_likelihoods(neg_words_train, V)
    likelihoods = {"pos": pos_likelihoods, "neg": neg_likelihoods}

    '''
    Testing
    '''
    pos_reviews = read_words('posTst.txt')
    neg_reviews = read_words('negTst.txt')
    print(pos_reviews)

    for pos_review in pos_reviews:
        review_words = makeBagOfWords([pos_review])
        prediction_class = test_review(review_words, priors, likelihoods)
        print(prediction_class)


    for neg_review in neg_reviews:
        review_words = makeBagOfWords([neg_review])
        prediction_class = test_review(review_words, priors, likelihoods)
        print(prediction_class)

if __name__ == '__main__':
    main()
