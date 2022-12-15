from naive_bayes_classifier import MyNaiveBayesClassifier


def read_words(filename):
    words = []
    with open(filename, 'r') as infile:
        lines = infile.readlines()
        for line in lines:
            words.append(line.strip())

    return words


def main():
    pos_words_train = read_words('pos.txt')
    neg_words_train = read_words('neg.txt')
    
    print(pos_words_train)


    clf = MyNaiveBayesClassifier()


if __name__ == '__main__':
    main()
