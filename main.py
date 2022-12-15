def read_words(filename):
    words = []
    with open(filename, 'r') as infile:
        lines = infile.readlines()
        for line in lines:
            words.append(line.strip())

    return words

def word_counts(words):
  counts = {}
  for word in words:
    if word in counts:
      counts[word] += 1
    else:
      counts[word] = 1

  return counts

def main():
    CLASSES = ["neg", "pos"]
    PRIORS = {"neg": 0.5, "pos": 0.5}

    pos_words_train = read_words('pos.txt')
    neg_words_train = read_words('neg.txt')
    
    print(pos_words_train)


if __name__ == '__main__':
    main()
