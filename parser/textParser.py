import json
import nltk
import time
from nltk.tokenize import wordpunct_tokenize
from nltk.metrics import edit_distance
nltk.download('punkt')


def compute_best_review_for_item(source = '../dataset/yelp_training_set_review.json'):
    best_item_votes = {}
    item_to_review = {}
    for line in open(source, mode='r'):
        review = json.loads(line.strip())
        review_id = review['review_id']
        item_id = review['business_id']
        votes = review['votes']
        funny = int(votes['funny'])
        useful = int(votes['useful'])
        cool = int(votes['cool'])
        total_votes = funny + useful + cool
        if item_id not in best_item_votes or total_votes > best_item_votes[item_id]:
            best_item_votes[item_id] = total_votes
            item_to_review[item_id] = review_id

    best_review_for_item_file = open('../dataset/best_review_for_item.txt', mode='w')
    for item_id, review_id in item_to_review.items():
        best_review_for_item_file.write(item_id + '\t' + review_id + '\n')

    best_review_for_item_file.close()


# Get's all words from the glove embeddings.
def getGloveWords(source = '../dataset/glove.6B.300d.txt'):
    embeddings = set()
    for line in open(source, mode='r', encoding='utf-8'):
        parts = line.strip().split(' ')
        wordForVector = parts[0]
        embeddings.add(wordForVector)

    return embeddings


#Get's the review text content.
def extractReviewTextContent(rawLine):
    review = json.loads(rawLine.strip())
    text = review['text']
    review_id = review['review_id']
    item_id = review['business_id']
    reviewWords = [word.lower() for word in wordpunct_tokenize(text)]
    return review_id, item_id, reviewWords


def findMostSimilarReplacement(word, gloveWords):
    #print('Finding most similar token for replacement for word ' + word)
    startTime = time.time()
    word_distance_with_other_words = [(gloveToken, edit_distance(word, gloveToken)) for gloveToken in gloveWords
                                      if (abs(len(word) - len(gloveToken)) <= 2)]

    mostSimilarWord = min(word_distance_with_other_words, key=lambda x: x[1])
    endTime = time.time()
    #print('Time for calculating edit distance for word: ' + str(endTime - startTime))
    return mostSimilarWord


def replaceTokens(reviewWordsNotInGlove, gloveWords):
    glove_words_as_list = list(gloveWords)
    replacement_tokens = {}
    file_to_write = open('replacement_words.txt', mode='w', encoding='utf-8')
    count = 0
    for unknownWord in reviewWordsNotInGlove:
            if count % 500 == 0:
                print('Searching for a replacement of '+str(count) + " word: " + unknownWord)

            replacementWordWithDistance = findMostSimilarReplacement(unknownWord, glove_words_as_list)
            replacementWord = replacementWordWithDistance[0]
            if replacementWordWithDistance[1] <= 2:
                replacement_tokens[unknownWord] = replacementWord
                file_to_write.write(unknownWord + '\t' + replacementWord + '\t' + str(replacementWordWithDistance[1]) + '\n')

            count += 1

    file_to_write.close()

    return replacement_tokens


def findNonGloveTokensReplacement(reviewsWords, gloveWords):
    print('Replacing non glove tokens')
    allReviewWords = set(word for review in reviewsWords for word in review)
    reviewWordsNotInGlove = allReviewWords.difference(gloveWords)
    return replaceTokens(reviewWordsNotInGlove, gloveWords)


def getTokensReplacementFromFile(filename='tokens_replacement.txt'):
    replacement_tokens={}
    tokens_file=open(filename, mode='r', encoding='utf-8')

    for line in tokens_file:
        parts = line.strip().split('\t')
        if len(parts) == 3:
            unknown_word = parts[0]
            replacement_word = parts[1]
            replacement_tokens[unknown_word] = replacement_word

    return replacement_tokens


def extractAndReplaceReviewsTokens(
        source='../dataset/yelp_training_set_review.json',
        contentExtractor = extractReviewTextContent,
        tokensExtractor = getTokensReplacementFromFile
        ):
    reviews_words = {}
    item_by_review = {}
    for line in open(source, mode='r'):
        f_review_id, item_id, review_words = contentExtractor(line)
        reviews_words[f_review_id] = review_words
        item_by_review[f_review_id] = item_id

    glove_tokens_replacement = tokensExtractor()
    processed_reviews_file = open('../dataset/processed_reviews_text.txt', mode='w', encoding='utf-8')
    for review_id, words in reviews_words.items():
        if len(words) > 0:
            item_id = item_by_review[review_id]
            processed_reviews_file.write(item_id + '\t' + review_id + '\t')
            review_words_cleaned = map(lambda word: glove_tokens_replacement.get(word, word), words)
            processed_reviews_file.write('\t'.join(review_words_cleaned))
            processed_reviews_file.write('\n')

    processed_reviews_file.close()

#extractAndReplaceReviewsTokens()

compute_best_review_for_item()