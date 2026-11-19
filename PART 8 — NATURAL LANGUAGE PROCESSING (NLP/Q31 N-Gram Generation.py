from nltk import word_tokenize
from nltk.util import bigrams, trigrams
text = 'the greatest glory in living lies not in never falling nut in raising every lies'
tokens = word_tokenize(text)
bigram_list = list(bigrams(tokens))
print('Bigrams:')
for bigram in bigram_list:
    print(bigram)
trigram_list = list(trigrams(tokens))
print('\nTrigrams:')
for trigram in trigram_list:
    print(trigram)