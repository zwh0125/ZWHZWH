import nltk
import matplotlib.pyplot as plt
from nltk import FreqDist
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer

# Load Moby Dick text
with open('moby_dick.txt', 'r', encoding='utf-8') as file:
    text = file.read()

# Tokenization
sentences = sent_tokenize(text)
words = word_tokenize(text)

# Stopwords filtering
stop_words = set(stopwords.words('english'))
filtered_words = [word.lower() for word in words if word.isalpha() and word.lower() not in stop_words]

# Parts-of-Speech tagging
pos_tags = nltk.pos_tag(filtered_words)

# POS frequency
fdist = FreqDist(tag for word, tag in pos_tags)
common_pos = fdist.most_common(5)

# Lemmatization
lemmatizer = WordNetLemmatizer()
top_20_tokens = [lemmatizer.lemmatize(word, pos) for word, pos in pos_tags][:20]

# Plotting frequency distribution
fdist.plot()
plt.show()

# Display results
print("Top 5 Parts of Speech and their Frequencies:")
for pos, count in common_pos:
    print(f"{pos}: {count}")

print("\nTop 20 Lemmatized Tokens:")
print(top_20_tokens)
