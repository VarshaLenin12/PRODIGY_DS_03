# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from textblob import TextBlob
from wordcloud import WordCloud

# Mount Google Drive to access files
from google.colab import drive
drive.mount('/content/drive')

# Read validation and training datasets from Google Drive
df1 = pd.read_csv('/content/drive/MyDrive/twitter_validation.csv')
df2 = pd.read_csv('/content/drive/MyDrive/twitter_training.csv')

# Display the first few rows of each dataset
df1.head()
df2.head()

# Display columns in each dataset
df1.columns
df2.columns

# Define a function to calculate sentiment polarity using TextBlob
def get_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

# Apply sentiment analysis to 'Irrelevant' column in validation dataset and 'Positive' column in training dataset
df1['Sentiment'] = df1['Irrelevant'].apply(get_sentiment)
df2['Sentiment'] = df2['Positive'].apply(get_sentiment)

# Visualize sentiment distribution for each dataset
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.histplot(df1['Sentiment'], bins=20, kde=True, color='blue')
plt.title('Sentiment Distribution - Facebook')

plt.subplot(1, 2, 2)
sns.histplot(df2['Sentiment'], bins=20, kde=True, color='green')
plt.title('Sentiment Distribution - Borderlands')

plt.tight_layout()
plt.show()

# Calculate and visualize average sentiment scores for each dataset
avg_sentiment_df1 = df1['Sentiment'].mean()
avg_sentiment_df2 = df2['Sentiment'].mean()

plt.bar(['Facebook', 'Borderlands'], [avg_sentiment_df1, avg_sentiment_df2], color=['blue', 'green'])
plt.title('Average Sentiment Score Comparison')
plt.ylabel('Average Sentiment Score')
plt.show()

# Generate word clouds for positive and negative sentiments in the training dataset
positive_text = ' '.join(df2[df2['Sentiment'] > 0]['Positive'])
negative_text = ' '.join(df2[df2['Sentiment'] < 0]['Positive'])

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
wordcloud_positive = WordCloud(width=800, height=400, background_color='white').generate(positive_text)
plt.imshow(wordcloud_positive, interpolation='bilinear')
plt.title('Positive Sentiment Word Cloud')

plt.subplot(1, 2, 2)
wordcloud_negative = WordCloud(width=800, height=400, background_color='black', colormap='Reds').generate(negative_text)
plt.imshow(wordcloud_negative, interpolation='bilinear')
plt.title('Negative Sentiment Word Cloud')

plt.tight_layout()
plt.show()
