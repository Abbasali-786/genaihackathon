import os
import requests
from datetime import datetime
from newspaper import Article
from groq import Groq

# Your News API key
API_KEY = '446dc1fa183e4e859a7fb0daf64a6f2c'
BASE_URL = 'https://newsapi.org/v2/everything'

# Set up Groq client (ensure your API key is correctly set)
client = Groq(
    api_key= "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"
)

# Function to fetch news based on topic
def get_news_by_topic(topic):
    params = {
        'q': topic,  # search query
        'apiKey': API_KEY,  # API key
        'language': 'en',  # language of the news articles
        'sortBy': 'publishedAt',  # sort news by latest
        'pageSize': 5  # limit to 5 articles (you can adjust this)
    }

    # Send GET request to News API
    response = requests.get(BASE_URL, params=params)

    news_list = []  # This will store the news articles

    if response.status_code == 200:
        data = response.json()

        if 'articles' in data:
            # Loop through each article and store the relevant data in the list
            for article in data['articles']:
                title = article['title']
                description = article['description']
                published_at = article['publishedAt']
                content = article.get('content', 'No full content available.')
                url = article['url']

                # Convert the publishedAt timestamp to a more readable format
                published_at = datetime.strptime(published_at, '%Y-%m-%dT%H:%M:%SZ')
                formatted_time = published_at.strftime('%Y-%m-%d %H:%M:%S')

                # Create a dictionary with article details
                article_data = {
                    'title': title,
                    'description': description,
                    'publishedAt': formatted_time,
                    'content': content,  # Here, full content is either from API or scraped
                    'url': url
                }

                # Append the article data to the news_list
                news_list.append(article_data)

        else:
            print("No news articles found for this topic.")
    else:
        print(f"Failed to fetch news. Status code: {response.status_code}")

    # Return the list of news articles
    return news_list

# Function to process each article URL using Newspaper3k
def fetch_full_article_with_newspaper(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text  # Return the full content of the article
    except Exception as e:
        return f"Error occurred during parsing: {str(e)}"

# Function to summarize an article using Groq's Llama 3 model
def summarize_article(article_content):
    # Construct a prompt for summarization
    prompt = f"Please summarize the following article:\n\n{article_content}"

    # Request the model to summarize the article content
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama3-8b-8192",  # Specify the model
    )

    # Extract the summarized content from the response
    summary = chat_completion.choices[0].message.content.strip()
    return summary

# Function to split text into smaller chunks
def split_text(text, max_length=4000):
    # Split the text into chunks that are no longer than max_length
    chunks = []
    while len(text) > max_length:
        split_index = text.rfind(' ', 0, max_length)  # Split at a space close to max_length
        chunks.append(text[:split_index])
        text = text[split_index:].strip()
    chunks.append(text)  # Append the final chunk
    return chunks

# Main execution
topic = input("Enter the topic you want news for: ")
news_data = get_news_by_topic(topic)

combined_content = ""  # Variable to hold all article content combined

# Process each article URL and fetch the full content
for article in news_data:
    print(f"Processing article: {article['title']}")
    article_content = fetch_full_article_with_newspaper(article['url'])
    
    # Combine the content of each article
    combined_content += f"\n\n{article['title']}:\n{article_content}"

# Split the combined content into smaller chunks if it exceeds the limit
chunks = split_text(combined_content)

# Summarize each chunk separately and combine the summaries
final_summary = ""
for chunk in chunks:
    print(f"Summarizing a chunk of text...")
    chunk_summary = summarize_article(chunk)
    final_summary += chunk_summary + "\n\n"

# Print the final summary
print(f"\nFinal Summary of all articles on '{topic}':\n")
print(final_summary)
