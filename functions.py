from urlextract import URLExtract
from wordcloud import WordCloud
from gensim import corpora, models
extract = URLExtract()
 

def fetch_stats(selected_user, df):
    if selected_user != 'OverAll':
        df = df[df['user'] == selected_user]
        
        
    num_msgs = df.shape[0]
    words = []
    for msg in df['user']:
        words.extend(msg.split(' '))
        
    num_med = df[df['msg'] == '<Media omitted>\n'].shape[0]
        
    link = []
    for msg in df['msg']:
        link.extend(extract.find_urls(msg))
            
    return num_msgs, len(words), num_med, len(link)
    
    
def monthly_timeline(selected_user, df):
    if selected_user != 'OverAll':
        df = df[df['user'] == selected_user]
        
    timeline = df.groupby(['year', 'month'])['msg'].count().reset_index()
    
    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i] + '-' + str(timeline['year'][i]))
        
    timeline['time'] = time
    
    return timeline


def daily_timeline(selected_user, df):
    if selected_user != 'OverAll':
        df = df[df['user'] == selected_user]
        
    timeline = df.groupby(['date'])['msg'].count().reset_index()
    
    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['date'][i])
        
    timeline['time'] = time
    
    return timeline


def activity_map(selected_user, df):
    if selected_user != 'OverAll':
        df = df[df['user'] == selected_user]
        
    active_month_df = df.groupby('month')['msg'].count().reset_index()
    month_list = active_month_df['month'].tolist()
    month_msg_list = active_month_df['msg'].tolist()
    
    active_day_df = df.groupby('day')['msg'].count().reset_index()
    day_list = active_day_df['day'].tolist()
    day_msg_list = active_day_df['msg'].tolist()
    
    return active_month_df, month_list, month_msg_list, active_day_df, day_list, day_msg_list


def most_chaty(df):
    x = df['user'].value_counts().head()
    
    percent = round((df['user'].value_counts() / df.shape[0]) * 100, 2)
    return x, percent


def create_wordcloud(selected_user, df):
    if selected_user != 'OverAll':
        df = df[df['user'] == selected_user]
        
    wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')
    df_wc = wc.generate(df['msg'].str.cat(sep=' '))
    return df_wc


def topic_modeling(selected_user, df):
    if selected_user != 'OverAll':
        df = df[df['user'] == selected_user]

    # Tokenize the messages
    tokenized_msgs = [msg.split() for msg in df['msg']]

    # Create a dictionary representation of the documents.
    dictionary = corpora.Dictionary(tokenized_msgs)

    # Filter out tokens that appear in less than 5 documents and more than 50% of the documents.
    dictionary.filter_extremes(no_below=5, no_above=0.5)

    # Convert the dictionary into a bag-of-words representation.
    corpus = [dictionary.doc2bow(tokens) for tokens in tokenized_msgs]

    # Train the LDA model
    lda_model = models.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=15)

    # Get topics and their keywords
    topics = lda_model.print_topics(num_words=5)

    return topics
