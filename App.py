import streamlit as st
import preprocessor 
import functions
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


st.sidebar.title('WhatsApp Chat Analyzing')

uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode('utf-8')
    df = preprocessor.preprocess(data)
    
    df['sentiment'] = df['msg'].apply(lambda x: TextBlob(x).sentiment.polarity)
    
    positive_threshold = 0.2
    negative_threshold = -0.2
    
    df['sentiment_category'] = df['sentiment'].apply(lambda x: 'happy' if x > positive_threshold else ('sad' if x < negative_threshold else 'neutral'))
        
    user_details = df['user'].unique().tolist()
    if 'Group Notification' in user_details:
        user_details.remove('Group Notification')
    user_details.sort()
    user_details.insert(0, 'OverAll')
    
    selected_user = st.sidebar.selectbox('Show Analysis as:', user_details)
    
    if st.sidebar.button('Analyse'):
        num_msgs, words, num_med, link = functions.fetch_stats(selected_user, df)
        
        st.title('OverAll Basic Statistics')
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.header('Total Messages')
            st.subheader(num_msgs)
        with col2:
            st.header('Total Words')
            st.subheader(words)
        with col3:
            st.header('Media Shared')
            st.subheader(num_med)
        with col4:
            st.header('Link Shared')
            st.subheader(link)
            
            
        timeline = functions.monthly_timeline(selected_user, df)
        st.title('Monthly Timeline')
        
        fig, ax = plt.subplots()
        ax.plot(timeline['time'], timeline['msg'], color='maroon')
        plt.xticks(rotation=90)
        st.pyplot(fig)
        
        timeline = functions.daily_timeline(selected_user, df)
        st.title('Daily Timeline')
        
        fig, ax = plt.subplots()
        ax.plot(timeline['time'], timeline['msg'], color='purple')
        plt.xticks(rotation=90)
        st.pyplot(fig)
        
        st.title('Sentiment Analysis')
        st.line_chart(df.groupby('date')['sentiment_category'].value_counts().unstack().fillna(0))
        
        
         # Perform user profiling using K-means clustering
        if selected_user == 'OverAll':
            # Extract features for clustering
            user_features = df.groupby('user').agg({
                'msg': 'count',
                'sentiment': 'mean',
                'date': lambda x: (x.max() - x.min()).days + 1 if x.min() and x.max() else 0  # Number of days active

            }).reset_index()
            
            # Standardize features
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(user_features.drop(columns=['user']))
            
            # Perform K-means clustering
            kmeans = KMeans(n_clusters=3, random_state=42)
            user_features['cluster'] = kmeans.fit_predict(scaled_features)
            
            # Map sentiment category to labels
            user_features['sentiment_category'] = user_features['sentiment'].apply(lambda x: 'happy' if x > positive_threshold else ('sad' if x < negative_threshold else 'neutral'))
            
            # Map cluster numbers to cluster names
            cluster_mapping = {0: 'Cluster 1', 1: 'Cluster 2', 2: 'Cluster 3'}
            user_features['cluster'] = user_features['cluster'].map(cluster_mapping)
            
            # Display clustering results
            st.title('User Profiling')
            st.write(user_features)
            
            # Visualize clustering results
            st.title('Cluster Distribution')
            st.bar_chart(user_features['cluster'].value_counts())
        
        
        st.title('Activity Map')
        col1, col2 = st.columns(2)
        
        active_month_df, month_list, month_msg_list, active_day_df, day_list, day_msg_list = functions.activity_map(selected_user, df)
        with col1:
            st.header('Most Active Month')
            fig, ax = plt.subplots()
            ax.bar(active_month_df['month'], active_month_df['msg'])
            ax.bar(month_list[month_msg_list.index(max(month_msg_list))], max(month_msg_list), color='green', label = 'Highest')
            ax.bar(month_list[month_msg_list.index(min(month_msg_list))], min(month_msg_list), color='red', label = 'Lowest')
            plt.xticks(rotation=90)
            st.pyplot(fig)
            
        with col2:
            st.header('Most Active Day')
            fig, ax = plt.subplots()
            ax.bar(active_day_df['day'], active_day_df['msg'])
            ax.bar(day_list[day_msg_list.index(max(day_msg_list))], max(day_msg_list), color='green', label = 'Highest')
            ax.bar(day_list[day_msg_list.index(min(day_msg_list))], min(day_msg_list), color='red', label = 'Lowest')
            plt.xticks(rotation=90)
            st.pyplot(fig)
            
            
        if selected_user == 'OverAll':
            st.title('Most Active Users')
            
            x, percent = functions.most_chaty(df)
            fig, ax = plt.subplots()
            
            col1, col2 = st.columns(2)
            with col1:
                ax.bar(x.index, x, color='cyan')
                st.pyplot(fig)
                
            with col2:
                st.dataframe(percent)
                
                
        df_wc = functions.create_wordcloud(selected_user, df)
        st.title('Most Common Words')
        fig, ax = plt.subplots()
        ax.imshow(df_wc)
        st.pyplot(fig)
        
        
        st.title('Topic Modeling')
        topics = functions.topic_modeling(selected_user, df)
        for idx, topic in topics:
            st.subheader(f'Topic {idx}:')
            st.write(topic)
