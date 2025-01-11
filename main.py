import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
from groq import Groq
import ast
import os
from dotenv import load_dotenv
import pandas as pd
from cassandra.cluster import Cluster


SECURE_CONNECT_BUNDLE = 'secure-connect-supermind.zip'


ASTRA_DB_KEYSPACE = 'default_keyspace'  
ASTRA_DB_TABLE = 'parth_collection'  

def connect_to_astra_db(bundle_path):
    """Connect to Astra DB using the secure connect bundle and return a session object."""
    cluster = Cluster(cloud={'secure_connect_bundle': bundle_path})
    session = cluster.connect()
    return session

def fetch_data_from_astra_db(session, keyspace, table_name):
    """Fetch data from Astra DB and return it as a Pandas DataFrame."""
    session.set_keyspace(keyspace)
    query = f"SELECT * FROM {table_name}"
    rows = session.execute(query)

    # Convert rows to a list of dictionaries
    data = [dict(row._asdict()) for row in rows]

    # Convert to Pandas DataFrame
    df = pd.DataFrame(data)
    return df


load_dotenv()
groq_key = os.getenv('groq_key')

if "messages" not in st.session_state:
    st.session_state.messages = []

st.set_page_config(
    page_title="Team Hackaholics",
    page_icon="data\\favicon.ico",
    layout="wide",  
    initial_sidebar_state="expanded",  
)

def run_llm(user_message):
    client = Groq(api_key=groq_key)

    model = "llama-3.3-70b-versatile"
    temperature = 0.7
    
    response = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "user", "content": user_message}
    ],
    temperature=temperature
    )
    
    return (response.choices[0].message.content)

def user_message(user_query):
    user_message = (
        f"""
            Given the following prompt, identify the mentioned content classes from the list: 'Reels', 'Carousel', 'Static Images', and 'Videos'. Return the identified classes as a list of strings, without any extra text. The prompt can mention one or more classes, but only return the ones that are mentioned.

            Example 1: Prompt: 'I am creating a new social media post with a static image and a carousel.' Response: ['Static Images', 'Carousel']

            Example 2: Prompt: 'This post will be a reel featuring a video.' Response: ['Reels', 'Videos']

            Now, process the following prompt: Prompt: '{user_query}'

            Response:"""
                    )
    
    return user_message

def single_class(df, post_type, question):
    filtered_data = df[df['Post_Type'] == post_type]
    
    # Calculate the averages of the other columns
    avg_likes = filtered_data['Number_of_Likes'].mean()
    avg_shares = filtered_data['Number_of_Shares'].mean()
    avg_comments = filtered_data['Number_of_Comments'].mean()
    avg_engagement = filtered_data['Engagement_Score'].mean()

    prompt_template = (f"""Provide a summary of the average metrics for {post_type} posts.\n
             Average Likes: {avg_likes}\n
             Average Shares: {avg_shares}\n
             Average Comments: {avg_comments}\n
             Average Engagement Score: {avg_engagement}\n
             Provide a response in a professional tone.""")
    
    output = run_llm(user_message=prompt_template)

    return output

def double_class(df, post_type_1,post_type_2,query):
    filtered_data_1 = df[df['Post_Type'] == post_type_1]
    
    # Calculate the averages of the other columns
    avg_likes_1 = filtered_data_1['Number_of_Likes'].mean()
    avg_shares_1 = filtered_data_1['Number_of_Shares'].mean()
    avg_comments_1 = filtered_data_1['Number_of_Comments'].mean()
    avg_engagement_1 = filtered_data_1['Engagement_Score'].mean()

    filtered_data_2 = df[df['Post_Type'] == post_type_1]
    
    # Calculate the averages of the other columns
    avg_likes_2 = filtered_data_2['Number_of_Likes'].mean()
    avg_shares_2 = filtered_data_2['Number_of_Shares'].mean()
    avg_comments_2 = filtered_data_2['Number_of_Comments'].mean()
    avg_engagement_2 = filtered_data_2['Engagement_Score'].mean()

    prompt_template = (f"""The user has asked to compare two types of posts: {post_type_1} and {post_type_2}. 

                    For each post type, here are the average metrics:
                    - Average Likes: {avg_likes_1} for {post_type_1} and {avg_likes_2} for {post_type_2}
                    - Average Shares: {avg_shares_1} for {post_type_1} and {avg_shares_2} for {post_type_2}
                    - Average Comments: {avg_comments_1} for {post_type_1} and {avg_comments_2} for {post_type_2}
                    - Average Engagement Score: {avg_engagement_1} for {post_type_1} and {avg_engagement_2} for {post_type_2}

                    Based on this data, provide a comparison of these two post types. Discuss which post type performs better or has higher engagement on average across these metrics. Additionally, consider any trends or patterns you observe that might highlight differences between the two post types. Be sure to provide a clear and concise comparison.

                    The user query is: "{query}"

                    Please consider how this query might influence the comparison between the two post types.
                    """)
    output = run_llm(user_message=prompt_template)

    return output

def upload_file():
    
    session = connect_to_astra_db(SECURE_CONNECT_BUNDLE)
    df = fetch_data_from_astra_db(session, ASTRA_DB_KEYSPACE, ASTRA_DB_TABLE)
    st.subheader("👀 Data Preview:")
    st.write(df.head(10))

    return df
    
def plot_engagement(df):
    st.subheader("📈 Engagement Metrics")

    if df is not None:
        post_types = df['Post_Type'].unique()

        # Average likes, shares, and comments per post type
        avg_metrics = df.groupby('Post_Type')[['Number_of_Likes', 'Number_of_Shares', 'Number_of_Comments']].mean().reset_index()

        # Bar plot for engagement metrics with a consistent size
        fig, ax = plt.subplots(figsize=(6,5.5))  # Adjusted size for better proportion
        avg_metrics.set_index('Post_Type').plot(kind='bar', ax=ax, color=plt.get_cmap("Set2").colors[:len(post_types)])
        ax.set_title('Average Engagement Metrics by Post Type', fontsize=14, fontweight='bold')
        ax.set_ylabel('Average Count', fontsize=12, fontweight='regular')
        ax.set_xlabel('Post Type', fontsize=12, fontweight='regular')

        return fig

def plot_post_type_distribution(df):
    st.subheader("📈 Post Type Distribution")

    if df is not None:
        post_type_counts = df['Post_Type'].value_counts()

        # Pie chart for post type distribution using matplotlib with consistent size
        fig, ax = plt.subplots(figsize=(6, 6))  # Same size as bar chart for consistent proportion
        post_type_counts.plot.pie(
            autopct='%1.1f%%',
            ax=ax,
            colors=plt.get_cmap("Set2").colors[:len(post_type_counts)],  # Use matplotlib's Set2 colormap
            textprops={'fontsize': 12, 'color': 'black'}  # Adjusted fontsize for consistency
        )

        # Customize the title and labels to match the bar chart style
        ax.set_ylabel('')  # Remove the ylabel for a cleaner look
        ax.set_title('Distribution of Post Types', fontsize=14, fontweight='bold')

        return fig

def show_chat(r_container):
    
    with r_container:
        if st.session_state.messages == []:
            return
        
        for message in st.session_state.messages:
            
            
            
            if message["role"] == "user":
                
                st.markdown(f"""
                    <div style="text-align:right; padding:10px;">
                        <div style="display:inline-block; background-color:#CF6BA9; padding:8px; border-radius:8px;">
                            {message['content']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                    # Assistant messages aligned to the left
                st.markdown(f"""
                    <div style="text-align:left; padding:10px;">
                        <div style="display:inline-block; background-color:#1338BE; padding:10px; border-radius:10px;">
                            {message['content']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

def chatbot(df):
    st.subheader("🤖 Chatbot Interface")
    output = None

    question = st.text_input("Ask something about the data:")
    response_container = st.container(height=600)
    
    st.session_state.messages.append({"role": "user", "content": question})
    if question:
        classes = run_llm(user_message=user_message(question))
        classes = ast.literal_eval(classes)

        if len(classes) == 1:
            post_type = classes[0]
            output = single_class(df, post_type,question)
            st.session_state.messages.append({"role": "assistant", "content": output})
            show_chat(response_container)

        else:
            post_type_1 = classes[0]
            post_type_2 = classes[1]
            output = double_class(df, post_type_1,post_type_2,question)
            st.session_state.messages.append({"role": "assistant", "content": output})
            show_chat(response_container)

def main():
    
    st.title(f" Social Media Engagement Analytics")
    
    df = upload_file()

    if df is not None:
        col1, col2 = st.columns(2)
        with col1:
            chatbot(df)
                
        with col2:
            bar_chart = plot_engagement(df)
            graph = st.container(height=630)
            graph.pyplot(bar_chart)

if __name__ == "__main__":
    main()
