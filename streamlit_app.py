import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from datetime import datetime
from bot_classifier import BotClassifier
from data_formatter import DataFormatter
from client import Client

# Page configuration
st.set_page_config(
    page_title="Bot Detector",
    page_icon="🤖",
    layout="wide"
)

# Title and description
st.title("🤖 Twitter Bot Detector")
st.write("Enter a Twitter username to analyze whether the account is likely a bot or human.")

# Initialize components
formatter = DataFormatter()
classifier = BotClassifier()
client = Client()

# API configuration
# Create input form
with st.form("username_form"):
    username = st.text_input(
        "Twitter Username", 
        placeholder="Enter username (without @)",
        help="Enter the Twitter username you want to analyze"
    )
    submit_button = st.form_submit_button("Analyze Account", type="primary")

# Process the form submission
if submit_button and username:
    # Remove @ if present
    username = username.replace('@', '').strip()
    
    if not username:
        st.error("Please enter a valid username.")
    else:
        with st.spinner(f"Analyzing @{username}..."):
            try:                
                response = client.get_twitter_data(username)
                if response.status_code == 200:
                    # Get data from API
                    data = response.json()
                    # Process data and calculate features
                    features = formatter.format_data(data)
                    # Classify with detailed information
                    result = classifier.predict_detailed(features)
                    
                    # Display results
                    st.success(f"Analysis completed for @{username}")
                    
                    # Main prediction result
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.subheader("🎯 Prediction Result")
                        is_human = result['prediction']['is_human']
                        confidence = result['prediction']['confidence']
                        
                        if is_human:
                            st.success(f"**HUMAN** (Confidence: {confidence:.1%})")
                        else:
                            st.error(f"**BOT** (Confidence: {confidence:.1%})")

                        # Model analysis

                        st.subheader("🔍 Model Analysis")
                        col4, col5, col6 = st.columns(3)
                        
                        with col4:
                            st.metric("Total Trees", result['model_analysis']['total_trees'])
                        
                        with col5:
                            st.metric("Human Votes", result['model_analysis']['human_votes'])
                            st.metric("Bot Votes", result['model_analysis']['bot_votes'])
                        
                        with col6:
                            st.metric("Vote Ratio", f"{result['model_analysis']['vote_ratio']:.1%}")
                            st.metric("Agreement Level", f"{result['model_analysis']['agreement_level']:.1%}")
                    
                    with col2:
                        st.subheader("📊 Account Statistics")
                        col7, col8 = st.columns(2)
                        with col7:
                            st.write(f"**Username**: {username}")
                            st.write(f"**Full Name**: {data['name']}")
                        with col8:
                            st.image(data['profile_image_url_https'], width=60)  
                        st.write(f"**Description**: {data['description']}")
                        st.write(f"**Location**: {data['location']}")
                        st.write(f"**Verified**: {data['verified']}")
                        try:
                            created_at_dt = datetime.strptime(data['created_at'], "%Y-%m-%dT%H:%M:%S.%fZ")
                            formatted_created_at = created_at_dt.strftime("%d-%m-%Y")
                        except Exception:
                            formatted_created_at = data['created_at']
                        st.write(f"**Created At**: {formatted_created_at}")
                        st.write(f"**Followers**: {features['followers_count']}")
                        st.write(f"**Following**: {features['friends_count']}")
                        st.write(f"**Account Age**: {features['account_age_days']} days")
                        st.write(f"**Average Tweets per Day**: {features['average_tweets_per_day']:.2f}")                  
                    
                    with col3:
                        # Feature importance
                        st.subheader("📈 Most Important Features")
                    
                        feature_df = result['feature_importance']['top_features']
                        
                        # Prepare data for heatmap
                        heatmap_features = [f['feature'] for f in feature_df]
                        heatmap_importances = [f['importance'] for f in feature_df]

                        df_heatmap = pd.DataFrame({
                            'Importance': heatmap_importances
                        }, index=heatmap_features)

                        # For Plotly Heatmap, index as x, single column as y
                        fig = go.Figure(
                            data=go.Heatmap(
                                z=[df_heatmap['Importance'].values],
                                x=df_heatmap.index,
                                y=['Importance'],
                                colorscale='OrRd',
                                colorbar=dict(title='Importance'),
                                text=[[f"{imp:.4f}" for imp in df_heatmap['Importance'].values]],
                                hoverinfo='x+z'
                            )
                        )
                        fig.update_layout(
                            yaxis=dict(showticklabels=False),
                            xaxis=dict(tickangle=45),
                            margin=dict(l=20, r=20, t=30, b=40),
                            height=280
                        )

                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Processed features (collapsible)
                    st.subheader("📈 Full Analysis")

                    with st.expander("🔧 All Processed Features"):
                        for key, value in features.items():
                            if isinstance(value, float):
                                st.write(f"**{key}**: {value:.4f}")
                            else:
                                st.write(f"**{key}**: {value}")
                
                else:
                    # Handle specific API errors with appropriate messages
                    if response.status_code == 402:
                        st.error("💳 **Payment Required**")
                        st.error("Not enough credits to perform this request. Please check your API subscription.")
                    elif response.status_code == 404:
                        st.error("👤 **User Not Found**")
                        st.error("The requested user does not exist or the username is invalid.")
                    elif response.status_code == 422:
                        st.error("⚠️ **Validation Failed**")
                        st.error("One of the required parameters was not provided or is invalid.")
                    elif response.status_code == 500:
                        st.error("🔧 **Internal Server Error**")
                        st.error("API internal error. The SocialData API failed to obtain the requested information. Please try again later.")
                    else:
                        st.error(f"❌ **API Error: {response.status_code}**")
                        st.error(f"Error details: {response.text}")
                    
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

elif submit_button and not username:
    st.warning("Please enter a username to analyze.")


