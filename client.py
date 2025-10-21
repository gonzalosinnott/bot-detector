import requests
import streamlit as st

class Client:
    def __init__(self):
        pass

    def get_twitter_data(self, username):

        API_KEY = st.secrets["API_KEY"]


        url = f'https://api.socialdata.tools/twitter/user/{username}'
        headers = {
            'Authorization': f'Bearer {API_KEY}',
            'Accept': 'application/json'
        }
        
        response = requests.get(url, headers=headers)

        return response
