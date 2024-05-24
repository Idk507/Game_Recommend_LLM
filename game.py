import pandas as pd  # Pandas for data manipulation
import streamlit as st
from pandasai import SmartDataframe  # SmartDataframe for interacting with data using LLM
from pandasai.llm.local_llm import LocalLLM  # Importing LocalLLM for local Meta Llama 3 model

# Function to chat with CSV data
def chat_with_csv(df, query):
    # Initialize LocalLLM with Meta Llama 3 model
    llm = LocalLLM(
        api_base="http://localhost:11434/v1",
        model="llama3"
    )
    # Initialize SmartDataframe with DataFrame and LLM configuration
    pandas_ai = SmartDataframe(df, config={"llm": llm})
    # Chat with the DataFrame using the provided query
    result = pandas_ai.chat(query)
    return result

# Set layout configuration for the Streamlit page
#st.set_page_config(layout='wide')
# Set title for the Streamlit application
st.title("Games Data ChatApp powered by LLM")

# Load the CSV file directly
csv_file_path = 'games.csv'

# Check if the CSV file exists
if csv_file_path:
    # Load and display the CSV file
    st.info("CSV loaded successfully")
    data = pd.read_csv(csv_file_path)
    st.dataframe(data.head(3), use_container_width=True)

    # Enter the query for analysis
    st.info("Chat Below")
    input_text = st.text_area("Enter the query")

    # Perform analysis
    if input_text:
        if st.button("Chat with csv"):
            st.info("Your Query: " + input_text)
            result = chat_with_csv(data, input_text)
            st.success(result)
