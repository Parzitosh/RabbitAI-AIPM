import streamlit as st
import pandas as pd
import google.generativeai as genai
import plotly.express as px

# Configure the page
st.set_page_config(page_title="Talking Rabbitt MVP", layout="wide")
st.title("🐇 Talking Rabbitt: Conversational Data Layer")
st.markdown("Replace a 10-minute manual Excel filter with a 5-second conversation.")

# Sidebar for API Configuration
st.sidebar.header("Configuration")
api_key = st.sidebar.text_input("Enter your Gemini API Key:", type="password")

# File Uploader
uploaded_file = st.file_uploader("Upload your standard sales CSV", type=["csv"]) # 

if uploaded_file is not None:
    # Read the CSV
    df = pd.read_csv(uploaded_file)
    
    st.write("### Data Preview")
    st.dataframe(df.head())
    
    # Natural Language Query Input
    query = st.text_input("Ask a question about your data (e.g., 'Which region had the highest revenue in Q1?'):") # 
    
    if st.button("Ask Rabbitt"):
        if not api_key:
            st.error("Please enter your API key in the sidebar to continue.")
        elif not query:
            st.warning("Please enter a question.")
        else:
            # Configure the LLM
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-2.5-flash')
            
            # Define Strict Guardrails (The "RAG" constraint)
            system_prompt = """
            You are 'Talking Rabbitt', an AI data analyst for enterprise executives. 
            You have been provided with a statistical summary and sample rows of a dataset.
            
            CRITICAL INSTRUCTIONS:
            1. You must ONLY answer questions that can be derived from the provided data context.
            2. If the user asks a question that is NOT related to the data (e.g., general knowledge, coding help, casual chat, or questions about data not in the schema), you MUST refuse to answer.
            3. If you refuse, reply EXACTLY with: "I can only answer questions related to the uploaded dataset. Please ask a question about the metrics provided."
            4. Keep your analytical answers clear, accurate, and concise.
            """
            
            context = f"Data Schema & Summary:\n{df.describe(include='all').to_string()}\n\nSample Data:\n{df.head(10).to_string()}\n\n"
            prompt = f"{system_prompt}\n{context}\nUser Question: {query}"
                        
            with st.spinner("Talking to your data..."):
                try:
                    # 1. Get the Text Answer
                    response = model.generate_content(prompt)
                    st.success("### Answer")
                    st.write(response.text) # 
                    
                    # 2. Automated Visualization
                    st.write("### Automated Insights") # 
                    
                    # Simple heuristic to grab columns for an automated chart
                    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
                    
                    if numeric_cols and cat_cols:
                        # Auto-plot the first numeric column grouped by the first categorical column
                        fig = px.bar(df, x=cat_cols[0], y=numeric_cols[0], 
                                     title=f"Distribution of {numeric_cols[0]} by {cat_cols[0]}",
                                     color=cat_cols[0])
                        st.plotly_chart(fig, use_container_width=True) # 
                    else:
                        st.info("Visualization requires at least one text column and one number column.")
                        
                except Exception as e:
                    st.error(f"An error occurred: {e}")