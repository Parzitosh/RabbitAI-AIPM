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
st.sidebar.markdown("---")
st.sidebar.markdown("**Need a key?** Get one for free at [Google AI Studio](https://aistudio.google.com/).")

# File Uploader & Demo Data Fallback
st.write("### 1. Connect Your Data")
uploaded_file = st.file_uploader("Upload your standard sales CSV", type=["csv"])
use_demo = st.checkbox("Or use demo sales data (Zero-friction test)")

df = None
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
elif use_demo:
    # Auto-load the dummy dataset for evaluators
    df = pd.DataFrame({
        "Date": ["2023-01-15", "2023-01-16", "2023-02-10", "2023-02-15", "2023-03-05", "2023-03-20", "2023-04-12", "2023-04-18"],
        "Region": ["North", "South", "North", "East", "West", "South", "East", "West"],
        "Product": ["Widget A", "Widget B", "Widget C", "Widget A", "Widget B", "Widget A", "Widget C", "Widget A"],
        "SalesRep": ["Alice", "Bob", "Alice", "Charlie", "Diana", "Bob", "Charlie", "Diana"],
        "UnitsSold": [50, 30, 20, 60, 45, 55, 25, 40],
        "Revenue": [1500, 1200, 2000, 1800, 1800, 1650, 2500, 1200]
    })

if df is not None:
    st.write("### Data Preview")
    st.dataframe(df.head())
    
    st.write("### 2. Talk to Your Data")
    # Natural Language Query Input
    query = st.text_input("Ask a question (e.g., 'Which region had the highest revenue?'):")
    
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
            
            # Construct the prompt with data context
            context = f"Data Schema & Summary:\n{df.describe(include='all').to_string()}\n\nSample Data:\n{df.head(10).to_string()}\n\n"
            prompt = f"{system_prompt}\n{context}\nUser Question: {query}"
            
            with st.spinner("Talking to your data..."):
                try:
                    # 1. Get the Text Answer
                    response = model.generate_content(prompt)
                    
                    # Output in a clean Chat UI bubble
                    with st.chat_message("assistant", avatar="🐇"):
                        st.markdown("**Talking Rabbitt:**")
                        st.write(response.text)
                    
                    # 2. Automated Visualization
                    st.write("### Automated Insights")
                    
                    # Simple heuristic to grab columns for an automated chart
                    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
                    
                    if numeric_cols and cat_cols:
                        # Auto-plot the first numeric column grouped by the first categorical column
                        fig = px.bar(df, x=cat_cols[0], y=numeric_cols[0], 
                                     title=f"Distribution of {numeric_cols[0]} by {cat_cols[0]}",
                                     color=cat_cols[0])
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Visualization requires at least one text column and one number column.")
                        
                except Exception as e:
                    st.error(f"An error occurred: {e}. Check your API key and try again.")