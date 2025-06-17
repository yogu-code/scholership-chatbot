import streamlit as st
import requests
from markdown import markdown
import html

# Streamlit page configuration
st.set_page_config(page_title="Scholarship Finder", page_icon="🎓", layout="wide")

# Title and description
st.title("Scholarship Finder")
st.markdown("Enter your scholarship query (e.g., 'Mumbai undergraduate scholarships 2025') to find relevant financial aid options.")

# Input form
with st.form(key="search_form"):
    query = st.text_input("Search for scholarships:", placeholder="e.g., Mumbai undergraduate scholarships 2025")
    submit_button = st.form_submit_button(label="Search")

# Handle form submission
if submit_button:
    if not query.strip():
        st.error("Please enter a valid query.")
    else:
        with st.spinner("Searching for scholarships..."):
            try:
                # Send request to Flask backend
                response = requests.post("http://localhost:5000/search", json={"query": query})
                response.raise_for_status()  # Raise an error for bad status codes
                data = response.json()

                # Extract and render the markdown response
                markdown_response = data["response"]["markdown"]
                html_response = markdown(markdown_response)  # Convert markdown to HTML
                st.markdown(html_response, unsafe_allow_html=True)  # Render HTML in Streamlit

                # Optionally display the extracted intent
                with st.expander("View Intent Details"):
                    st.json(data["response"]["intent"])

            except requests.exceptions.RequestException as e:
                st.error(f"Failed to fetch results: {str(e)}. Please ensure the backend is running and try again.")

# Footer
st.markdown("---")
st.markdown("Powered by Streamlit | Backend by Flask | Data sourced via Google Custom Search API")