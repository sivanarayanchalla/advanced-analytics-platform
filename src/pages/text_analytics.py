import streamlit as st
import pandas as pd
import plotly.express as px

def show_text_analytics(analytics):
    """Text Analytics page"""
    st.header("📝 Text Analytics")
    
    if 'current_data' not in st.session_state or st.session_state['current_data'] is None:
        st.warning("📁 No data loaded yet!")
        st.info("Please go to **Data Management** page to upload your data first.")
        return
    
    df = st.session_state['current_data']
    
    st.info("📝 Text Analytics - Analyze text data for insights and patterns")
    
    # Find text columns
    text_columns = []
    for col in df.columns:
        if df[col].dtype == 'object' and df[col].str.len().mean() > 10:  # Likely text if average length > 10 chars
            text_columns.append(col)
    
    if text_columns:
        st.success(f"✅ Found potential text columns: {', '.join(text_columns)}")
        
        # Let user select text column
        text_column = st.selectbox("Select Text Column:", text_columns)
        
        # Text analysis options
        st.subheader("🔍 Text Analysis Options")
        
        analysis_type = st.selectbox(
            "Select Analysis Type:",
            ["Text Overview", "Word Frequency", "Text Length Analysis"]
        )
        
        if analysis_type == "Text Overview":
            # Basic text statistics
            st.subheader("📊 Text Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_documents = len(df)
                st.metric("Total Documents", total_documents)
            
            with col2:
                avg_length = df[text_column].str.len().mean()
                st.metric("Avg Text Length", f"{avg_length:.0f} chars")
            
            with col3:
                total_words = df[text_column].str.split().str.len().sum()
                st.metric("Total Words", total_words)
            
            with col4:
                unique_words = len(set(' '.join(df[text_column].dropna()).split()))
                st.metric("Unique Words", unique_words)
            
            # Sample texts
            st.subheader("📝 Sample Texts")
            st.dataframe(df[[text_column]].head(10), width='stretch')
            
        elif analysis_type == "Word Frequency":
            # Simple word frequency analysis
            from collections import Counter
            import string
            
            st.subheader("🔤 Word Frequency Analysis")
            
            # Get all words
            all_text = ' '.join(df[text_column].dropna().astype(str))
            words = [word.lower().strip(string.punctuation) for word in all_text.split() if word.strip(string.punctuation)]
            
            # Count word frequencies
            word_freq = Counter(words)
            top_words = word_freq.most_common(20)
            
            # Create word frequency chart
            words_df = pd.DataFrame(top_words, columns=['Word', 'Frequency'])
            fig = px.bar(
                words_df,
                x='Frequency',
                y='Word',
                title="Top 20 Most Frequent Words",
                orientation='h'
            )
            st.plotly_chart(fig, use_container_width=True)
            
        elif analysis_type == "Text Length Analysis":
            st.subheader("📏 Text Length Distribution")
            
            # Calculate text lengths
            df['text_length'] = df[text_column].str.len()
            
            # Plot length distribution
            fig = px.histogram(
                df,
                x='text_length',
                title="Distribution of Text Lengths",
                nbins=30
            )
            st.plotly_chart(fig, use_container_width=True)
            
    else:
        st.warning("No suitable text columns found. Text analysis requires columns with substantial text content.")
        
    st.markdown("---")
    st.subheader("🚀 Advanced NLP Features Coming Soon")
    st.info("""
    **Planned Text Analytics Features:**
    - **Sentiment Analysis**: Positive/negative/neutral sentiment scoring
    - **Topic Modeling**: Discover hidden topics in text collections
    - **Named Entity Recognition**: Extract people, places, organizations
    - **Text Classification**: Automatically categorize text documents
    - **Keyword Extraction**: Identify important keywords and phrases
    - **Text Summarization**: Generate summaries of long texts
    """)