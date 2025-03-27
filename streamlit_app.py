import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
import time
from streamlit_lottie import st_lottie
import json
import requests

# Page configuration
st.set_page_config(
    page_title="Political Polarization via LLMs",
    page_icon="🗳️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; color: #4F8BF9;}
    .sub-header {font-size: 1.5rem; margin-bottom: 1rem;}
    .model-card {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #f8f9fa;
        margin-bottom: 1rem;
        border-left: 4px solid #4F8BF9;
    }
    .highlight {background-color: #ffffcc; padding: 0.2rem;}
    .stButton button {background-color: #4F8BF9; color: white;}
    .chart-container {margin-top: 2rem;}
    .comment-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f1f3f5;
        margin-bottom: 1rem;
        border-left: 4px solid #adb5bd;
    }
    .republican {border-left: 4px solid #ff6b6b;}
    .democrat {border-left: 4px solid #339af0;}
    .independent {border-left: 4px solid #20c997;}
    .model-comparison {
        display: flex;
        flex-wrap: wrap;
        gap: 1rem;
    }
    .model-result {
        flex: 1;
        min-width: 200px;
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8f9fa;
        border-left: 4px solid #4F8BF9;
    }
</style>
""", unsafe_allow_html=True)

# Load Lottie animation
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Lottie animations
lottie_analysis = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_qp1q7mct.json")
lottie_success = load_lottieurl("https://assets4.lottiefiles.com/packages/lf20_jR229R.json")

# Sidebar
with st.sidebar:
    st.title("🗳️ Political Analysis")
    st.markdown("---")
    
    st.markdown("### About")
    st.info(
        """
        This tool analyzes political stances and news channel preferences using predictions from multiple LLMs like GPT-4o, Llama, Mistral, and Copilot based on YouTube comment data.
        
        **Features:**
        - Analyze your own text
        - Try interactive examples
        - Compare multiple LLM predictions
        - Visualize political distributions
        """
    )
    
    st.markdown("### Models Used")
    st.markdown("- GPT-4o-mini")
    st.markdown("- Mistral")
    st.markdown("- Copilot")
    st.markdown("- Llama 3.2 70B")
    
    # Interactive model weights
    st.markdown("### Customize Model Weights")
    st.caption("Adjust how much you trust each model")
    gpt_weight = st.slider("GPT-4o-mini", 0.0, 1.0, 0.8, 0.1)
    mistral_weight = st.slider("Mistral", 0.0, 1.0, 0.7, 0.1)
    copilot_weight = st.slider("Copilot", 0.0, 1.0, 0.6, 0.1)
    llama_weight = st.slider("Llama 3.2", 0.0, 1.0, 0.5, 0.1)
    
    st.markdown("---")
    st.markdown("Made by Sai Rupa Jhade")

# Main content
st.markdown('<p class="main-header">🗳️ Political Polarization Analysis with LLMs</p>', unsafe_allow_html=True)
st.markdown("""
Analyze political stances and news channel preferences using predictions from multiple Large Language Models 
based on YouTube comment data. Enter your own text, try interactive examples, or compare models.
""")

# Create tabs
tab1, tab2, tab3 = st.tabs(["Text Analysis", "Interactive Examples", "Model Comparison"])

# Static sample data (can be replaced with real data or API calls)
model_predictions = {
    "GPT-4o-mini": {
        "political_party": "republican",
        "party_probabilities": {"republican": 75.2, "democrat": 15.8, "independent": 9.0},
        "preferred_channel": "Fox News",
        "explanation": "User used strong conservative language, consistent with Republican views.",
        "confidence": 75.2
    },
    "Mistral": {
        "political_party": "republican",
        "party_probabilities": {"republican": 68.5, "democrat": 18.3, "independent": 13.2},
        "preferred_channel": "Fox News",
        "explanation": "Detected clear alignment with Republican ideology through comment sentiment.",
        "confidence": 68.5
    },
    "Copilot": {
        "political_party": "democrat",
        "party_probabilities": {"republican": 32.1, "democrat": 58.7, "independent": 9.2},
        "preferred_channel": "MSNBC",
        "explanation": "Language patterns suggest progressive viewpoints more aligned with Democratic positions.",
        "confidence": 58.7
    },
    "Llama 3.2 70B": {
        "political_party": "independent",
        "party_probabilities": {"republican": 42.3, "democrat": 39.8, "independent": 17.9},
        "preferred_channel": "ABC News",
        "explanation": "Mixed political signals without clear partisan alignment in the language used.",
        "confidence": 42.3
    }
}

# Sample comments for demonstration
sample_comments = [
    "The economy is doing great under this administration!",
    "We need to secure our borders and protect American jobs.",
    "Healthcare should be a right for all citizens.",
    "Lower taxes will help stimulate economic growth.",
    "Climate change is the biggest threat we face today.",
    "The second amendment rights should not be infringed.",
    "We need more funding for public education.",
    "Government regulation is hurting small businesses.",
    "Income inequality is a serious problem we must address.",
    "Strong military is essential for our national security."
]

# Function to simulate analysis with some randomness
def analyze_text(text, model):
    # Add loading animation
    with st.spinner(f"🧠 {model} is analyzing..."):
        # Simulate processing time
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.01)  # Faster for demo purposes
            progress_bar.progress(i + 1)
        
        # Base result from the model
        base_result = model_predictions[model]
        
        # Add some randomness based on text content
        republican_keywords = ["taxes", "borders", "jobs", "amendment", "military", "regulation"]
        democrat_keywords = ["healthcare", "climate", "inequality", "education", "rights"]
        
        text_lower = text.lower()
        
        # Count keywords
        rep_count = sum(1 for word in republican_keywords if word in text_lower)
        dem_count = sum(1 for word in democrat_keywords if word in text_lower)
        
        # Adjust probabilities based on keywords
        party_probs = dict(base_result["party_probabilities"])
        party_probs["republican"] += rep_count * 5 - dem_count * 3
        party_probs["democrat"] += dem_count * 5 - rep_count * 3
        
        # Add randomness
        party_probs = {
            k: max(min(v + random.uniform(-10, 10), 100), 0) 
            for k, v in party_probs.items()
        }
        
        # Normalize probabilities
        total = sum(party_probs.values())
        party_probs = {k: (v/total)*100 for k, v in party_probs.items()}
        
        # Determine highest probability party
        max_party = max(party_probs.items(), key=lambda x: x[1])[0]
        
        # Determine preferred channel based on party
        channel_probs = {
            "republican": {"Fox News": 0.6, "NewsMax": 0.3, "ABC News": 0.1},
            "democrat": {"CNN": 0.5, "MSNBC": 0.4, "ABC News": 0.1},
            "independent": {"ABC News": 0.6, "CNN": 0.2, "Fox News": 0.2}
        }
        
        channels = list(channel_probs[max_party].keys())
        channel_weights = list(channel_probs[max_party].values())
        preferred_channel = random.choices(channels, weights=channel_weights)[0]
        
        # Generate explanation
        explanations = {
            "republican": [
                f"Analysis detected {rep_count} conservative-leaning terms, consistent with Republican views.",
                "Comment sentiment aligns with traditional conservative values.",
                "Word choice and phrasing match patterns commonly seen in right-leaning discourse."
            ],
            "democrat": [
                f"Analysis detected {dem_count} progressive-leaning terms, typical of Democratic rhetoric.",
                "Comment shows concern for social issues often championed by Democrats.",
                "Language patterns match those commonly found in left-leaning discourse."
            ],
            "independent": [
                "Comment shows a balanced perspective without strong partisan language.",
                "Analysis found mixed political signals without clear partisan alignment.",
                "Language doesn't strongly correlate with either major party's typical rhetoric."
            ]
        }
        
        explanation = random.choice(explanations[max_party])
        
        result = {
            "political_party": max_party,
            "party_probabilities": party_probs,
            "preferred_channel": preferred_channel,
            "explanation": explanation,
            "confidence": party_probs[max_party]
        }
        
        # Clear progress bar
        progress_bar.empty()
        
        return result

# Tab 1: Text Analysis
with tab1:
    st.markdown('<p class="sub-header">✍️ Analyze Your Text</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        user_text = st.text_area(
            "Enter a comment or statement to analyze:",
            height=100,
            placeholder="Example: The economy is doing great under this administration!"
        )
    
    with col2:
        st_lottie(lottie_analysis, height=150, key="analysis_animation")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        model_choice = st.selectbox(
            "🤖 Choose a Language Model",
            options=list(model_predictions.keys())
        )
        
        analyze_button = st.button("🔍 Analyze Text", use_container_width=True)
    
    with col2:
        compare_models = st.checkbox("Compare all models")
        show_keywords = st.checkbox("Highlight political keywords")
    
    if user_text and analyze_button:
        # Highlight keywords if option is selected
        highlighted_text = user_text
        if show_keywords:
            republican_keywords = ["taxes", "borders", "jobs", "amendment", "military", "regulation"]
            democrat_keywords = ["healthcare", "climate", "inequality", "education", "rights"]
            
            for keyword in republican_keywords:
                if keyword in user_text.lower():
                    highlighted_text = highlighted_text.replace(
                        keyword, f"<span style='background-color: #ffcccc; padding: 0.2rem;'>{keyword}</span>"
                    )
            
            for keyword in democrat_keywords:
                if keyword in user_text.lower():
                    highlighted_text = highlighted_text.replace(
                        keyword, f"<span style='background-color: #cce5ff; padding: 0.2rem;'>{keyword}</span>"
                    )
            
            st.markdown(f"""
            <div class="comment-box">
                <p><strong>Input Text (with highlighted keywords):</strong></p>
                <p>{highlighted_text}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="comment-box">
                <p><strong>Input Text:</strong> {user_text}</p>
            </div>
            """, unsafe_allow_html=True)
        
        if compare_models:
            # Show success animation
            st_lottie(lottie_success, height=150, key="success")
            
            st.markdown("### 📊 Model Comparison")
            
            # Analyze with all models
            results = {}
            for model in model_predictions.keys():
                results[model] = analyze_text(user_text, model)
            
            # Display results in a grid
            results_container = st.container()
            with results_container:
                st.markdown('<div class="model-comparison">', unsafe_allow_html=True)
                
                for model, result in results.items():
                    # Determine CSS class based on predicted party
                    party_class = result["political_party"]
                    
                    st.markdown(f'''
                    <div class="model-result {party_class}">
                        <h4>{model}</h4>
                        <p><strong>Political Party:</strong> {result["political_party"].capitalize()}</p>
                        <p><strong>Confidence:</strong> {result["confidence"]:.1f}%</p>
                        <p><strong>Channel:</strong> {result["preferred_channel"]}</p>
                    </div>
                    ''', unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Create a bar chart comparing model predictions
            st.markdown("### 📈 Party Probability Comparison")
            
            fig, ax = plt.subplots(figsize=(10, 5))
            models = list(results.keys())
            
            # Extract confidence values for each party across models
            rep_values = [pred["party_probabilities"]["republican"] for pred in results.values()]
            dem_values = [pred["party_probabilities"]["democrat"] for pred in results.values()]
            ind_values = [pred["party_probabilities"].get("independent", 0) for pred in results.values()]
            
            x = range(len(models))
            width = 0.25
            
            ax.bar([i - width for i in x], rep_values, width, label='Republican', color='#ff6b6b')
            ax.bar(x, dem_values, width, label='Democrat', color='#339af0')
            ax.bar([i + width for i in x], ind_values, width, label='Independent', color='#20c997')
            
            ax.set_ylabel('Probability (%)')
            ax.set_title('Political Party Probability by Model')
            ax.set_xticks(x)
            ax.set_xticklabels(models)
            ax.legend()
            
            st.pyplot(fig)
            
            # Calculate weighted consensus based on user's model weights
            st.markdown("### 🧠 Weighted Consensus")
            
            weights = {
                "GPT-4o-mini": gpt_weight,
                "Mistral": mistral_weight,
                "Copilot": copilot_weight,
                "Llama 3.2 70B": llama_weight
            }
            
            # Normalize weights
            total_weight = sum(weights.values())
            weights = {k: v/total_weight for k, v in weights.items()}
            
            # Calculate weighted probabilities
            weighted_probs = {
                "republican": sum(results[model]["party_probabilities"]["republican"] * weights[model] for model in models),
                "democrat": sum(results[model]["party_probabilities"]["democrat"] * weights[model] for model in models),
                "independent": sum(results[model]["party_probabilities"].get("independent", 0) * weights[model] for model in models)
            }
            
            # Determine consensus party
            consensus_party = max(weighted_probs.items(), key=lambda x: x[1])[0]
            
            st.markdown(f"""
            <div class="model-card {consensus_party}">
                <h3>Weighted Consensus</h3>
                <p>Based on your model weight preferences, the consensus analysis is:</p>
                <p><strong>Political Party:</strong> {consensus_party.capitalize()}</p>
                <p><strong>Confidence:</strong> {weighted_probs[consensus_party]:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Visualize weighted consensus
            fig, ax = plt.subplots(figsize=(8, 5))
            parties = list(weighted_probs.keys())
            probs = list(weighted_probs.values())
            
            colors = ["#ff6b6b", "#339af0", "#20c997"]
            ax.bar(parties, probs, color=colors[:len(parties)])
            ax.set_ylabel("Probability (%)")
            ax.set_title("Weighted Consensus - Party Probability Distribution")
            
            for i, v in enumerate(probs):
                ax.text(i, v + 1, f"{v:.1f}%", ha='center')
            
            st.pyplot(fig)
            
        else:
            # Single model analysis
            result = analyze_text(user_text, model_choice)
            
            # Show success animation
            st_lottie(lottie_success, height=150, key="success")
            
            st.markdown("### 🧠 Prediction Result")
            
            # Determine CSS class based on predicted party
            party_class = result["political_party"]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"### {model_choice} Analysis")
                st.markdown(f"**Political Party:** {result['political_party'].capitalize()}")
                st.markdown(f"**Preferred Channel:** {result['preferred_channel']}")
                st.markdown(f"**Confidence:** {result['confidence']:.1f}%")
                st.markdown(f"**Explanation:** {result['explanation']}")
                
                # Add a sentiment gauge
                sentiment_value = result['confidence'] / 100
                st.markdown("### Sentiment Strength")
                st.progress(sentiment_value)
                
                if sentiment_value > 0.7:
                    st.info("Strong political sentiment detected")
                elif sentiment_value > 0.4:
                    st.info("Moderate political sentiment detected")
                else:
                    st.info("Weak political sentiment detected")
            
            with col2:
                # Party probability chart
                fig, ax = plt.subplots(figsize=(8, 5))
                parties = list(result["party_probabilities"].keys())
                probs = [result["party_probabilities"][p] for p in parties]
                
                colors = ["#ff6b6b", "#339af0", "#20c997"]
                ax.bar(parties, probs, color=colors[:len(parties)])
                ax.set_ylabel("Probability (%)")
                ax.set_title(f"{model_choice} - Party Probability Distribution")
                
                for i, v in enumerate(probs):
                    ax.text(i, v + 1, f"{v:.1f}%", ha='center')
                
                st.pyplot(fig)
                
                # Add a radar chart for more detailed analysis
                categories = ['Economic', 'Social', 'Foreign Policy', 'Environmental', 'Constitutional']
                
                # Generate random values based on the party
                if result['political_party'] == 'republican':
                    values = [70 + random.uniform(-10, 10), 30 + random.uniform(-10, 10), 
                             80 + random.uniform(-10, 10), 20 + random.uniform(-10, 10), 
                             75 + random.uniform(-10, 10)]
                elif result['political_party'] == 'democrat':
                    values = [40 + random.uniform(-10, 10), 80 + random.uniform(-10, 10), 
                             50 + random.uniform(-10, 10), 85 + random.uniform(-10, 10), 
                             40 + random.uniform(-10, 10)]
                else:
                    values = [50 + random.uniform(-10, 10), 50 + random.uniform(-10, 10), 
                             50 + random.uniform(-10, 10), 50 + random.uniform(-10, 10), 
                             50 + random.uniform(-10, 10)]
                
                # Create radar chart
                fig = plt.figure(figsize=(6, 6))
                ax = fig.add_subplot(111, polar=True)
                
                # Compute angle for each category
                angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
                values.append(values[0])  # Close the loop
                angles.append(angles[0])  # Close the loop
                categories.append(categories[0])  # Close the loop
                
                ax.plot(angles, values, 'o-', linewidth=2, color=colors[parties.index(result['political_party'])])
                ax.fill(angles, values, alpha=0.25, color=colors[parties.index(result['political_party'])])
                ax.set_thetagrids(np.degrees(angles[:-1]), categories[:-1])
                ax.set_ylim(0, 100)
                ax.set_title('Political Stance by Category')
                ax.grid(True)
                
                st.pyplot(fig)

# Tab 2: Interactive Examples
with tab2:
    st.markdown('<p class="sub-header">🎮 Interactive Examples</p>', unsafe_allow_html=True)
    
    st.markdown("""
    Explore how different language models analyze various political statements. 
    Select a comment from the list or try the random comment generator.
    """)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        example_option = st.radio(
            "Choose an option:",
            ["Select from examples", "Generate random comment"]
        )
        
        if example_option == "Select from examples":
            selected_comment = st.selectbox(
                "Select a comment to analyze:",
                options=sample_comments
            )
        else:
            if st.button("🎲 Generate Random Comment", use_container_width=True):
                selected_comment = random.choice(sample_comments)
                st.session_state.random_comment = selected_comment
            else:
                selected_comment = st.session_state.get('random_comment', sample_comments[0])
            
            st.markdown(f"""
            <div class="comment-box">
                <p><strong>Generated Comment:</strong> {selected_comment}</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        example_model = st.selectbox(
            "Choose model:",
            options=list(model_predictions.keys())
        )
    
    # Interactive visualization of model predictions
    if st.button("🔍 Analyze Example", use_container_width=True):
        # Analyze with selected model
        result = analyze_text(selected_comment, example_model)
        
        # Show success animation
        st_lottie(lottie_success, height=150, key="example_success")
        
        # Display result
        st.markdown("### 🧠 Example Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Determine CSS class based on predicted party
            party_class = result["political_party"]
            
            st.markdown(f"""
            <div class="model-card {party_class}">
                <h3>{example_model} Analysis</h3>
                <p><strong>Comment:</strong> {selected_comment}</p>
                <p><strong>Political Party:</strong> {result['political_party'].capitalize()}</p>
                <p><strong>Preferred Channel:</strong> {result['preferred_channel']}</p>
                <p><strong>Confidence:</strong> {result['confidence']:.1f}%</p>
                <p><strong>Explanation:</strong> {result['explanation']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Add interactive elements
            st.markdown("### 🔄 Try Different Variations")
            
            variation_slider = st.slider(
                "Adjust comment intensity:",
                min_value=0.5,
                max_value=1.5,
                value=1.0,
                step=0.1
            )
            
            if variation_slider != 1.0:
                # Adjust probabilities based on intensity
                adjusted_probs = {
                    k: min(100, max(0, v * variation_slider)) 
                    for k, v in result["party_probabilities"].items()
                }
                
                # Normalize
                total = sum(adjusted_probs.values())
                adjusted_probs = {k: (v/total)*100 for k, v in adjusted_probs.items()}
                
                # Determine new highest probability party
                new_party = max(adjusted_probs.items(), key=lambda x: x[1])[0]
                
                st.markdown(f"""
                <div class="model-card {new_party}">
                    <h4>Adjusted Analysis (Intensity: {variation_slider}x)</h4>
                    <p><strong>Political Party:</strong> {new_party.capitalize()}</p>
                    <p><strong>Confidence:</strong> {adjusted_probs[new_party]:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            # Party probability chart
            fig, ax = plt.subplots(figsize=(8, 5))
            parties = list(result["party_probabilities"].keys())
            probs = [result["party_probabilities"][p] for p in parties]
            
            colors = ["#ff6b6b", "#339af0", "#20c997"]
            ax.bar(parties, probs, color=colors[:len(parties)])
            ax.set_ylabel("Probability (%)")
            ax.set_title(f"{example_model} - Party Probability Distribution")
            
            for i, v in enumerate(probs):
                ax.text(i, v + 1, f"{v:.1f}%", ha='center')
            
            st.pyplot(fig)
            
            # Add interactive word cloud
            st.markdown("### 🔤 Key Terms Analysis")
            
            # Generate word frequencies based on the party
            if result['political_party'] == 'republican':
                words = {
                    "economy": 0.8, "taxes": 0.7, "jobs": 0.6, "security": 0.5, "freedom": 0.7,
                    "military": 0.6, "business": 0.5, "government": 0.4, "regulation": 0.5,
                    "borders": 0.6, "immigration": 0.5, "tradition": 0.4, "family": 0.5
                }
            elif result['political_party'] == 'democrat':
                words = {
                    "healthcare": 0.8, "education": 0.7, "climate": 0.7, "equality": 0.6, 
                    "rights": 0.7, "diversity": 0.6, "community": 0.5, "progress": 0.6,
                    "justice": 0.5, "reform": 0.4, "environment": 0.6, "workers": 0.5
                }
            else:
                words = {
                    "balance": 0.7, "compromise": 0.6, "moderate": 0.7, "pragmatic": 0.6,
                    "independent": 0.8, "centrist": 0.7, "bipartisan": 0.6, "practical": 0.5,
                    "reform": 0.5, "common-sense": 0.6, "middle-ground": 0.5
                }
            
            # Create a simple visual representation of word frequencies
            for word, freq in sorted(words.items(), key=lambda x: x[1], reverse=True):
                st.markdown(
                    f'<span style="font-size: {int(freq * 24)}px; margin-right: 10px; color: {colors[parties.index(result["political_party"])]};">{word}</span>',
                    unsafe_allow_html=True
                )

# Tab 3: Model Comparison
with tab3:
    st.markdown('<p class="sub-header">🤖 Compare Model Performance</p>', unsafe_allow_html=True)
    
    st.markdown("""
    This section allows you to compare how different LLMs perform on the same text inputs.
    Enter multiple statements to see how each model classifies them.
    """)
    
    # Input multiple statements
    statements = st.text_area(
        "Enter multiple statements (one per line):",
        height=150,
        placeholder="The economy is doing great under this administration!\nWe need to secure our borders and protect American jobs.\nHealthcare should be a right for all citizens."
    )
    
    compare_button = st.button("🔄 Compare Models", use_container_width=True)
    
    if statements and compare_button:
        # Split statements into list
        statement_list = [s.strip() for s in statements.split('\n') if s.strip()]
        
        if statement_list:
            # Show progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i in range(100):
                # Update progress bar
                progress_bar.progress(i + 1)
                
                # Update status text based on progress
                if i < 20:
                    status_text.text("🔍 Analyzing statements...")
                elif i < 40:
                    status_text.text("🧠 Processing with multiple models...")
                elif i < 70:
                    status_text.text("📊 Comparing results...")
                elif i < 90:
                    status_text.text("📈 Generating visualizations...")
                else:
                    status_text.text("🎁 Finalizing comparison...")
                
                time.sleep(0.02)  # Faster for demo purposes
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            # Create comparison table
            comparison_data = []
            
            for statement in statement_list:
                row = {"Statement": statement}
                
                for model in ["GPT-4o-mini", "Mistral", "Copilot", "Llama 3.2 70B"]:
                    result = analyze_text(statement, model)
                    row[f"{model} Prediction"] = result["political_party"].capitalize()
                    row[f"{model} Confidence"] = f"{result['confidence']:.1f}%"
                
                comparison_data.append(row)
            
            comparison_df = pd.DataFrame(comparison_data)
            
            # Display comparison table
            st.markdown("### 📊 Model Comparison Results")
            st.dataframe(comparison_df)
            
            # Calculate agreement statistics
            st.markdown("### 🔍 Model Agreement Analysis")
            
            agreement_data = []
            
            for statement in statement_list:
                predictions = {}
                
                for model in ["GPT-4o-mini", "Mistral", "Copilot", "Llama 3.2 70B"]:
                    result = analyze_text(statement, model)
                    predictions[model] = result["political_party"]
                
                # Count unique predictions
                unique_predictions = set(predictions.values())
                
                # Calculate agreement percentage
                agreement_pct = (1 - (len(unique_predictions) - 1) / 3) * 100
                
                agreement_data.append({
                    "Statement": statement,
                    "Unique Predictions": len(unique_predictions),
                    "Agreement": f"{agreement_pct:.1f}%",
                    "Predictions": ", ".join([f"{m}: {p.capitalize()}" for m, p in predictions.items()])
                })
            
            agreement_df = pd.DataFrame(agreement_data)
            
            # Display agreement statistics
            st.dataframe(agreement_df)
            
            # Visualize agreement
            st.markdown("### 📈 Model Agreement Visualization")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            agreement_values = [float(a.replace("%", "")) for a in agreement_df["Agreement"]]
            statements_short = [s[:30] + "..." if len(s) > 30 else s for s in agreement_df["Statement"]]
            
            colors = []
            for val in agreement_values:
                if val >= 80:
                    colors.append("#20c997")  # High agreement
                elif val >= 50:
                    colors.append("#ffd43b")  # Medium agreement
                else:
                    colors.append("#ff6b6b")  # Low agreement
            
            ax.barh(statements_short, agreement_values, color=colors)
            ax.set_xlabel("Agreement (%)")
            ax.set_title("Model Agreement by Statement")
            
            for i, v in enumerate(agreement_values):
                ax.text(v + 1, i, f"{v:.1f}%", va='center')
            
            st.pyplot(fig)
            
            # Add interactive model comparison
            st.markdown("### 🔄 Interactive Model Comparison")
            
            selected_statement = st.selectbox(
                "Select a statement to analyze in detail:",
                options=statement_list
            )
            
            if selected_statement:
                # Analyze with all models
                detailed_results = {}
                for model in ["GPT-4o-mini", "Mistral", "Copilot", "Llama 3.2 70B"]:
                    detailed_results[model] = analyze_text(selected_statement, model)
                
                # Create comparison visualization
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Set up bar positions
                models = list(detailed_results.keys())
                parties = ["republican", "democrat", "independent"]
                x = np.arange(len(models))
                width = 0.25
                
                # Plot bars for each party
                for i, party in enumerate(parties):
                    values = [detailed_results[model]["party_probabilities"].get(party, 0) for model in models]
                    ax.bar(x + (i-1)*width, values, width, label=party.capitalize(), 
                          color=["#ff6b6b", "#339af0", "#20c997"][i])
                
                # Set chart properties
                ax.set_ylabel('Probability (%)')
                ax.set_title(f'Detailed Model Comparison for: "{selected_statement}"')
                ax.set_xticks(x)
                ax.set_xticklabels(models)
                ax.legend()
                ax.grid(axis='y', linestyle='--', alpha=0.7)
                
                st.pyplot(fig)
                
                # Show detailed explanations
                st.markdown("### 🧠 Model Explanations")
                
                for model, result in detailed_results.items():
                    party_class = result["political_party"]
                    
                    st.markdown(f"""
                    <div class="model-card {party_class}">
                        <h4>{model}</h4>
                        <p><strong>Political Party:</strong> {result["political_party"].capitalize()}</p>
                        <p><strong>Confidence:</strong> {result["confidence"]:.1f}%</p>
                        <p><strong>Explanation:</strong> {result["explanation"]}</p>
                    </div>
                    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #888;">
        Made with ❤️ by Sai Rupa Jhade | Data Science Portfolio
    </div>
    """, 
    unsafe_allow_html=True
)
