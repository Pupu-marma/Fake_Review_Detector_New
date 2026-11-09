import re
import html
import math
import joblib
import streamlit as st
import pandas as pd
import plotly.express as px

from pathlib import Path


# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="ReviewDetector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =====================================================
# 2. BEST MODEL LOADING - LINEAR SVM
# =====================================================

BASE_DIR = Path(__file__).parent

SVM_MODEL_PATH = BASE_DIR / "models" / "linear_svm_5000.pkl"


@st.cache_resource
def load_model(model_path):

    if model_path.exists():
        return joblib.load(model_path)

    return None


best_model = load_model(SVM_MODEL_PATH)

# =====================================================
# 3. SESSION STATE
# =====================================================

if "page" not in st.session_state:
    st.session_state.page = "Home"

if "history" not in st.session_state:
    st.session_state.history = []

if "total_analyzed" not in st.session_state:
    st.session_state.total_analyzed = 0

if "bulk_result" not in st.session_state:
    st.session_state.bulk_result = None


# =====================================================
# 4. TEXT PREPROCESSING
# =====================================================

def clean_text(text):

    text = str(text)
    text = html.unescape(text)
    text = re.sub(r"<.*?>", " ", text)
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


# =====================================================
# 5. BEST MODEL PREDICTION
# =====================================================

def get_label(prediction):

    prediction = str(prediction).upper()

    if prediction == "CG":
        return "Fake Review", "#EF4444"

    elif prediction == "OR":
        return "Original Review", "#22C55E"

    return str(prediction), "#3B82F6"


def get_model_prediction(selected_model, review):

    cleaned = clean_text(review)

    prediction = selected_model.predict([cleaned])[0]

    label, color = get_label(prediction)

    return label, color


# =====================================================
# 6. CUSTOM CSS
# =====================================================

st.markdown("""
<style>

.stApp {
    background-color: #0B1220;
    color: #F8FAFC;
}

header[data-testid="stHeader"] {
    background-color: #0B1220;
}

html, body, [class*="css"] {
    font-size: 14px !important;
}

p, li, label {
    font-size: 14px !important;
    line-height: 1.6 !important;
    color: #E2E8F0;
}

h1 {
    font-size: 27px !important;
}

h2 {
    font-size: 21px !important;
}

h3 {
    font-size: 17px !important;
}

/* SIDEBAR */

section[data-testid="stSidebar"] {
    background-color: #111D32;
    border-right: 1px solid #263B59;
}

section[data-testid="stSidebar"] h2 {
    color: #60A5FA !important;
    font-size: 21px !important;
}

section[data-testid="stSidebar"] h3 {
    color: #93C5FD !important;
    font-size: 15px !important;
}

section[data-testid="stSidebar"] p {
    color: #CBD5E1 !important;
    font-size: 12px !important;
}

/* SIDEBAR BUTTONS */

section[data-testid="stSidebar"] .stButton button {
    background-color: #172840;
    color: #F8FAFC !important;
    border: 1px solid #334D70;
    border-radius: 8px;
    font-size: 13px !important;
    min-height: 38px;
    text-align: left;
}

section[data-testid="stSidebar"] .stButton button:hover {
    background-color: #2563EB;
    border-color: #60A5FA;
}

/* MAIN TITLE */

.main-title {
    text-align: center;
    color: #60A5FA;
    font-size: 32px;
    font-weight: 800;
    padding: 10px 0 18px 0;
}

/* WELCOME BANNER */

.welcome-banner {
    background: linear-gradient(135deg, #172840, #1E40AF);
    padding: 27px 18px;
    border-radius: 15px;
    text-align: center;
    margin-bottom: 22px;
    border: 1px solid #45617F;
}

.welcome-banner h2 {
    color: #FFFFFF !important;
    font-size: 25px !important;
    margin-bottom: 10px;
}

.welcome-banner p {
    color: #DBEAFE !important;
    font-size: 14px !important;
}

/* CARDS */

.custom-card {
    background: #172840;
    border: 1px solid #334D70;
    border-radius: 13px;
    padding: 19px;
    min-height: 125px;
    margin-bottom: 14px;
}

.custom-card h3 {
    color: #93C5FD !important;
    font-size: 16px !important;
}

.custom-card p {
    color: #F8FAFC !important;
    font-size: 13px !important;
    line-height: 1.8 !important;
}

/* METRIC CARDS */

.metric-card {
    background: #172840;
    border: 1px solid #334D70;
    border-radius: 12px;
    padding: 17px;
    text-align: center;
}

.metric-number {
    color: #60A5FA;
    font-size: 27px;
    font-weight: bold;
}

.metric-label {
    color: #CBD5E1;
    font-size: 12px;
}

/* RESULT CARDS */

.result-card {
    background: #172840;
    border-radius: 14px;
    padding: 22px;
    border: 1px solid #334D70;
    text-align: center;
    min-height: 145px;
}

.result-card h2 {
    font-size: 22px !important;
}

/* INPUTS */

.stTextArea textarea {
    background-color: #172840 !important;
    color: white !important;
    border: 1px solid #45617F !important;
    font-size: 14px !important;
}

/* BUTTONS */

.stButton button {
    background: #2563EB;
    color: white !important;
    border: none;
    border-radius: 8px;
    font-size: 13px !important;
    font-weight: 600;
    min-height: 39px;
}

.stButton button:hover {
    background: #1D4ED8;
}

/* FILE UPLOADER */

div[data-testid="stFileUploader"] {
    background: #172840;
    border: 1px dashed #45617F;
    border-radius: 12px;
    padding: 13px;
}

hr {
    border-color: #334D70;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# 7. REUSABLE CARD
# =====================================================

def show_card(title, content):

    st.markdown(f"""
    <div class="custom-card">
        <h3>{title}</h3>
        <p>{content}</p>
    </div>
    """, unsafe_allow_html=True)
    
# =====================================================
# 8. RESULT DISPLAY FUNCTION
# =====================================================

def show_prediction_card(prediction, color):

    with st.container(border=True):

        if prediction == "Fake Review":

            st.markdown("## :red[Fake Review]")

        elif prediction == "Original Review":

            st.markdown("## :green[Original Review]")

        else:

            st.subheader(prediction)

        st.caption("Review Classification Result")
                


# =====================================================
# 9. SIDEBAR NAVIGATION
# =====================================================

with st.sidebar:

    st.markdown("""
    <h2>🔍 ReviewDetector</h2>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### MAIN MENU")

    if st.button("🏠 Home", key="nav_home",
                 use_container_width=True):
        st.session_state.page = "Home"

    if st.button("🔍 Single Review", key="nav_single",
                 use_container_width=True):
        st.session_state.page = "Single Review"

    if st.button("📊 Bulk Analysis", key="nav_bulk",
                 use_container_width=True):
        st.session_state.page = "Bulk Analysis"

    st.markdown("---")

    st.markdown("### INFORMATION")

    if st.button("⚙️ How It Works", key="nav_how",
                 use_container_width=True):
        st.session_state.page = "How It Works"

    if st.button("ℹ️ About Project", key="nav_about",
                 use_container_width=True):
        st.session_state.page = "About Project"

    st.markdown("---")

    st.markdown("""
    <div style="text-align:center;">
        <p>AI-Powered Review Detection</p>
        <p>Machine Learning Project</p>
    </div>
    """, unsafe_allow_html=True)


# =====================================================
# 10. MAIN HEADER
# =====================================================

st.markdown("""
<div class="main-title">
    🔍 ReviewDetector
</div>
""", unsafe_allow_html=True)


# =====================================================
# 11. HOME PAGE
# =====================================================

if st.session_state.page == "Home":

    st.markdown("""
    <div class="welcome-banner">
        <h2>Welcome to ReviewDetector</h2>
        <p>AI-Powered Fake Review Detection System</p>
        <p>Detect Fake and Original Reviews Using Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)

    fake_count = sum(
        1 for item in st.session_state.history
        if item["Prediction"] == "Fake Review"
    )

    original_count = sum(
        1 for item in st.session_state.history
        if item["Prediction"] == "Original Review"
    )

    st.markdown("## Dashboard Overview")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-number">
                {st.session_state.total_analyzed}
            </div>
            <div class="metric-label">
                Total Reviews Analyzed
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-number">{fake_count}</div>
            <div class="metric-label">Fake Reviews</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-number">{original_count}</div>
            <div class="metric-label">Original Reviews</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("## Platform Overview")

    st.write(
        "ReviewDetector is a Machine Learning based application "
        "designed to identify fake and original customer reviews. "
        "It provides an easy way to analyze customer feedback."
    )

    col1, col2 = st.columns(2)

    with col1:
        show_card(
            "🔍 Single Review Detection",
            "Analyze one customer review and identify whether "
            "it is Fake or Original using two Machine Learning models."
        )

    with col2:
        show_card(
            "📊 Bulk Review Analysis",
            "Upload a CSV file and analyze multiple customer "
            "reviews simultaneously."
        )

    st.markdown("## Recent Analysis")

    if st.session_state.history:

        history_df = pd.DataFrame(
            st.session_state.history[-10:][::-1]
        )

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

    else:
        st.info(
            "No reviews analyzed yet. Start with Single Review Detection."
        )


# =====================================================
# 12. SINGLE REVIEW
# =====================================================

elif st.session_state.page == "Single Review":

    st.markdown("## Single Review Detection")

    st.write(
        "Enter a customer review to identify "
        "whether it is Fake or Original."
    )

    review = st.text_area(
        "Enter Customer Review",
        height=160,
        placeholder="Type or paste your customer review here..."
    )

    if st.button(
        "🔍 Analyze Review",
        use_container_width=True
    ):

        if best_model is None:

            st.error(
                f"Best model not found: {SVM_MODEL_PATH}"
            )

        elif not review.strip():

            st.warning("Please enter a review first.")

        else:

            try:

                with st.spinner("Analyzing review..."):

                    prediction, color = get_model_prediction(
                        best_model,
                        review
                    )

                st.session_state.total_analyzed += 1

                st.session_state.history.append({
                    "Review": review[:100],
                    "Prediction": prediction
                })

                st.markdown("---")

                st.markdown("### Analysis Result")

                show_prediction_card(
                    prediction,
                    color
                )

                if prediction == "Fake Review":

                    st.error("This review is classified as Fake.")

                elif prediction == "Original Review":

                    st.success(
                        "This review is classified as Original."
                    )

            except Exception as error:

                st.error(f"Prediction Error: {error}")


# =====================================================
# 13. BULK ANALYSIS
# =====================================================

elif st.session_state.page == "Bulk Analysis":

    st.markdown("## Bulk Review Analysis")

    st.write(
        "Upload a CSV file containing customer reviews."
    )

    uploaded_file = st.file_uploader(
        "Upload CSV File",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            try:
                df = pd.read_csv(uploaded_file)

            except UnicodeDecodeError:

                uploaded_file.seek(0)

                df = pd.read_csv(
                    uploaded_file,
                    encoding="latin-1"
                )

            st.success("CSV uploaded successfully!")

            st.markdown("### Dataset Preview")

            st.dataframe(
                df.head(),
                use_container_width=True,
                hide_index=True
            )

            if len(df.columns) == 0:

                st.warning("CSV file has no columns.")

            else:

                possible_columns = [
                    "review",
                    "reviews",
                    "text",
                    "review_text",
                    "content"
                ]

                matching = [
                    col for col in df.columns
                    if str(col).lower() in possible_columns
                ]

                default_column = (
                    matching[0] if matching else df.columns[0]
                )

                review_column = st.selectbox(
                    "Select Review Column",
                    df.columns,
                    index=list(df.columns).index(default_column)
                )

                if st.button(
                    "📊 Analyze All Reviews",
                    use_container_width=True
                ):

                    if best_model is None:

                        st.error(
                            f"Model file not found: {MODEL_PATH}"
                        )

                    else:

                        reviews = (
                            df[review_column]
                            .fillna("")
                            .astype(str)
                        )

                        valid_mask = reviews.str.strip() != ""

                        results = df.loc[valid_mask].copy()

                        if results.empty:

                            st.warning("No valid reviews found.")

                        else:

                            with st.spinner("Analyzing reviews..."):

                                cleaned = reviews[
                                    valid_mask
                                ].apply(clean_text)

                                predictions = best_model.predict(
                                      cleaned.tolist()
                                      )

                                results["Prediction"] = [
                                    get_label(p)[0]
                                    for p in predictions
                                ]

                            st.session_state.bulk_result = results

                            st.session_state.total_analyzed += len(
                                results
                            )

                if st.session_state.bulk_result is not None:

                    result_df = st.session_state.bulk_result

                    fake_total = (
                        result_df["Prediction"] == "Fake Review"
                    ).sum()

                    original_total = (
                        result_df["Prediction"] == "Original Review"
                    ).sum()

                    st.markdown("---")
                    st.markdown("## Analysis Results")

                    m1, m2, m3 = st.columns(3)

                    m1.metric("Total Reviews", len(result_df))
                    m2.metric("Fake Reviews", int(fake_total))
                    m3.metric(
                        "Original Reviews",
                        int(original_total)
                    )

                    chart_data = pd.DataFrame({
                        "Review Type": [
                            "Fake Review",
                            "Original Review"
                        ],
                        "Count": [
                            int(fake_total),
                            int(original_total)
                        ]
                    })

                    chart1, chart2 = st.columns(2)

                    with chart1:

                        st.markdown("### Review Distribution")

                        fig = px.bar(
                            chart_data,
                            x="Review Type",
                            y="Count",
                            color="Review Type",
                            text="Count",
                            color_discrete_map={
                                "Fake Review": "#EF4444",
                                "Original Review": "#22C55E"
                            }
                        )

                        fig.update_layout(
                            template="plotly_dark",
                            paper_bgcolor="#0B1220",
                            plot_bgcolor="#0B1220",
                            showlegend=False
                        )

                        st.plotly_chart(
                            fig,
                            use_container_width=True
                        )

                    with chart2:

                        st.markdown(
                            "### Percentage Distribution"
                        )

                        pie = px.pie(
                            chart_data,
                            names="Review Type",
                            values="Count",
                            color="Review Type",
                            color_discrete_map={
                                "Fake Review": "#EF4444",
                                "Original Review": "#22C55E"
                            },
                            hole=0.45
                        )

                        pie.update_layout(
                            template="plotly_dark",
                            paper_bgcolor="#0B1220"
                        )

                        st.plotly_chart(
                            pie,
                            use_container_width=True
                        )

                    st.markdown("### Detailed Results")

                    st.dataframe(
                        result_df,
                        use_container_width=True,
                        hide_index=True
                    )

                    csv_data = result_df.to_csv(
                        index=False
                    ).encode("utf-8-sig")

                    st.download_button(
                        "⬇️ Download Analysis Results",
                        data=csv_data,
                        file_name="review_analysis_results.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

        except Exception as error:

            st.error(f"CSV Error: {error}")


# =====================================================
# 14. HOW IT WORKS
# =====================================================

elif st.session_state.page == "How It Works":

    st.markdown("## How ReviewDetector Works")

    st.write(
        "The system uses Natural Language Processing "
        "and Machine Learning to classify customer reviews."
    )

    steps = [
        (
            "01",
            "📝 Review Input",
            "The user enters a customer review into the application."
        ),
        (
            "02",
            "🧹 Text Preprocessing",
            "The review text is cleaned and prepared for machine learning."
        ),
        (
            "03",
            "🔢 Feature Extraction",
            "TF-IDF converts text into numerical features."
        ),
        (
            "04",
            "🧠 Machine Learning",
            "The trained classification models analyze extracted features."
        ),
        (
            "05",
            "🔍 Classification",
            "The models predict Fake (CG) or Original (OR)."
        ),
        (
            "06",
            "📊 Result Visualization",
            "The application displays prediction and analysis results."
        )
    ]

    for number, title, description in steps:

        st.markdown(f"""
        <div class="custom-card">
            <h3>{number} &nbsp; {title}</h3>
            <p>{description}</p>
        </div>
        """, unsafe_allow_html=True)


# =====================================================
# 15. ABOUT PROJECT
# =====================================================

elif st.session_state.page == "About Project":

    st.markdown("## About ReviewDetector")

    st.write(
        "ReviewDetector is an academic Computer Science "
        "and Engineering project designed to identify fake "
        "and original customer reviews using Machine Learning."
    )

    st.markdown("### Project Objectives")

    show_card(
        "Objective 01",
        "Detect misleading and fake customer reviews."
    )

    show_card(
        "Objective 02",
        "Classify customer reviews using Machine Learning."
    )

    show_card(
        "Objective 03",
        "Provide an easy-to-use interface for review analysis."
    )

    show_card(
        "Objective 04",
        "Support analysis of individual and multiple reviews."
    )

    st.markdown("## Technologies Used")

    col1, col2 = st.columns(2)

    with col1:

        show_card(
            "Programming & Tools",
            "🐍 Python<br>"
            "🌐 Streamlit<br>"
            "📊 Plotly<br>"
            "📁 Pandas"
        )

    with col2:

        show_card(
            "Machine Learning",
            "⚙️ Scikit-learn<br>"
            "📝 Natural Language Processing<br>"
            "🔢 TF-IDF Feature Extraction<br>"
            "🧠 Text Classification"
        )

    st.markdown("## Project Information")

    show_card(
        "Project Details",
        "<b>Project Name:</b> ReviewDetector<br>"
        "<b>Project Type:</b> Academic CSE Project<br>"
        "<b>Domain:</b> Machine Learning & NLP<br>"
        "<b>Application:</b> Fake Review Detection"
    )



# =====================================
# 16. FOOTER
# =====================================

st.divider()

st.markdown(
    "<div style='text-align:center;'>"
    "<span style='color:#94A3B8; font-size:12px;'>"
    "🔍 ReviewDetector | Fake Review Detection System"
    "</span><br>"
    "<span style='color:#64748B; font-size:11px;'>"
    "Machine Learning & Natural Language Processing"
    "</span><br>"
    "<span style='color:#64748B; font-size:10px;'>"
    "Developed as an Academic CSE Project"
    "</span>"
    "</div>",
    unsafe_allow_html=True
)
