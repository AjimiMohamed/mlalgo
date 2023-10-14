import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Theming - Light Mode
st.set_page_config(
    page_title="Chess Endgame Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="♟️",  # Chess pawn as a favicon
)

st.title("🔍 Chess Endgame Analysis")
st.write("Analyze the King-Rook vs. King endgame dataset from the UCI repository. Explore the data, compare algorithms, and predict outcomes.")


# Load the dataset
data_path = "krkopt.data"
data = pd.read_csv(data_path, names=["White King File", "White King Rank", "White Rook File", "White Rook Rank", "Black King File", "Black King Rank", "Outcome"])

# Display the first few rows of the dataset
st.subheader("Dataset Preview")
st.write(data.head())

# Preprocessing
file_mapping = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8}
data['White King File'] = data['White King File'].map(file_mapping)
data['White Rook File'] = data['White Rook File'].map(file_mapping)
data['Black King File'] = data['Black King File'].map(file_mapping)
X = data.drop(columns="Outcome")
y = data["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Tabs for different sections
tabs = ["User Input & Predictions", "Exploratory Data Analysis", "Model Training & Evaluation"]
selected_tab = st.selectbox("Choose a tab:", tabs)

if selected_tab == "User Input & Predictions":
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        white_king_file = st.selectbox("Select White King File", [""] + list(file_mapping.keys()))
    with col2:
        white_king_rank = st.number_input("Enter White King Rank", 1, 8, 1, format="%i")
    with col3:
        white_rook_file = st.selectbox("Select White Rook File", [""] + list(file_mapping.keys()))
    with col4:
        white_rook_rank = st.number_input("Enter White Rook Rank", 1, 8, 1, format="%i")
    with col5:
        black_king_file = st.selectbox("Select Black King File", [""] + list(file_mapping.keys()))
    with col6:
        black_king_rank = st.number_input("Enter Black King Rank", 1, 8, 1, format="%i")
    if white_king_file and white_rook_file and black_king_file:
        white_king_file_numeric = file_mapping[white_king_file]
        white_rook_file_numeric = file_mapping[white_rook_file]
        black_king_file_numeric = file_mapping[black_king_file]
        user_input_data = [[white_king_file_numeric, white_king_rank, white_rook_file_numeric, white_rook_rank, black_king_file_numeric, black_king_rank]]
        dt_model = DecisionTreeClassifier()
        dt_model.fit(X_train, y_train)
        if st.button("Predict Outcome"):
            prediction = dt_model.predict(user_input_data)
            st.markdown(f"### Predicted Outcome: **{prediction[0]}**")

elif selected_tab == "Exploratory Data Analysis":
   # EDA: Distribution of Outcomes
    st.subheader("Distribution of Outcomes")
    fig, ax = plt.subplots(figsize=(4, 2))
    data["Outcome"].value_counts().plot(kind='bar', ax=ax)
    st.pyplot(fig)

    # EDA: Distribution of White King File Positions
    st.subheader("Distribution of White King File Positions")
    fig, ax = plt.subplots(figsize=(4, 2))
    data["White King File"].value_counts().sort_index().plot(kind='bar', ax=ax)
    st.pyplot(fig)

    # EDA: Distribution of White Rook File Positions
    st.subheader("Distribution of White Rook File Positions")
    fig, ax = plt.subplots(figsize=(4, 2))
    data["White Rook File"].value_counts().sort_index().plot(kind='bar', ax=ax)
    st.pyplot(fig)

    # EDA: Distribution of Black King File Positions
    st.subheader("Distribution of Black King File Positions")
    fig, ax = plt.subplots(figsize=(4, 2))
    data["Black King File"].value_counts().sort_index().plot(kind='bar', ax=ax)
    st.pyplot(fig)

elif selected_tab == "Model Training & Evaluation":
    st.subheader("Train and Evaluate ML Models")
    algorithm = st.selectbox("Choose an algorithm", ["Decision Tree", "SVM", "Gradient Boosting"])
    if algorithm == "Decision Tree":
        max_depth = st.slider("Max Depth", 1, 18, 10)
        model = DecisionTreeClassifier(max_depth=max_depth)
    elif algorithm == "SVM":
        C = st.slider("C (Regularization)", 0.01, 10.0)
        model = SVC(C=C)
    elif algorithm == "Gradient Boosting":
        n_estimators = st.slider("Number of Estimators", 10, 100)
        model = GradientBoostingClassifier(n_estimators=n_estimators)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions) * 100
    st.markdown(f"### Accuracy: **{accuracy:.2f}%**")
