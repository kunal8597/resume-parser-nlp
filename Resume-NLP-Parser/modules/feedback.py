import streamlit as st
from datetime import datetime

def process_feedback_mode():
    st.markdown(
        """
        <div style="
            background: linear-gradient(135deg, #EA4F27, #FD714F);
            padding: 2rem;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            text-align: center;
        ">
            <h1 style="color: white; font-family: Arial, sans-serif; font-weight: bold; margin: 0;">
                Resume Parser using NLP
            </h1>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <style>
        .main {
            background: linear-gradient(135deg, #03045E, #023E8A);
            padding: 2rem;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        .header {
            text-align: center;
            color: #333;
        }
        .section-title {
            color: #48CAE4;
            font-weight: bold;
            margin-top: 2rem;
        }
        .section-content {
            margin-bottom: 2rem;
        }
        .progress-bar {
            background: linear-gradient(90deg, #f63366 0%, #d6d6d6 100%);
            height: 30px;
            border-radius: 5px;
            display: flex;
            align-items: center;
            position: relative;
        }
        .progress-bar div {
            color: white;
            text-align: center;
            width: 100%;
            position: absolute;
            top: 0;
            left: 0;
            line-height: 30px;
        }
        </style>
        """, unsafe_allow_html=True
    )

    st.title("Feedback Section")
    st.subheader("Provide Feedback")

    # Feedback Form
    user_name = st.text_input("Your Name:")
    feedback = st.text_area("Provide feedback on the resume parser:", height=100)
    if st.button("Submit Feedback"):
        add_feedback(user_name, feedback)
        st.success("Feedback submitted successfully!")

def add_feedback(user_name, feedback):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open('data/feedback_data.csv', 'a') as file:
        file.write(f"User Name: {user_name}\n")
        file.write(f"Feedback: {feedback}\n")
        file.write(f"Timestamp: {timestamp}\n")
        file.write("-" * 50 + "\n")
