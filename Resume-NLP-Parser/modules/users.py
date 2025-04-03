import streamlit as st
import sqlite3
import pandas as pd
from resume_parser import (
    display_job_listings, extract_resume_info_from_pdf, extract_contact_number_from_resume, 
    extract_education_from_resume, extract_experience, fetch_jobs_from_csv, generate_report, score_resume_for_job, suggest_skills_for_job, show_colored_skills, 
    calculate_resume_score, extract_resume_info, improvement_suggestions
)

def create_table():
    conn = sqlite3.connect('data/user_pdfs.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_uploaded_pdfs (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            data BLOB NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def insert_pdf(name, data):
    conn = sqlite3.connect('data/user_pdfs.db')
    cursor = conn.cursor()
    cursor.execute('INSERT INTO user_uploaded_pdfs (name, data) VALUES (?, ?)', (name, data))
    conn.commit()
    conn.close()

def generate_report(resume_info, improvements, skills, resume_score, contact_number, education_info, experience_info):
    report_content = f"""
    Resume Report

    Personal Information:
    First Name: {resume_info['first_name']}
    Last Name: {resume_info['last_name']}
    Email: {resume_info['email']}
    Phone Number: {contact_number}
    Degree/Major: {resume_info['degree_major']}

    Education:
    {', '.join(education_info) if education_info else "No education information found"}

    Skills:
    {', '.join(skills)}

    Experience:
    Level of Experience: {experience_info['level_of_experience']}
    Suggested Position: {experience_info['suggested_position']}

   

    Resume Score: {resume_score}/100

    Improvements:
    {improvements}
    """
    return report_content


def process_user_mode():
    create_table() 

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

    st.markdown('<div class="main">', unsafe_allow_html=True)
    st.write("Upload your resume in PDF format below. The system will analyze and provide feedback on your resume.")
    uploaded_file = st.file_uploader("Upload a PDF resume", type="pdf")
    

    if uploaded_file:
        st.write("File uploaded successfully!")

        pdf_name = uploaded_file.name
        pdf_data = uploaded_file.getvalue()

        insert_pdf(pdf_name, pdf_data)

        pdf_text = extract_resume_info_from_pdf(uploaded_file)
        resume_info = extract_resume_info(pdf_text)

        st.markdown('<div class="section-title">Extracted Information:</div>', unsafe_allow_html=True)
        st.write(f"**First Name:** {resume_info['first_name']}")
        st.write(f"**Last Name:** {resume_info['last_name']}")
        st.write(f"**Email:** {resume_info['email']}")

        contact_number = extract_contact_number_from_resume(pdf_text)
        st.write(f"**Phone Number:** {contact_number if contact_number else 'Not found'}")
        
        st.write(f"**Degree/Major:** {resume_info['degree_major']}")

        st.markdown('<div class="section-title">Education:</div>', unsafe_allow_html=True)
        education_info = extract_education_from_resume(pdf_text)
        st.write(', '.join(education_info) if education_info else "No education information found")

        st.markdown('<div class="section-title">Skills:</div>', unsafe_allow_html=True)
        skills = resume_info['skills']
        show_colored_skills(skills)

        st.markdown('<div class="section-title">Experience:</div>', unsafe_allow_html=True)
        experience_info = extract_experience(pdf_text)
        st.write(f"**Level of Experience:** {experience_info['level_of_experience']}")
        st.write(f"**Suggested Position:** {experience_info['suggested_position']}")

        st.markdown('<div class="section-title">Resume Score:</div>', unsafe_allow_html=True)
        resume_score = calculate_resume_score(resume_info)
        st.write(f"**Resume Score:** {resume_score}/100")

        percentage = resume_score
        bar = (
            f'<div class="progress-bar" style="width: {percentage}%;">'
            f'<div>{percentage}%</div>'
            '</div>'
        )
        st.markdown(bar, unsafe_allow_html=True)

        st.markdown('<div class="section-title">Improvements:</div>', unsafe_allow_html=True)
        improvements = improvement_suggestions(resume_score)
        st.write(improvements)

        st.markdown('<div class="section-title">Matching Job Listings:</div>', unsafe_allow_html=True)

  
        file_path = 'data/jobs.csv'  
        jobs = fetch_jobs_from_csv(file_path)

        if not jobs.empty:
        
            jobs['description'] = jobs['description'].fillna('')  

            
            if skills:
                jobs['match_score'] = jobs['description'].apply(
                    lambda desc: sum(skill.lower() in desc.lower() for skill in skills)
                )
                matching_jobs = jobs.sort_values(by='match_score', ascending=False)

                sample_size = min(5, len(matching_jobs)) 
                limited_jobs = matching_jobs.sample(n=sample_size, random_state=1)  
                display_job_listings(limited_jobs)
            else:
                st.write("No skills found in the resume to match against job listings.")
        else:
            st.write("No job listings found.")

        if st.button("Generate and Download Report"):
            report = generate_report(resume_info, improvements, skills, resume_score, contact_number, education_info, experience_info)
            st.download_button(
                label="Download Report",
                data=report,
                file_name="resume_report.txt",
                mime="text/plain",)

        st.markdown('<div class="section-title">Suggested Skills for the Desired Job:</div>', unsafe_allow_html=True)
        desired_job = st.text_input("Enter the job you are looking for:")
     
            
        suggested_skills = suggest_skills_for_job(desired_job)
        st.write(suggested_skills)

        st.markdown('<div class="section-title">Job Description Matching:</div>', unsafe_allow_html=True)
        job_description = st.text_area("Enter the job description:", height=200)

        if job_description:
            job_match_result = score_resume_for_job(resume_info, job_description)

            st.write(f"**Match Score:** {job_match_result['overall_match']:.2f}%")
            st.write(f"**Matching Skills:** {', '.join(job_match_result['matching_skills'])}")
            st.write(f"**Missing Skills:** {', '.join(job_match_result['missing_skills'])}")
            
        else:
            st.write("Please enter a job description to calculate the match score.")


    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == '__main__':
    process_user_mode()
    
   
