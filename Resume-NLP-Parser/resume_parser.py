from http import client
import random
import re
from bs4 import BeautifulSoup
import fitz
import base64
import openai
import pandas as pd
import streamlit as st
import spacy
import csv
import nltk
import time
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import KFold
import openai
import requests


from jobspy import scrape_jobs


nltk.download('punkt')

nlp = spacy.load('en_core_web_sm')

def load_keywords(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        return set(row[0] for row in reader)

# ----------------------------------Extract Name----------------------------------
def extract_name(doc):
    for ent in doc.ents:
        if ent.label_ == 'PERSON':
            # Split the entity into words and ensure they start with uppercase letters
            names = ent.text.split()
            
            # Ensure that each part of the name has a capital letter and avoid titles
            if len(names) >= 2 and all(name.istitle() for name in names):
                # Check if the names look like a valid first and last name
                first_name = names[0]
                last_name = ' '.join(names[1:])
                
                # Further validation can be added here, such as checking for known titles
                return first_name, last_name
    return None, None

# ----------------------------------Extract Email---------------------------------
def extract_email(doc):
    matcher = spacy.matcher.Matcher(nlp.vocab)
    email_pattern = [{'LIKE_EMAIL': True}]
    matcher.add('EMAIL', [email_pattern])

    matches = matcher(doc)
    for match_id, start, end in matches:
        if match_id == nlp.vocab.strings['EMAIL']:
            return doc[start:end].text
    return ""
# --------------------------------------------------------------------------------

# ----------------------------------Extract Ph No---------------------------------
def extract_contact_number_from_resume(doc):
    contact_number = None
    text = doc.text  
    pattern = r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
    match = re.search(pattern, text)
    if match:
        contact_number = match.group()
    return contact_number
# --------------------------------------------------------------------------------

# --------------------------------Extract Education-------------------------------
def extract_education_from_resume(doc):
    universities = set()  

    
    doc = nlp(doc)

    
    keywords = ["university", "institute", "college", "academy", "school", "faculty", "of"]

  
    for entity in doc.ents:
        if entity.label_ == "ORG" and any(keyword in entity.text.lower() for keyword in keywords):
            university_name = entity.text.strip()
            universities.add(university_name)

   
    if not universities:
        for keyword in keywords:
            pattern = re.compile(rf"\b{keyword}\b", re.IGNORECASE)
            matches = pattern.findall(doc.text)
            for match in matches:
                universities.add(match.strip())

    
    return list(universities)[:1]  

# ----------------------------------Extract Skills--------------------------------
def csv_skills(doc):
    skills_keywords = load_keywords('data/newSkills.csv')
    skills = set()

    for keyword in skills_keywords:
        if keyword.lower() in doc.text.lower():
            skills.add(keyword)

    return skills

nlp_skills = spacy.load('TrainedModel/skills') 

def extract_skills_from_ner(doc):
    non_skill_labels = {'DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL', 'EMAIL'}
    
    skills = set()
    for ent in nlp_skills(doc.text).ents:
        if ent.label_ == 'SKILLS':
           
            if ent.label_ not in non_skill_labels and not ent.text.isdigit():
                
                skill_text = ''.join(filter(str.isalpha, ent.text))
                if skill_text:
                    skills.add(skill_text)
    return skills

def is_valid_skill(skill_text):
    
    return len(skill_text) > 1 and not any(char.isdigit() for char in skill_text)

def extract_skills(doc):
    skills_csv = csv_skills(doc)
    skills_ner = extract_skills_from_ner(doc)
    
    filtered_skills_csv = {skill for skill in skills_csv if is_valid_skill(skill)}
    filtered_skills_ner = {skill for skill in skills_ner if is_valid_skill(skill)}
    
    combined_skills = filtered_skills_csv.union(filtered_skills_ner)  
    return list(combined_skills)  

def generate_search_term(skills):
    # Select top skills (you can tweak the number of skills used)
    # Convert the list of skills into a single search term
    if skills:
        search_term = " ".join(skills[:20])  # You can adjust how many skills to use in the search
      # Default fallback if no skills are found
    
    return search_term

# --------------------------------------------------------------------------------

# ----------------------------------Extract Major---------------------------------
def extract_major(doc):
    major_keywords = load_keywords('data/majors.csv')

    for keyword in major_keywords:
        if keyword.lower() in doc.text.lower():
            return keyword

    return ""
# --------------------------------------------------------------------------------

# --------------------------------Extract Experience-------------------------------
def extract_experience(doc):
    verbs = [token.text for token in doc if token.pos_ == 'VERB']

    senior_keywords = ['lead', 'manage', 'direct', 'oversee', 'supervise', 'orchestrate', 'govern']
    mid_senior_keywords = ['develop', 'design', 'analyze', 'implement', 'coordinate', 'execute', 'strategize']
    mid_junior_keywords = ['assist', 'support', 'collaborate', 'participate', 'aid', 'facilitate', 'contribute']
    
    if any(keyword in verbs for keyword in senior_keywords):
        level_of_experience = "Senior"
    elif any(keyword in verbs for keyword in mid_senior_keywords):
        level_of_experience = "Mid-Senior"
    elif any(keyword in verbs for keyword in mid_junior_keywords):
        level_of_experience = "Mid-Junior"
    else:
        level_of_experience = "Entry Level"

    suggested_position = suggest_position(verbs)

    return {
        'level_of_experience': level_of_experience,
        'suggested_position': suggested_position
    }

# --------------------------------------------------------------------------------
# ----------------------------------Fetch Job Listings----------------------------------

# -----------------------------------Suggestions----------------------------------
def load_positions_keywords(file_path):
    positions_keywords = {}
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            position = row['position']
            keywords = [keyword.lower()
                        for keyword in row['keywords'].split(',')]
            positions_keywords[position] = keywords
    return positions_keywords


def suggest_position(verbs):
    positions_keywords = load_positions_keywords('data/position.csv')
    verbs = [verb.lower() for verb in verbs]
    for position, keywords in positions_keywords.items():
        if any(keyword in verbs for keyword in keywords):
            return position

    return "Position Not Identified"


def extract_resume_info_from_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page_num in range(doc.page_count):
        page = doc[page_num]
        text += page.get_text()
    return nlp(text)




def show_colored_skills(skills):
    st.write(', '.join(skills))
def calculate_resume_score(resume_info):
    score = 0
    max_score = 100
    
    
    weights = {
        'first_name': 5,
        'last_name': 5,
        'email': 5,
        'degree_major': 25,
        'skills': 35,
        'experience': 25
    }

    
    def calculate_skill_score(skills, max_skill_weight):
        skill_count = len(skills)
        unique_skills = len(set(skills))
        
       
        diversity_score = unique_skills / skill_count if skill_count else 0
        return min(max_skill_weight, (diversity_score * skill_count) * (max_skill_weight / 10))

    
    def calculate_experience_score(experience, max_experience_weight):
        
        years_of_experience = experience.get('years', 0)
        level_of_experience = experience.get('level_of_experience', 'Entry Level')
        
     
        level_multiplier = {'Entry Level': 0.5, 'Mid-Junior': 0.7, 'Mid-Senior': 0.9, 'Senior': 1.0}
        score = min(max_experience_weight, years_of_experience * level_multiplier.get(level_of_experience, 0.5) * (max_experience_weight / 10))
        
        return score

    
    if resume_info['first_name']:
        score += weights['first_name']
    if resume_info['last_name']:
        score += weights['last_name']
    if resume_info['email']:
        score += weights['email']
    if resume_info['degree_major']:
        score += weights['degree_major']
    if resume_info['skills']:
        score += calculate_skill_score(resume_info['skills'], weights['skills'])
    if resume_info.get('experience'):
        score += calculate_experience_score(resume_info['experience'], weights['experience'])
    
   
    score = min(score, max_score)
    
    return score
def improvement_suggestions(score):
    suggestions = []

    
    if score < 30:
        suggestions.append(
            "Your resume score indicates significant room for improvement. Consider the following actions:\n"
            "- **Personal Details:** Ensure that all your personal details, including name, email, and phone number, are accurately presented. Double-check for typos, incorrect formatting, or missing information. Make sure your contact information is prominently displayed and easy to locate.\n"
            "- **Resume Structure:** Review the overall structure of your resume. It should be well-organized and easy to read. Use clear headings and bullet points to make the information accessible. Ensure consistency in formatting, font style, and size.\n"
            "- **Content Completeness:** Ensure that all relevant sections are included and thoroughly filled out. This includes personal details, education, work experience, skills, and any additional sections like certifications or awards. Each section should be comprehensive and provide valuable information to potential employers.\n"
            "- **Grammar and Spelling:** Carefully proofread your resume for grammatical errors, spelling mistakes, and awkward phrasing. Consider using tools like grammar checkers or having someone else review your resume to catch any issues you might have missed."
        )
    elif score < 35:
        suggestions.append(
            "To improve your resume further:\n"
            "- **Objective Statement:** Consider adding an objective statement or professional summary to clearly articulate your career goals and what you bring to potential employers. This should provide a concise overview of your skills and aspirations.\n"
            "- **Education Details:** Provide more detailed information about your educational background. Include relevant coursework, honors, and GPA if applicable. This can help to strengthen your qualifications, especially if you are early in your career.\n"
            "- **Formatting Consistency:** Ensure that formatting is consistent throughout your resume. This includes the use of fonts, bullet points, and spacing. Consistency helps in presenting a professional image.\n"
            "- **Work Experience Details:** Add more detail to your work experience. Focus on specific responsibilities and accomplishments in each role to provide a clearer picture of your contributions."
        )
    elif score < 40:
        suggestions.append(
            "Focus on refining these areas:\n"
            "- **Achievements:** Emphasize any notable achievements or awards in your work experience or education sections. This could include employee of the month awards, academic honors, or successful project completions.\n"
            "- **Professional Experience:** Detail any internships, volunteer work, or side projects that demonstrate relevant skills and experiences. Even if these are not directly related to your career goals, they can show your dedication and ability to apply skills in various contexts.\n"
            "- **Keywords Optimization:** Incorporate relevant keywords from job descriptions into your resume. This helps in passing Applicant Tracking Systems (ATS) and ensures that your resume aligns with the job requirements.\n"
            "- **Action-Oriented Language:** Use strong action verbs and impactful language to describe your responsibilities and achievements. This makes your contributions more compelling and highlights your proactive approach."
        )
    elif score < 45:
        suggestions.append(
            "Consider these enhancements:\n"
            "- **Skills Relevance:** Make sure your skills section aligns closely with the job you're applying for. Use job descriptions to identify key skills and include those that are most relevant to the role.\n"
            "- **Experience Impact:** In your experience section, highlight how your work has had a positive impact. For example, mention any improvements you made, problems you solved, or innovations you introduced.\n"
            "- **Professional Development:** Include any relevant professional development activities, such as courses, workshops, or certifications. This demonstrates your commitment to continuous learning and growth.\n"
            "- **Portfolio Inclusion:** If applicable, include a link to your portfolio or relevant projects that showcase your work. This provides concrete evidence of your skills and accomplishments."
        )
    elif score < 50:
        suggestions.append(
            "To push your resume to the next level:\n"
            "- **Formatting and Design:** Review the overall design of your resume. Ensure that it is visually appealing and professionally formatted. Use white space effectively and ensure that headings and bullet points are consistent.\n"
            "- **Keywords and Phrases:** Incorporate relevant keywords and phrases from the job descriptions you are targeting. This can help your resume pass through Applicant Tracking Systems (ATS) and catch the attention of hiring managers.\n"
            "- **Quantitative Evidence:** Where possible, include quantitative evidence of your accomplishments. Numbers such as percentages, dollar amounts, or time frames can make your achievements more compelling and credible.\n"
            "- **Tailoring for Each Job:** Customize your resume for each job application by highlighting the most relevant skills and experiences for each specific job description."
        )
    elif score < 55:
        suggestions.append(
            "Further refine your resume with these tips:\n"
            "- **Achievements Detailing:** Provide more detailed descriptions of your achievements. Focus on specific results and the impact you had in your previous roles.\n"
            "- **Industry Relevance:** Ensure that your resume reflects current industry trends and requirements. Update your skills and experiences to align with the latest developments in your field.\n"
            "- **Professional Summary Enhancement:** Revise your professional summary to make it more impactful. Highlight key skills and experiences that align with your career goals and the positions you're targeting.\n"
            "- **Personal Branding:** Develop a personal brand statement that conveys your unique value proposition and differentiates you from other candidates."
        )
    elif score < 60:
        suggestions.append(
            "Enhance your resume by focusing on:\n"
            "- **Professional Summary:** Strengthen your professional summary by making it more specific and focused. Highlight your most important skills and experiences that are directly relevant to the position you are applying for.\n"
            "- **Leadership Roles:** Highlight any leadership roles or responsibilities you have had. This can include leading projects, managing teams, or taking initiative in your previous positions.\n"
            "- **Networking Achievements:** Include any notable networking achievements or industry connections that may add value to your application. This could be significant collaborations or endorsements from industry professionals.\n"
            "- **Detailed Skills Section:** Expand your skills section to include specific examples of how you have applied these skills in your work or projects."
        )
    elif score < 65:
        suggestions.append(
            "Improve in these areas:\n"
            "- **Skills and Certifications:** Add any additional certifications or advanced skills that may be relevant to your industry. Consider taking online courses or earning certifications to boost your qualifications.\n"
            "- **Tailored Content:** Ensure that each section of your resume is tailored to highlight your strengths in relation to the job you are applying for. Avoid using generic content and focus on showcasing what makes you a strong candidate.\n"
            "- **Project Descriptions:** Include detailed descriptions of any significant projects you have worked on. Highlight your role, the challenges faced, and the outcomes achieved.\n"
            "- **Cross-Functional Skills:** Emphasize any cross-functional skills or experiences that demonstrate your ability to work effectively across different areas or teams."
        )
    elif score < 70:
        suggestions.append(
            "Consider these detailed improvements:\n"
            "- **Projects and Portfolio:** Include a section dedicated to specific projects or a portfolio if applicable. Highlight significant projects with clear descriptions of your role, the outcomes, and the technologies or methods used.\n"
            "- **Soft Skills:** Highlight soft skills that are important for the role you are applying for. Provide examples of how you have demonstrated these skills in your previous positions or projects.\n"
            "- **Strategic Career Goals:** Clearly articulate your career trajectory and strategic goals in your resume. Outline how your past experiences align with your long-term career aspirations.\n"
            "- **Global Experience:** If applicable, include experiences or skills that demonstrate a global perspective or international experience. This can be particularly valuable for roles that require cross-cultural competency or global awareness."
        )
    elif score < 75:
        suggestions.append(
            "To further elevate your resume:\n"
            "- **Professional Branding:** Develop a personal brand statement that reflects your unique value proposition. This statement should succinctly convey what sets you apart from other candidates.\n"
            "- **References and Testimonials:** If possible, include references or testimonials from past employers or colleagues. Positive endorsements can strengthen your credibility and provide additional validation of your skills and achievements.\n"
            "- **Industry Involvement:** Show evidence of your involvement in industry-related activities, such as speaking at conferences, participating in industry panels, or contributing to professional publications.\n"
            "- **Advanced Certifications:** Pursue and include advanced certifications relevant to your field. This demonstrates a commitment to continuous learning and professional development."
        )
    elif score < 80:
        suggestions.append(
            "Enhance these aspects:\n"
            "- **Industry Trends:** Demonstrate awareness of industry trends and how you have adapted to or leveraged these trends in your work. This shows that you are proactive and engaged with the evolving demands of your field.\n"
            "- **Professional Development:** Show evidence of ongoing professional development, such as attending workshops, conferences, or participating in relevant professional organizations.\n"
            "- **Strategic Networking:** Highlight strategic networking efforts, such as building relationships with industry leaders or participating in influential professional groups.\n"
            "- **Innovative Projects:** Showcase any innovative or unique projects that highlight your creativity and problem-solving skills. Provide details on how these projects were developed and their impact."
        )
    elif score < 85:
        suggestions.append(
            "Focus on refining these details:\n"
            "- **Tailored Cover Letter: Pair your resume with a tailored cover letter that complements your resume and addresses the specific requirements of the job. Use the cover letter to elaborate on key aspects of your resume and provide additional context.\n"
            "- Networking Evidence: If applicable, include any networking or professional connections that can add value to your application. This could include collaborations with notable individuals or affiliations with industry leaders.\n"
            "- Quantitative Impact: Emphasize the quantitative impact of your achievements. Include specific metrics, such as revenue growth, cost savings, or efficiency improvements, to demonstrate the effectiveness of your contributions.\n"
            "- Professional Accomplishments: Highlight any major professional accomplishments that set you apart. This could include awards, recognitions, or significant milestones in your career."
        )
    elif score < 90:
        suggestions.append(
            "Refine your resume with these advanced strategies:\n"
            "- Global Perspective: Incorporate any global or international experiences that demonstrate your ability to work in diverse environments or on a global scale. This can be particularly valuable for roles with an international focus.\n"
            "- High-Impact Projects: Showcase high-impact projects that had a significant effect on your organization or industry. Provide detailed descriptions of your contributions and the outcomes achieved.\n"
            "- Thought Leadership: Include evidence of thought leadership, such as publishing articles, giving presentations, or contributing to industry research. This demonstrates your expertise and influence in your field.\n"
            "- **Strategic Vision:** Clearly articulate your strategic vision and long-term goals. Show how your past experiences and achievements align with your future career aspirations and the positions you are targeting."
        )
    else:
        suggestions.append(
            "Your resume is highly impressive, but there are always ways to refine it further:\n"
            "- **Unique Projects:** Showcase any unique or innovative projects that demonstrate exceptional skills or creativity. Highlight projects that have received recognition or that make you stand out in your field.\n"
            "- **Global Perspective:** If applicable, include experiences or skills that demonstrate a global perspective or international experience. This can be particularly valuable for roles that require cross-cultural competency or global awareness.\n"
            "- **Personal Branding:** Continuously refine your personal brand to ensure it reflects your evolving career goals and achievements. This includes updating your resume, LinkedIn profile, and other professional materials.\n"
            "- **Professional Legacy:** Consider how you want to shape your professional legacy. Include any contributions to your field or industry that have had a lasting impact, such as mentoring others or initiating significant changes."
        )

    return " ".join(suggestions)


def extract_resume_info(doc):
    first_lines = '\n'.join(doc.text.splitlines()[:10])
    first_name, last_name = extract_name(doc)
    email = extract_email(doc)
    skills = extract_skills(doc)
    degree_major = extract_major(doc)
    experience = extract_experience(doc)

    return {'first_name': first_name, 'last_name': last_name, 'email': email, 'degree_major': degree_major, 'skills': skills, 'experience': experience}


def suggest_skills_for_job(desired_job):
    job_skills_mapping = {}
    
    with open('data/suggestedSkills.csv', newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            job_title = row[0].lower()
            skills = row[1:]
            job_skills_mapping[job_title] = skills
    
    desired_job_lower = desired_job.lower()
    if desired_job_lower in job_skills_mapping:
        suggested_skills = job_skills_mapping[desired_job_lower]
        return suggested_skills
    else:
        return []
    

def dynamic_job_search(resume_info):
    # Extract skills from resume info
    extracted_skills = resume_info.get('skills', [])
    
    # Generate search term dynamically
    if extracted_skills:
        search_term = ", ".join(extracted_skills)
    else:
        st.warning("No skills found in the resume. Using default search terms.")
        search_term = "software engineer, web developer, data scientist, manager, hr, writing, data entry, retail"
    
    # Scrape jobs dynamically based on extracted skills
    jobs = scrape_jobs(
        site_name=["indeed", "linkedin", "glassdoor"],
        search_term=search_term,
        results_wanted=10,
        hours_old=24,
        country_indeed='USA'
    )
    
    if not jobs.empty:
        st.success(f"Found {len(jobs)} job listings!")
        return jobs
    else:
        st.warning("No jobs found matching the extracted skills.")
        return pd.DataFrame()

def fetch_jobs_from_csv(file_path):  # Corrected parameter definition
    try:
        jobs = pd.read_csv(file_path)
        return jobs
    except FileNotFoundError:
        st.error("Job listings file not found.")
        return pd.DataFrame()
    
def display_job_listings(jobs):
    
    job_list = []
    for _, job in jobs.iterrows():
        job_data = {
            'title': job.get('title', 'No title'),
            'company': job.get('company', 'No company'),
            'location': job.get('location', 'No location'),
            'description': job.get('description', 'No description'),
            'posted': job.get('posted_date', 'No posted date'),
            'url': job.get('url', 'No URL')
        }
        job_list.append(job_data)

        # Display the job listing in the Streamlit app
        st.markdown(f"**Title:** {job_data['title']}")
        st.markdown(f"**Company:** {job_data['company']}")
        st.markdown(f"**Location:** {job_data['location']}")
        
        
      
        st.markdown("---")  # Separator between jobs

def score_resume_for_job(resume_info, job_description):



    # Extract key requirements from job description
    job_doc = nlp(job_description)
    job_skills = extract_skills(job_doc)
    
    # Match resume skills with job skills
    resume_skills = set(resume_info['skills'])
    matching_skills = resume_skills.intersection(set(job_skills))
    
    # Calculate match percentage
    match_score = len(matching_skills) / len(job_skills) * 100 if job_skills else 0
    
    # Calculate other relevance factors
    # ...
    
    return {
        'overall_match': match_score,
        'matching_skills': list(matching_skills),
        'missing_skills': list(set(job_skills) - resume_skills)
    }

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

    Jobs:
    {', '.join(limited_jobs)}

    Experience:
    Level of Experience: {experience_info['level_of_experience']}
    Suggested Position: {experience_info['suggested_position']}

    Resume Score: {resume_score}/100

 

    Improvements:
    {improvements}
    """
    return report_content


def extract_element(doc, element):
   
    if element == 'first_name':
        first_name, _ = extract_name(doc)
        return first_name
    elif element == 'last_name':
        _, last_name = extract_name(doc)
        return last_name
    elif element == 'skills':
        return extract_skills(doc)
    elif element == 'email':
        return extract_email(doc)
    elif element == 'major':
        return extract_major(doc)
    elif element == 'education':
        return extract_education_from_resume(doc.text)
    elif element == 'phone_number':
        return extract_contact_number_from_resume(doc)
    return None

# Example ground truth data (replace with your actual data)
ground_truth_data = [
    {
        'resume_text': """Dhruv Mittal
mdhruv367@gmail.com
91-8112290907
COMPUTER SCIENCE
Vellore Institution of Technology
Skills: Organization, Git, MongoDB, Problem-Solving, Collaboration, GitHub, Go, Leadership, OS, React, RESTful APIs, Forecasting, Initiative, Node.js, Communication, Cross-Functional Collaboration, Adaptability, JavaScript, Unity, Java, Express.js""",
        'ground_truth': {
            'first_name': 'Dhruv',
            'last_name': 'Mittal',
            'email': 'mdhruv367@gmail.com',
            'phone_number': '+918112290907',
            'major': 'COMPUTER SCIENCE',
            'education': ['Vellore Institution of Technology'],
            'skills': ['Organization', 'Git', '', 'Problem-Solving', 'Collaboration', 'GitHub', 'Go', 'Leadership', 'OS', 'React', 'RESTful APIs', 'Forecasting', 'Initiative', 'Node.js', 'Communication', 'Cross-Functional Collaboration', 'Adaptability', 'JavaScript', 'Unity', 'Java', 'Express.js']
        }
    },
        {
        'resume_text': """Jane Doe
jane.doe@example.com
123-456-7890
Mechanical Engineering
MIT
Skills: CAD, SolidWorks, MATLAB""",
        'ground_truth': {
            'first_name': 'Jane',
            'last_name': 'Doe',
            'email': 'jane.doe@example.com',
            'phone_number': '123-456-7890',
            'major': 'Mechanical Engineering',
            'education': ['MIT'],
            'skills': ['CAD', 'SolidWorks', 'MATLAB']
        }
    },
        {
        'resume_text': """Peter Smith
petersmith@example.net
(555) 987-6543
Electrical Engineering""",
        'ground_truth': {
            'first_name': 'Peter',
            'last_name': 'Smith',
            'email': 'petersmith@example.net',
            'phone_number': '(555) 987-6543',
            'major': 'Electrical Engineering',
            'education': [],
            'skills': []
        }
    },
    
    # ... Add many more diverse examples
]
def evaluate_parser(resumes_data):
    """Evaluates the resume parser using precision, recall, and rough accuracy for all ground truth elements."""
    all_metrics = {}
    try:
        if not resumes_data:
            print("Error: resumes_data is empty.")
            return None

        elements_in_ground_truth = set(resumes_data[0]['ground_truth'].keys())

        for element in elements_in_ground_truth:
            true_labels = []
            predicted_labels = []

            for resume in resumes_data:
                doc = nlp(resume['resume_text'])
                extracted_value = extract_element(doc, element)
                true_value = resume['ground_truth'].get(element)

                if isinstance(true_value, list):
                    extracted_value = extracted_value or []
                    all_items = list(set(true_value + extracted_value))
                    true_labels.extend([1 if item in true_value else 0 for item in all_items])
                    predicted_labels.extend([1 if item in extracted_value else 0 for item in all_items])
                else:
                    true_labels.append(1 if extracted_value == true_value else 0)
                    predicted_labels.append(1 if extracted_value is not None else 0)

            if any(true_labels):
                try:
                    precision = precision_score(true_labels, predicted_labels, zero_division=0)
                    recall = recall_score(true_labels, predicted_labels, zero_division=0)
                    rough_accuracy = (precision + recall) / 2.0  # Calculate rough accuracy
                    all_metrics[element] = {
                        'precision': precision,
                        'recall': recall,
                        'rough_accuracy': rough_accuracy  # Include rough accuracy
                    }
                except ValueError as e:
                    print(f"ValueError during metric calculation for {element}: {e}. True Labels: {true_labels}, Predicted Labels: {predicted_labels}")
            else:
                print(f"Warning: No true positive labels for element '{element}'. Skipping metrics calculation.")
        return all_metrics

    except IndexError:
        print("Error: resumes_data is incorrectly formatted.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


evaluation_results = evaluate_parser(ground_truth_data)

if evaluation_results:
    for element, metrics in evaluation_results.items():
        print(f"Metrics for {element}:")
        print(f"  Precision: {metrics['precision']:.2f}")
        print(f"  Recall: {metrics['recall']:.2f}")
        
else:
    print("Evaluation failed.")


# ...
'''
def show_pdf(uploaded_file):
    try:
        with open(uploaded_file.name, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    except AttributeError:
        base64_pdf = base64.b64encode(uploaded_file.read()).decode('utf-8')

    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

'''