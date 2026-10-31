from flask import Flask, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pypdf import PdfReader
import re

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB Limit for PDF uploads

# Master list of tech skills our engine scans for
SKILL_DATABASE = [
    "python", "java", "c++", "c", "javascript", "typescript", "html", "css",
    "react", "angular", "vue", "node.js", "express", "django", "flask", "fastapi",
    "sql", "mysql", "postgresql", "mongodb", "redis", "aws", "azure", "gcp",
    "docker", "kubernetes", "git", "github", "linux", "rest api", "graphql",
    "machine learning", "deep learning", "nlp", "tensorflow", "pytorch", 
    "scikit-learn", "pandas", "numpy", "data analysis", "power bi", "tableau",
    "cybersecurity", "networking", "oops", "data structures", "algorithms",
    "feature engineering", "data cleaning", "matplotlib", "logistic regression", 
    "linear regression", "decision trees", "clustering"
]

def extract_text_from_pdf(pdf_file):
    try:
        reader = PdfReader(pdf_file)
        extracted = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                extracted.append(page_text)
        return "\n".join(extracted)
    except Exception:
        return ""

def format_skill_name(s):
    return s.upper() if len(s) <= 3 else s.title()

@app.route('/', methods=['GET', 'POST'])
def home():
    match_score = None
    skill_score_pct = 0
    context_score_pct = 0
    found_skills = []
    missing_skills = []
    
    job_role = ""
    key_reqs = ""
    jd_text = ""
    resume_text = ""
    uploaded_filename = ""
    word_count = 0
    status_label = ""
    error_msg = ""

    if request.method == 'POST':
        job_role = request.form.get('job_role', '').strip()
        key_reqs = request.form.get('key_reqs', '').strip()
        jd_text = request.form.get('jd', '').strip()
        resume_text = request.form.get('resume', '').strip()

        # Handle PDF Upload
        pdf_file = request.files.get('resume_pdf')
        if pdf_file and pdf_file.filename != '':
            if pdf_file.filename.lower().endswith('.pdf'):
                uploaded_filename = pdf_file.filename
                pdf_extracted = extract_text_from_pdf(pdf_file)
                if pdf_extracted.strip():
                    resume_text = pdf_extracted
                else:
                    error_msg = "Could not extract text from this PDF. Please paste text instead."
            else:
                error_msg = "Please upload a valid .pdf file."

        if key_reqs and resume_text and not error_msg:
            word_count = len(resume_text.split())

            # Combine everything for TF-IDF Contextual Matching
            combined_jd = f"{job_role} {jd_text} {key_reqs}".strip()
            combined_lower = combined_jd.lower()
            resume_lower = resume_text.lower()

            # 1. SMART CUSTOM REQUIREMENT EXTRACTION (Ignores long sentences as tags)
            # Splits by commas, semicolons, newlines, or bullet points
            raw_reqs = re.split(r'[,;\n•]+', key_reqs)
            for r in raw_reqs:
                req = r.strip().lower()
                req = req.rstrip('.') # Remove trailing full stops
                
                # ONLY treat it as a visual Pill Tag if it's less than 35 characters (a proper skill/tool name)
                if req and len(req) <= 35:
                    formatted = format_skill_name(req)
                    # Smart substring match
                    if req in resume_lower:
                        if formatted not in found_skills:
                            found_skills.append(formatted)
                    else:
                        if formatted not in missing_skills:
                            missing_skills.append(formatted)

            # 2. EXTRACT STANDARD TECH SKILLS FROM PARAGRAPHS
            for skill in SKILL_DATABASE:
                pattern = r'\b' + re.escape(skill) + r'\b'
                # If the skill is mentioned anywhere in the user's requirements or JD
                if re.search(pattern, combined_lower):
                    formatted = format_skill_name(skill)
                    # Prevent duplicates if it was already caught by custom requirements
                    if formatted not in found_skills and formatted not in missing_skills:
                        if re.search(pattern, resume_lower):
                            found_skills.append(formatted)
                        else:
                            missing_skills.append(formatted)

            # 3. TF-IDF VECTORIZATION (Handles the overall paragraph meaning)
            vectorizer = TfidfVectorizer(stop_words='english')
            try:
                tfidf_matrix = vectorizer.fit_transform([combined_jd, resume_text])
                text_similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            except ValueError:
                text_similarity = 0.0

            # 4. HYBRID SCORING FORMULA
            contextual_score = min(text_similarity * 1.6, 1.0)
            context_score_pct = round(contextual_score * 100, 1)

            total_jd_skills = len(found_skills) + len(missing_skills)
            if total_jd_skills > 0:
                skill_ratio = len(found_skills) / total_jd_skills
                skill_score_pct = round(skill_ratio * 100, 1)
                final_score = (skill_ratio * 0.65) + (contextual_score * 0.35)
                match_score = round(final_score * 100, 1)
            else:
                skill_score_pct = context_score_pct
                match_score = context_score_pct

            # 5. VERDICT LABEL
            if match_score >= 70:
                status_label = "Strong Alignment — Ready to Apply"
            elif match_score >= 45:
                status_label = "Moderate Alignment — Add Missing Skills"
            else:
                status_label = "Low Alignment — Needs Tailoring"
        elif not error_msg:
            error_msg = "Please provide Key Requirements and a Candidate Resume."

    return render_template('index.html',
                           match_score=match_score,
                           skill_score_pct=skill_score_pct,
                           context_score_pct=context_score_pct,
                           found_skills=found_skills,
                           missing_skills=missing_skills,
                           job_role=job_role,
                           key_reqs=key_reqs,
                           jd_text=jd_text,
                           resume_text=resume_text,
                           uploaded_filename=uploaded_filename,
                           word_count=word_count,
                           status_label=status_label,
                           error_msg=error_msg)

if __name__ == '__main__':
    app.run(debug=True)