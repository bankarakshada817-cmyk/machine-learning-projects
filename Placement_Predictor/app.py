import os
import joblib
import PyPDF2
import docx2txt
from flask import Flask, render_template, request

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Uploads folder naseel tar create hoto
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 4 features vala trained model load karto
model = joblib.load('model.pkl')

def extract_text_from_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    text = ""
    try:
        if ext == '.pdf':
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted
        elif ext in ['.docx', '.doc']:
            text = docx2txt.process(file_path)
    except Exception as e:
        print(f"Error reading resume: {e}")
    return text.lower()

def analyze_resume(resume_text):
    key_skills = ['python', 'java', 'sql', 'machine learning', 'data analysis', 'html', 'css', 'javascript', 'git']
    found_skills = [skill for skill in key_skills if skill in resume_text]
    score = min(100, int((len(found_skills) / 5) * 100))
    return {
        'found_skills': found_skills,
        'skill_count': len(found_skills),
        'score': score
    }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # 1. Personal & Academic Inputs
            name = request.form['name']
            email = request.form['email']
            cgpa = float(request.form['cgpa'])
            projects = int(request.form['projects'])
            internships = int(request.form['internships'])
            
            # 2. Resume File Processing
            resume_file = request.files['resume']
            resume_analysis = {'found_skills': [], 'skill_count': 0, 'score': 0}
            
            if resume_file and resume_file.filename != '':
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], resume_file.filename)
                resume_file.save(file_path)
                
                resume_text = extract_text_from_file(file_path)
                resume_analysis = analyze_resume(resume_text)

            # 3. Model Input (Exact 4 features expected by SVC)
            features = [[cgpa, projects, internships, resume_analysis['score']]]
            
            # Prediction (0 ki 1)
            prediction = int(model.predict(features)[0])
            
            user_info = {
                'name': name,
                'email': email,
                'cgpa': cgpa,
                'projects': projects,
                'internships': internships
            }
            
            # Render Result Page
            return render_template('result.html', user_info=user_info, resume_analysis=resume_analysis, prediction=prediction)
        
        except Exception as e:
            print("Error during prediction:", e)
            return f"Error: {e}", 500

if __name__ == '__main__':
    app.run(debug=True)