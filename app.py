from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the course list and similarity matrix
try:
    similarity = pickle.load(open('models/similarity.pkl', 'rb'))
    courses_df = pickle.load(open('models/courses.pkl', 'rb'))
    course_list_dicts = pickle.load(open('models/course_list.pkl', 'rb'))
    users_df = pd.read_csv('users.csv')
except FileNotFoundError:
    print("Error: One or more model files not found. Make sure 'models/similarity.pkl', 'models/courses.pkl', and 'models/course_list.pkl' exist.")
    exit()
except Exception as e:
    print(f"Error loading model files: {e}")
    exit()

course_names = courses_df['course_name'].values.tolist()
course_url_dict = courses_df.set_index('course_name')['course_url'].to_dict()


def filter_courses(user):
    filtered_indices = []

    for idx in range(len(courses_df)):
        difficulty = courses_df.iloc[idx]['difficulty_level']
        keep = True

        # Skill level filter - proper matching
        if user['skill_level'] == "Intermediate" and difficulty == "Beginner":
            keep = False
        if user['skill_level'] == "Advanced" and difficulty in ["Beginner", "Intermediate"]:
            keep = False

        # Score-based filtering
        if user['avg_score'] > 75 and difficulty == "Beginner":
            keep = False
        if user['avg_score'] < 50 and difficulty == "Advanced":
            keep = False

        if keep:
            filtered_indices.append(idx)

    return filtered_indices


def recommend_with_similarity(user_input):
    course_name = user_input['interest'] + " " + user_input['skill_level']

    if course_name not in course_names:
        matches = [c for c in course_names if user_input['interest'].lower() in c.lower()]
        if not matches:
            return []
        course_name = matches[0]

    index = course_names.index(course_name)
    distances = similarity[index]

    # Valid courses filter lagayein
    valid_indices = filter_courses(user_input)

    filtered = [(i, distances[i]) for i in valid_indices]

    # Sort karein
    course_list = sorted(filtered, key=lambda x: x[1], reverse=True)[:20]

    recommendations = []
    for i in course_list:
        row = courses_df.iloc[i[0]]
        # Difficulty bhi aage pass karein
        recommendations.append({
            "name": row.course_name, 
            "url": row.course_url,
            "difficulty": row.difficulty_level
        })

    return recommendations


def hybrid_recommendation(user):
    ml_recs = recommend_with_similarity(user)
    scored = []

    for idx, course in enumerate(ml_recs):
        name = course['name']
        diff = course['difficulty']
        score = 0

        # Base ML ranking weight
        score += (20 - idx)

        # 1. Skill match boost (ab ye 100% sahi kaam karega)
        if user['skill_level'] == diff:
            score += 8

        # 2. Performance logic
        if user['avg_score'] < 60 and diff == "Beginner":
            score += 6
        elif user['avg_score'] > 75 and diff == "Advanced":
            score += 6

        # 3. Time-based tuning (naam mein 'Short' dhoondhna reliable nahi tha)
        if user['time_spent'] < 2:
            score += 3
        elif user['time_spent'] > 4:
            score += 6

        # 4. Completion behavior
        if user['completed'] == "no":
            score += 4

        scored.append((course, score))

    # Sort based on final dynamic scores
    scored = sorted(scored, key=lambda x: x[1], reverse=True)

    return [item[0] for item in scored[:6]]


@app.route('/', methods=['GET', 'POST'])
def index():
    recommended_courses = []

    if request.method == 'POST':
        user = {
            "interest": request.form['interest'],
            "skill_level": request.form['skill_level'],
            "avg_score": float(request.form['avg_score']),
            "completed": request.form['completed'],
            "time_spent": float(request.form['time_spent'])
        }

        recommended_courses = hybrid_recommendation(user)
    return render_template(
        'index.html',
        recommendations=recommended_courses
    )

if __name__ == '__main__':
    app.run(debug=True)