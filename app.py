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

def recommend_with_inputs(user):
    recommendations = []

    for _, course in courses_df.iterrows():
        score = 0

        # 1. Interest match
        if user['interest'].lower() in course['course_name'].lower():
            score += 0.4

        # 2. Skill level match
        if user['skill_level'].lower() in course['course_name'].lower():
            score += 0.2

        # 3. Performance-based
        if user['avg_score'] < 60 and 'Beginner' in course['course_name']:
            score += 0.2
        elif user['avg_score'] > 75 and 'Advanced' in course['course_name']:
            score += 0.2

        # 4. Completion logic (NEW)
        if user['completed'] == "no":
            score += 0.1   # recommend similar courses
        else:
            score -= 0.05  # avoid repetition

        # 5. Time spent logic (NEW)
        if user['time_spent'] < 2:
            score += 0.1  # suggest short/easy courses
        elif user['time_spent'] > 5:
            score += 0.1  # suggest deeper courses

        recommendations.append((course['course_name'], course['course_url'], score))

    recommendations = sorted(recommendations, key=lambda x: x[2], reverse=True)

    return [{"name": r[0], "url": r[1]} for r in recommendations[:6]]

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

        recommended_courses = recommend_with_inputs(user)

    return render_template(
        'index.html',
        recommendations=recommended_courses
    )

if __name__ == '__main__':
    app.run(debug=True)