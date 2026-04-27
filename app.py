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
    print("Error: One or more model files not found.")
    exit()
except Exception as e:
    print(f"Error loading model files: {e}")
    exit()

course_names = courses_df['course_name'].values.tolist()
course_url_dict = courses_df.set_index('course_name')['course_url'].to_dict()


# ─────────────────────────────────────────────
# IMPROVEMENT 1: Richer user analytics profile
# Returns computed insights about the learner
# ─────────────────────────────────────────────
def build_user_analytics(user):
    """Generate a human-readable analytics profile for the learner."""
    profile = {}

    # Performance tier
    if user['avg_score'] >= 80:
        profile['performance_tier'] = "High Performer"
        profile['performance_color'] = "green"
    elif user['avg_score'] >= 60:
        profile['performance_tier'] = "Steady Learner"
        profile['performance_color'] = "blue"
    else:
        profile['performance_tier'] = "Needs Support"
        profile['performance_color'] = "orange"

    # Engagement level
    if user['time_spent'] >= 4:
        profile['engagement'] = "Highly Engaged"
    elif user['time_spent'] >= 2:
        profile['engagement'] = "Moderately Engaged"
    else:
        profile['engagement'] = "Low Engagement"

    # Completion behaviour
    profile['completion_status'] = "Consistent Completer" if user['completed'] == "yes" else "Tends to Drop Out"

    # Learning recommendation
    if user['avg_score'] < 60 and user['skill_level'] == "Advanced":
        profile['insight'] = "⚠️ Score suggests a skill mismatch — consider bridging with Intermediate content first."
    elif user['avg_score'] >= 80 and user['skill_level'] == "Beginner":
        profile['insight'] = "🚀 Strong scores suggest you're ready to level up to Intermediate courses."
    elif user['completed'] == "no" and user['time_spent'] < 2:
        profile['insight'] = "💡 Short, focused courses may improve your completion rate."
    else:
        profile['insight'] = "✅ Your profile looks well-balanced. Keep going!"

    return profile


def filter_courses(user):
    filtered_indices = []
    for idx in range(len(courses_df)):
        difficulty = courses_df.iloc[idx]['difficulty_level']
        keep = True
        if user['skill_level'] == "Intermediate" and difficulty == "Beginner":
            keep = False
        if user['skill_level'] == "Advanced" and difficulty in ["Beginner", "Intermediate"]:
            keep = False
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
    valid_indices = filter_courses(user_input)
    filtered = [(i, distances[i]) for i in valid_indices]
    course_list = sorted(filtered, key=lambda x: x[1], reverse=True)[:20]

    recommendations = []
    for i in course_list:
        row = courses_df.iloc[i[0]]
        recommendations.append({
            "name": row.course_name,
            "url": row.course_url,
            "difficulty": row.difficulty_level,
            "similarity_score": float(i[1])
        })
    return recommendations


# ─────────────────────────────────────────────
# IMPROVEMENT 2: Explainability — WHY tags
# Each recommendation now carries reasons
# ─────────────────────────────────────────────
def build_why_tags(user, course_diff, base_rank, score):
    """Return a list of human-readable reason strings for a recommendation."""
    reasons = []

    if user['skill_level'] == course_diff:
        reasons.append(f"✅ Matches your {course_diff} level")
    if user['avg_score'] < 60 and course_diff == "Beginner":
        reasons.append("📈 Good for building foundational skills")
    if user['avg_score'] > 75 and course_diff == "Advanced":
        reasons.append("🏆 Aligns with your high performance")
    if user['completed'] == "no":
        reasons.append("🔁 Recommended to improve completion rate")
    if user['time_spent'] < 2:
        reasons.append("⏱️ Suitable for low-time learners")
    elif user['time_spent'] > 4:
        reasons.append("📚 Great for dedicated learners")
    if base_rank <= 3:
        reasons.append("🔥 Top content match for your interest")

    if not reasons:
        reasons.append("🎯 Strong similarity to your interest area")

    return reasons


# ─────────────────────────────────────────────
# IMPROVEMENT 3: Match % score for each course
# ─────────────────────────────────────────────
def compute_match_percent(score, max_score):
    if max_score == 0:
        return 50
    raw = (score / max_score) * 100
    # Normalize to 60–100% range for friendlier display
    normalized = 60 + (raw / 100) * 40
    return round(min(normalized, 99))


def hybrid_recommendation(user):
    ml_recs = recommend_with_similarity(user)
    scored = []

    for idx, course in enumerate(ml_recs):
        diff = course['difficulty']
        score = 0
        score += (20 - idx)
        if user['skill_level'] == diff:
            score += 8
        if user['avg_score'] < 60 and diff == "Beginner":
            score += 6
        elif user['avg_score'] > 75 and diff == "Advanced":
            score += 6
        if user['time_spent'] < 2:
            score += 3
        elif user['time_spent'] > 4:
            score += 6
        if user['completed'] == "no":
            score += 4
        scored.append((course, score, idx))

    scored = sorted(scored, key=lambda x: x[1], reverse=True)
    max_score = scored[0][1] if scored else 1

    results = []
    for rank, (course, score, original_rank) in enumerate(scored[:6]):
        why_tags = build_why_tags(user, course['difficulty'], original_rank, score)
        match_pct = compute_match_percent(score, max_score)

        # ── IMPROVEMENT 4: Learning path position ──
        if rank == 0:
            path_label = "Start Here"
            path_color = "indigo"
        elif rank <= 2:
            path_label = "Next Step"
            path_color = "blue"
        else:
            path_label = "Level Up"
            path_color = "purple"

        results.append({
            "name": course['name'],
            "url": course['url'],
            "difficulty": course['difficulty'],
            "match_percent": match_pct,
            "why_tags": why_tags,
            "path_label": path_label,
            "path_color": path_color,
            "rank": rank + 1
        })

    return results


@app.route('/', methods=['GET', 'POST'])
def index():
    recommended_courses = []
    user_analytics = None
    user = None

    if request.method == 'POST':
        user = {
            "interest": request.form['interest'],
            "skill_level": request.form['skill_level'],
            "avg_score": float(request.form['avg_score']),
            "completed": request.form['completed'],
            "time_spent": float(request.form['time_spent'])
        }
        recommended_courses = hybrid_recommendation(user)

        # IMPROVEMENT 1: Pass analytics profile to template
        user_analytics = build_user_analytics(user)

    return render_template(
        'index.html',
        recommendations=recommended_courses,
        user_analytics=user_analytics,
        user=user
    )

if __name__ == '__main__':
    app.run(debug=True)