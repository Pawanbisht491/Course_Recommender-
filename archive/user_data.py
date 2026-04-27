import pandas as pd
import random

interests = ['AI', 'Web Development', 'Data Science', 'Cybersecurity']
levels = ['Beginner', 'Intermediate', 'Advanced']
completion = ['Yes', 'No']

data = []

for i in range(1, 201):
    data.append({
        "user_id": i,
        "interest": random.choice(interests),
        "skill_level": random.choice(levels),
        "avg_score": random.randint(40, 95),
        "time_spent": random.randint(1, 5),
        "completed_courses": random.choice(completion)
    })

df = pd.DataFrame(data)
df.to_csv("users.csv", index=False)

print("Dataset created!")