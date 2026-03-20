import os
from functools import lru_cache

import pandas as pd
import streamlit as st
import xgboost as xgb
from openai import OpenAI
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="Fitness AI Predictor",
    page_icon="💪",
    layout="wide"
)

# ---------------------------
# Config
# ---------------------------
CSV_PATH = "bodyPerformance.csv"
IMAGE_PATH = "fitness.jpg"

RENAME_MAP = {
    "body fat_%": "body_fat",
    "gripForce": "grip_force",
    "sit and bend forward_cm": "sit_bend_forward",
    "sit-ups counts": "sit_ups",
    "broad jump_cm": "broad_jump",
    "class": "target",
    "BMI": "bmi"
}

FEATURE_COLS = [
    "age",
    "gender",
    "height_cm",
    "weight_kg",
    "body_fat",
    "diastolic",
    "systolic",
    "grip_force",
    "sit_bend_forward",
    "sit_ups",
    "broad_jump",
    "bmi",
    "strength_ratio"
]

TARGET_COL = "target"

TARGET_MAP = {
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3,
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3
}

REVERSE_TARGET_MAP = {
    0: "A",
    1: "B",
    2: "C",
    3: "D"
}

# ---------------------------
# Styling
# ---------------------------
st.markdown("""
<style>
.stApp {
    background: #001a33;
    color: white;
}

.card {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.15);
}

.stButton>button,
.stFormSubmitButton>button {
    background: #0099cc;
    color: white;
}

.stButton>button:hover,
.stFormSubmitButton>button:hover {
    background: #00b3e6;
}

.block-container {
    padding-top: 2.5rem;
    padding-bottom: 2rem;
    max-width: 1100px;
}

.main-title {
    text-align: center;
    font-size: 2.7rem;
    font-weight: 800;
    color: white;
    margin-top: 1rem;
    margin-bottom: 0.5rem;
    line-height: 1.3;
    padding-top: 0.3rem;
}

.sub-title {
    text-align: center;
    color: #e6f5c9;
    margin-bottom: 1.5rem;
    font-size: 1.05rem;
}

.card {
    padding: 1.2rem 1.3rem;
    border-radius: 20px;
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.15);
    line-height: 1.8;
    color: white;
}

.stNumberInput label,
.stSelectbox label,
.stTextInput label,
.stMarkdown,
label,
p {
    color: white !important;
    font-weight: 600;
}

div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div {
    background: #f8fafc !important;
    border-radius: 12px !important;
    border: 1px solid #d1d5db !important;
}

input, textarea {
    color: #111827 !important;
    -webkit-text-fill-color: #111827 !important;
    font-weight: 600 !important;
}

input::placeholder,
textarea::placeholder {
    color: #6b7280 !important;
    opacity: 1 !important;
}

div[data-baseweb="select"] span {
    color: #111827 !important;
    font-weight: 600 !important;
}

div[data-baseweb="select"] svg,
div[data-baseweb="input"] svg {
    fill: #111827 !important;
    color: #111827 !important;
}

.stButton>button,
.stFormSubmitButton>button {
    background: #4a6b00 !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.8rem 1.2rem !important;
    font-weight: 700 !important;
    width: 100%;
}

.stButton>button:hover,
.stFormSubmitButton>button:hover {
    background: #5f8700 !important;
}

div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.12);
    padding: 14px;
    border-radius: 16px;
}

div[data-testid="stMetricLabel"] {
    color: #e6f5c9 !important;
}

div[data-testid="stMetricValue"] {
    color: white !important;
}

[data-testid="stDataFrame"] {
    background: white;
    border-radius: 14px;
    overflow: hidden;
}

div[data-testid="stAlert"] {
    border-radius: 14px !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Helpers
# ---------------------------
def friendly_feature_name(feature: str) -> str:
    mapping = {
        "age": "Age",
        "height_cm": "Height",
        "weight_kg": "Weight",
        "body_fat": "Body Fat",
        "diastolic": "Diastolic Blood Pressure",
        "systolic": "Systolic Blood Pressure",
        "grip_force": "Grip Force",
        "sit_bend_forward": "Sit and Bend Forward",
        "sit_ups": "Sit-Ups Count",
        "broad_jump": "Broad Jump",
        "bmi": "BMI",
        "strength_ratio": "Strength-to-Weight Ratio"
    }
    return mapping.get(feature, feature)


def calculate_bmi(weight_kg: float, height_cm: float) -> float:
    height_m = height_cm / 100
    if height_m <= 0:
        return 0.0
    return round(weight_kg / (height_m ** 2), 2)


def bmi_label(bmi: float) -> str:
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal"
    elif bmi < 30:
        return "Overweight"
    return "Obese"


def class_meaning(label: str) -> str:
    meanings = {
        "A": "Excellent fitness level",
        "B": "Good fitness level",
        "C": "Average fitness level",
        "D": "Needs improvement"
    }
    return meanings.get(label, "Unknown")


def generate_fitness_recommendation(predicted_class: str, top_feature: str) -> list[str]:
    focus = friendly_feature_name(top_feature)
    return [
        f"Follow a consistent weekly exercise routine suitable for class {predicted_class}.",
        f"Pay extra attention to improving {focus} through focused training.",
        "Track your progress every 2 to 4 weeks using the same fitness measurements."
    ]


def fallback_plan(predicted_class: str, top_feature: str) -> str:
    focus = friendly_feature_name(top_feature)
    return f"""
## Overall Assessment
Your predicted fitness class is **{predicted_class}**.

## Main Area to Improve
Your main focus area is **{focus}**.

## Starter 4-Week Plan
### Week 1
- 3 days walking for 20–25 minutes
- 2 days light full-body exercises
- 10 minutes stretching daily

### Week 2
- 3 days brisk walking or cycling for 25–30 minutes
- 2 days strength work: squats, wall push-ups, glute bridges
- 10–12 minutes mobility work

### Week 3
- 3 days cardio for 30 minutes
- 2 days strength work with 2–3 sets each
- Core training twice per week

### Week 4
- 3 days cardio for 30–35 minutes
- 3 strength sessions
- Flexibility session on recovery day

## Weekly Schedule
- Day 1: Cardio + stretching
- Day 2: Full body strength
- Day 3: Recovery walk
- Day 4: Core + flexibility
- Day 5: Full body strength
- Day 6: Cardio
- Day 7: Rest

## Recovery and Nutrition
- Sleep 7–9 hours
- Drink water regularly
- Eat enough protein
- Reduce highly processed foods

## Motivation
Focus on consistency. Small progress every week matters more than perfection.
"""


def generate_ai_coach(result_data: dict, user_data: dict, api_key: str) -> str:
    client = OpenAI(api_key=api_key)

    bmi = calculate_bmi(user_data["weight_kg"], user_data["height_cm"])
    main_focus = friendly_feature_name(result_data["top_feature"])

    prompt = f"""
You are a professional fitness coach and exercise specialist.

The machine learning model predicted:
- Fitness class: {result_data['predicted_class']}
- Meaning: {result_data['class_meaning']}
- Main focus area: {main_focus}
- Model explanation: {result_data['explanation']}

User measurements:
- Age: {user_data['age']}
- Gender: {user_data['gender']}
- Height: {user_data['height_cm']} cm
- Weight: {user_data['weight_kg']} kg
- BMI: {bmi}
- Body fat: {user_data['body_fat']}%
- Blood pressure: {user_data['systolic']}/{user_data['diastolic']}
- Grip force: {user_data['grip_force']}
- Sit and bend forward: {user_data['sit_bend_forward']} cm
- Sit-ups count: {user_data['sit_ups']}
- Broad jump: {user_data['broad_jump']} cm
- Strength-to-weight ratio: {round(user_data['grip_force'] / user_data['weight_kg'], 4)}

Create a detailed personalized fitness plan.

Return the answer using these exact sections:
1. Overall Assessment
2. What This Fitness Class Means
3. Main Weak Area to Improve
4. 4-Week Workout Plan
5. Weekly Schedule
6. Exercise Instructions
7. Recovery and Sleep Advice
8. Nutrition Tips
9. Motivation

Rules:
- Keep the plan practical for a beginner to intermediate user.
- Mention sets, reps, and rest time when useful.
- Keep language clear and supportive.
- Do not diagnose medical conditions.
"""

    response = client.responses.create(
        model="gpt-5.4",
        input=prompt
    )
    return response.output_text


def generate_ai_reason(result_data: dict, user_data: dict, api_key: str) -> str:
    client = OpenAI(api_key=api_key)

    bmi = calculate_bmi(user_data["weight_kg"], user_data["height_cm"])
    strength_ratio = round(user_data["grip_force"] / user_data["weight_kg"], 4)

    prompt = f"""
You are a fitness assessment expert.

A machine learning model predicted this user as class {result_data['predicted_class']}.
Class meaning: {result_data['class_meaning']}

User data:
- Age: {user_data['age']}
- Gender: {user_data['gender']}
- Height: {user_data['height_cm']} cm
- Weight: {user_data['weight_kg']} kg
- BMI: {bmi}
- Body fat: {user_data['body_fat']}%
- Diastolic: {user_data['diastolic']}
- Systolic: {user_data['systolic']}
- Grip force: {user_data['grip_force']}
- Sit and bend forward: {user_data['sit_bend_forward']} cm
- Sit-ups: {user_data['sit_ups']}
- Broad jump: {user_data['broad_jump']} cm
- Strength-to-weight ratio: {strength_ratio}

Top model feature:
- {friendly_feature_name(result_data['top_feature'])}

Write a short, clear explanation of WHY this result happened.

Requirements:
- Mention the likely weak areas using the user numbers.
- Explain in simple English.
- If the class is D, clearly say what most likely pushed it down.
- End with 3 short improvement points.
- Keep it concise and practical.
"""

    response = client.responses.create(
        model="gpt-5.4",
        input=prompt
    )
    return response.output_text


# ---------------------------
# Data + Model
# ---------------------------
def load_data() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    df = df.rename(columns=RENAME_MAP)
    return df


def preprocess_data(df: pd.DataFrame):
    if df is None:
        raise ValueError("Dataset is None.")

    df = df.copy()
    df = df.rename(columns=RENAME_MAP)

    # add bmi if not already in file
    if "bmi" not in df.columns:
        df["bmi"] = df["weight_kg"] / ((df["height_cm"] / 100) ** 2)

    # add engineered feature
    df["strength_ratio"] = df["grip_force"] / df["weight_kg"]

    required_cols = FEATURE_COLS + [TARGET_COL]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in dataset: {missing_cols}")

    gender_series = df["gender"].astype(str).str.strip().str.upper()
    gender_map = {
        "F": 0,
        "M": 1,
        "0": 0,
        "1": 1
    }
    df["gender"] = gender_series.map(gender_map)

    target_series = df[TARGET_COL].astype(str).str.strip().str.upper()
    y = target_series.map(TARGET_MAP)

    X = df[FEATURE_COLS].copy()

    if X.isnull().any().any():
        null_cols = X.columns[X.isnull().any()].tolist()
        raise ValueError(f"Some feature values could not be encoded correctly. Problem columns: {null_cols}")

    if y.isnull().any():
        bad_vals = sorted(df[TARGET_COL].dropna().astype(str).unique().tolist())
        raise ValueError(f"Unexpected target values found: {bad_vals}")

    return X, y


@lru_cache
def get_model_bundle():
    df = load_data()
    X, y = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # new tuned model
    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=4,
        eval_metric="mlogloss",
        random_state=42,
        tree_method="hist",
        n_estimators=500,
        max_depth=7,
        learning_rate=0.08,
        subsample=1.0,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=0.5,
        reg_alpha=0,
        reg_lambda=1.5
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    importance_df = pd.DataFrame({
        "Feature": FEATURE_COLS,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

    return model, acc, f1, importance_df


def predict_local(payload: dict) -> dict:
    model, acc, f1, importance_df = get_model_bundle()

    bmi = payload.get("bmi")
    if bmi is None:
        bmi = payload["weight_kg"] / ((payload["height_cm"] / 100) ** 2)

    strength_ratio = payload["grip_force"] / payload["weight_kg"]

    input_df = pd.DataFrame([{
        "age": payload["age"],
        "gender": 1 if str(payload["gender"]).strip().upper() == "M" else 0,
        "height_cm": payload["height_cm"],
        "weight_kg": payload["weight_kg"],
        "body_fat": payload["body_fat"],
        "diastolic": payload["diastolic"],
        "systolic": payload["systolic"],
        "grip_force": payload["grip_force"],
        "sit_bend_forward": payload["sit_bend_forward"],
        "sit_ups": payload["sit_ups"],
        "broad_jump": payload["broad_jump"],
        "bmi": bmi,
        "strength_ratio": strength_ratio
    }])

    pred_num = int(model.predict(input_df)[0])
    pred_label = REVERSE_TARGET_MAP[pred_num]
    top_feature = str(importance_df.iloc[0]["Feature"])

    explanation = (
        f"The model predicted class {pred_label}. "
        f"The strongest global signal in the model is currently {friendly_feature_name(top_feature)}."
    )

    recommendation = generate_fitness_recommendation(
        predicted_class=pred_label,
        top_feature=top_feature
    )

    return {
        "predicted_class": pred_label,
        "class_meaning": class_meaning(pred_label),
        "top_feature": top_feature,
        "explanation": explanation,
        "recommendation": recommendation,
        "model_accuracy": round(float(acc), 4),
        "model_f1": round(float(f1), 4)
    }

# ---------------------------
# Header
# ---------------------------
st.markdown("<div class='main-title'>💪 Fitness AI Predictor</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='sub-title'>Enter your body measurements to predict your fitness class and get an AI coaching plan</div>",
    unsafe_allow_html=True
)

# ---------------------------
# Image
# ---------------------------
if os.path.exists(IMAGE_PATH):
    st.image(IMAGE_PATH, use_container_width=True)
else:
    st.warning(f"Image not found. Expected file: {IMAGE_PATH}")

st.markdown("## 🟢 Prediction Form")

# ---------------------------
# Prediction Form
# ---------------------------
with st.form("predict_form"):
    a1, a2 = st.columns(2)

    with a1:
        age = st.number_input("Age", min_value=10, max_value=100, value=25)
        gender = st.selectbox("Gender", ["M", "F"])
        height_cm = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=170.0)
        weight_kg = st.number_input("Weight (kg)", min_value=30.0, max_value=250.0, value=70.0)
        grip_force = st.number_input("Grip Force", min_value=0.0, max_value=100.0, value=35.0)
        sit_ups = st.number_input("Sit-Ups Count", min_value=0, max_value=200, value=20)

    with a2:
        body_fat = st.number_input("Body Fat (%)", min_value=0.0, max_value=70.0, value=20.0)
        diastolic = st.number_input("Diastolic", min_value=30.0, max_value=200.0, value=80.0)
        systolic = st.number_input("Systolic", min_value=50.0, max_value=300.0, value=120.0)
        sit_bend_forward = st.number_input("Sit and Bend Forward (cm)", min_value=-50.0, max_value=50.0, value=10.0)
        broad_jump = st.number_input("Broad Jump (cm)", min_value=0.0, max_value=400.0, value=180.0)

    st.markdown("### 🔐 OpenAI API Key")
    api_key = st.text_input(
        "Paste your OpenAI API key here",
        type="password",
        help="Used only to generate the AI plan and the reason behind the prediction."
    )

    submitted = st.form_submit_button("Predict + Generate AI Advice")

# ---------------------------
# Prediction Logic
# ---------------------------
if submitted:
    payload = {
        "age": int(age),
        "gender": gender,
        "height_cm": float(height_cm),
        "weight_kg": float(weight_kg),
        "body_fat": float(body_fat),
        "diastolic": float(diastolic),
        "systolic": float(systolic),
        "grip_force": float(grip_force),
        "sit_bend_forward": float(sit_bend_forward),
        "sit_ups": int(sit_ups),
        "broad_jump": float(broad_jump)
    }

    try:
        with st.spinner("Running ML prediction..."):
            result = predict_local(payload)

        predicted_class = result["predicted_class"]
        class_meaning_text = result["class_meaning"]
        top_feature = friendly_feature_name(result["top_feature"])
        bmi = calculate_bmi(payload["weight_kg"], payload["height_cm"])

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Predicted Class", predicted_class)
        r2.metric("Accuracy", result["model_accuracy"])
        r3.metric("Macro F1", result["model_f1"])
        r4.metric("BMI", bmi)

        if predicted_class == "A":
            st.success(f"Fitness Level: {class_meaning_text}")
        elif predicted_class == "B":
            st.info(f"Fitness Level: {class_meaning_text}")
        elif predicted_class == "C":
            st.warning(f"Fitness Level: {class_meaning_text}")
        else:
            st.error(f"Fitness Level: {class_meaning_text}")

        c1, c2 = st.columns(2)

        with c1:
            st.markdown("### 📌 Main Focus Area")
            st.write(top_feature)

        with c2:
            st.markdown("### 🧾 BMI Status")
            st.write(bmi_label(bmi))

        st.markdown("### 🧠 Model Explanation")
        st.write(result["explanation"])

        st.markdown("### 📍 Basic Model Recommendations")
        for rec in result["recommendation"]:
            st.write(f"- {rec}")

        st.markdown("### 📋 Input Summary")
        summary_df = pd.DataFrame({
            "Metric": [
                "Age",
                "Gender",
                "Height (cm)",
                "Weight (kg)",
                "Body Fat (%)",
                "Diastolic",
                "Systolic",
                "Grip Force",
                "Sit and Bend Forward (cm)",
                "Sit-Ups Count",
                "Broad Jump (cm)",
                "BMI",
                "Strength-to-Weight Ratio"
            ],
            "Value": [
                age,
                gender,
                height_cm,
                weight_kg,
                body_fat,
                diastolic,
                systolic,
                grip_force,
                sit_bend_forward,
                sit_ups,
                broad_jump,
                bmi,
                round(grip_force / weight_kg, 4)
            ]
        })
        st.dataframe(summary_df, use_container_width=True)

        if api_key.strip():
            try:
                with st.spinner("Generating AI explanation..."):
                    ai_reason = generate_ai_reason(
                        result_data=result,
                        user_data=payload,
                        api_key=api_key
                    )
                st.markdown("## 🔥 Why this class?")
                st.markdown(ai_reason)
            except Exception as ai_error:
                st.warning(f"AI reason generation failed. Details: {ai_error}")

        if api_key.strip():
            try:
                with st.spinner("Generating AI coaching plan..."):
                    ai_text = generate_ai_coach(
                        result_data=result,
                        user_data=payload,
                        api_key=api_key
                    )
            except Exception as ai_error:
                st.warning(f"AI generation failed. Using fallback plan instead. Details: {ai_error}")
                ai_text = fallback_plan(result["predicted_class"], result["top_feature"])
        else:
            ai_text = fallback_plan(result["predicted_class"], result["top_feature"])

        st.markdown("## ✨ AI Fitness Coach Plan")
        st.markdown(ai_text)

    except Exception as e:
        st.error(f"Error: {e}")