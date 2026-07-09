import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from datetime import datetime


st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon="📊",
    layout="wide"
)


# Load Files

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")
label_encoders = joblib.load("label_encoders.pkl")
shap_background = joblib.load("shap_background.pkl")

explainer = shap.LinearExplainer(
    model,
    shap_background
)

# ==============================
# Mapping Dictionaries
# ==============================

education_map = {
    "Below College": 1,
    "College": 2,
    "Bachelor": 3,
    "Master": 4,
    "Doctor": 5
}

rating_map = {
    "Low": 1,
    "Medium": 2,
    "High": 3,
    "Very High": 4
}

performance_map = {
    "Excellent": 3,
    "Outstanding": 4
}

# ==============================
# Manual recommendation mapping
# feature -> targeted action when this feature is a top driver of risk
# ==============================

recommendation_map = {
    "OverTime": "Review workload and current overtime hours; frequent overtime is strongly linked to attrition.",
    "MonthlyIncome": "Conduct a salary review against market and internal peers.",
    "DailyRate": "Conduct a salary review against market and internal peers.",
    "HourlyRate": "Conduct a salary review against market and internal peers.",
    "MonthlyRate": "Conduct a salary review against market and internal peers.",
    "PercentSalaryHike": "Revisit the salary hike / compensation growth plan.",
    "YearsSinceLastPromotion": "Discuss promotion timeline and growth opportunities.",
    "JobLevel": "Discuss promotion timeline and growth opportunities.",
    "StockOptionLevel": "Review stock option / long-term incentive eligibility.",
    "EnvironmentSatisfaction": "Check in on team environment and workplace conditions.",
    "JobSatisfaction": "Schedule a one-on-one to understand job satisfaction concerns.",
    "RelationshipSatisfaction": "Check in on relationship with manager/peers.",
    "WorkLifeBalance": "Discuss flexible scheduling or workload adjustments to improve work-life balance.",
    "JobInvolvement": "Discuss engagement and involvement in current role/projects.",
    "DistanceFromHome": "Discuss remote/hybrid work options given commute distance.",
    "BusinessTravel": "Review business travel frequency and its impact on the employee.",
    "NumCompaniesWorked": "Note prior job-hopping history; consider a longer-term career conversation.",
    "TotalWorkingYears": "Discuss long-term career path within the company.",
    "YearsAtCompany": "Discuss long-term career path within the company.",
    "YearsInCurrentRole": "Discuss role rotation or new challenges within the company.",
    "YearsWithCurrManager": "Check the relationship and communication with the current manager.",
    "TrainingTimesLastYear": "Increase access to training and development opportunities.",
    "Department": "Review department-specific engagement and workload norms.",
    "JobRole": "Review role-specific expectations and support.",
    "Age": "Tailor retention approach to career stage (e.g. growth path for early career, stability for senior).",
    "MaritalStatus": "No direct action; consider alongside other personal/lifestyle factors.",
    "Education": "Discuss alignment between education background and current role.",
    "EducationField": "Discuss alignment between education background and current role.",
    "Gender": "No direct action.",
    "PerformanceRating": "Discuss recognition and feedback around recent performance.",
}

DEFAULT_RECOMMENDATION = "Conduct a one-on-one retention conversation to understand this factor better."



# Title

st.title("📊 Employee Attrition Prediction System")

st.markdown(
"""
Predict whether an employee is likely to leave the organization using Machine Learning.
"""
)

st.divider()


# Sidebar

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go To",
    [
        "Prediction",
        "About Model"
    ]
)


# ==============================
# Prediction Page
# ==============================

if page == "Prediction":

    st.header("Employee Information")

    # ==========================
    # Personal Information
    # ==========================

    with st.expander("👤 Personal Information", expanded=True):

        col1, col2 = st.columns(2)

        with col1:

            age = st.number_input(
                "Age",
                min_value=18,
                max_value=60,
                value=30
            )

            gender = st.selectbox(
                "Gender",
                ["Female", "Male"]
            )

            marital_status = st.selectbox(
                "Marital Status",
                ["Single", "Married", "Divorced"]
            )

            education = st.selectbox(
                "Education",
                [
                    "Below College",
                    "College",
                    "Bachelor",
                    "Master",
                    "Doctor"
                ]
            )

            education_field = st.selectbox(
                "Education Field",
                [
                    "Life Sciences",
                    "Medical",
                    "Marketing",
                    "Technical Degree",
                    "Human Resources",
                    "Other"
                ]
            )

        with col2:

            business_travel = st.selectbox(
                "Business Travel",
                [
                    "Travel_Rarely",
                    "Travel_Frequently",
                    "Non-Travel"
                ]
            )

            department = st.selectbox(
                "Department",
                [
                    "Sales",
                    "Research & Development",
                    "Human Resources"
                ]
            )

            job_role = st.selectbox(
                "Job Role",
                [
                    "Sales Executive",
                    "Research Scientist",
                    "Laboratory Technician",
                    "Manufacturing Director",
                    "Healthcare Representative",
                    "Manager",
                    "Sales Representative",
                    "Research Director",
                    "Human Resources"
                ]
            )

            overtime = st.selectbox(
                "OverTime",
                [
                    "No",
                    "Yes"
                ]
            )

    # ==========================
    # Job Information
    # ==========================

    with st.expander("💼 Job Information", expanded=True):

        col1, col2 = st.columns(2)

        with col1:

            daily_rate = st.number_input(
                "Daily Rate",
                min_value=100,
                max_value=1500,
                value=800
            )

            hourly_rate = st.number_input(
                "Hourly Rate",
                min_value=30,
                max_value=100,
                value=60
            )

            monthly_income = st.number_input(
                "Monthly Income",
                min_value=1000,
                value=5000
            )

            monthly_rate = st.number_input(
                "Monthly Rate",
                min_value=1000,
                value=15000
            )

            percent_salary_hike = st.slider(
                "Percent Salary Hike",
                10,
                30,
                15
            )

        with col2:

            job_level = st.selectbox(
                "Job Level",
                [1, 2, 3, 4, 5]
            )

            stock_option = st.selectbox(
                "Stock Option Level",
                [0, 1, 2, 3]
            )

            distance = st.slider(
                "Distance From Home",
                1,
                30,
                5
            )

            total_working_years = st.slider(
                "Total Working Years",
                0,
                40,
                10
            )

            companies = st.slider(
                "Number of Companies Worked",
                0,
                10,
                2
            )

    # ==========================
    # Satisfaction
    # ==========================

    with st.expander("😊 Satisfaction & Performance", expanded=True):

        col1, col2 = st.columns(2)

        satisfaction = [
            "Low",
            "Medium",
            "High",
            "Very High"
        ]

        with col1:

            environment = st.selectbox(
                "Environment Satisfaction",
                satisfaction
            )

            job_satisfaction = st.selectbox(
                "Job Satisfaction",
                satisfaction
            )

            relationship = st.selectbox(
                "Relationship Satisfaction",
                satisfaction
            )

        with col2:

            work_life = st.selectbox(
                "Work Life Balance",
                satisfaction
            )

            involvement = st.selectbox(
                "Job Involvement",
                satisfaction
            )

            performance = st.selectbox(
                "Performance Rating",
                [
                    "Excellent",
                    "Outstanding"
                ]
            )

    # ==========================
    # Experience
    # ==========================

    with st.expander("📈 Experience", expanded=True):

        col1, col2 = st.columns(2)

        with col1:

            years_company = st.slider(
                "Years At Company",
                0,
                40,
                5
            )

            years_role = st.slider(
                "Years In Current Role",
                0,
                20,
                3
            )

        with col2:

            promotion = st.slider(
                "Years Since Last Promotion",
                0,
                15,
                1
            )

            manager = st.slider(
                "Years With Current Manager",
                0,
                20,
                3
            )

            training = st.slider(
                "Training Times Last Year",
                0,
                10,
                2
            )

    predict = st.button(
        "Predict Attrition",
        use_container_width=True
    )

    if predict:

        # Convert UI values to numbers
        education = education_map[education]

        environment = rating_map[environment]
        job_satisfaction = rating_map[job_satisfaction]
        relationship = rating_map[relationship]
        work_life = rating_map[work_life]
        involvement = rating_map[involvement]

        performance = performance_map[performance]

        # Encode categorical columns
        business_travel = label_encoders["BusinessTravel"].transform([business_travel])[0]
        department = label_encoders["Department"].transform([department])[0]
        education_field = label_encoders["EducationField"].transform([education_field])[0]
        gender = label_encoders["Gender"].transform([gender])[0]
        job_role = label_encoders["JobRole"].transform([job_role])[0]
        marital_status = label_encoders["MaritalStatus"].transform([marital_status])[0]
        overtime = label_encoders["OverTime"].transform([overtime])[0]
        over18 = label_encoders["Over18"].transform(["Y"])[0]

        # Create input dataframe
        input_data = pd.DataFrame({
            "Age": [age],
            "BusinessTravel": [business_travel],
            "DailyRate": [daily_rate],
            "Department": [department],
            "DistanceFromHome": [distance],
            "Education": [education],
            "EducationField": [education_field],
            "EmployeeCount": [1],
            "EmployeeNumber": [1],
            "EnvironmentSatisfaction": [environment],
            "Gender": [gender],
            "HourlyRate": [hourly_rate],
            "JobInvolvement": [involvement],
            "JobLevel": [job_level],
            "JobRole": [job_role],
            "JobSatisfaction": [job_satisfaction],
            "MaritalStatus": [marital_status],
            "MonthlyIncome": [monthly_income],
            "MonthlyRate": [monthly_rate],
            "NumCompaniesWorked": [companies],
            "Over18": [over18],
            "OverTime": [overtime],
            "PercentSalaryHike": [percent_salary_hike],
            "PerformanceRating": [performance],
            "RelationshipSatisfaction": [relationship],
            "StandardHours": [80],
            "StockOptionLevel": [stock_option],
            "TotalWorkingYears": [total_working_years],
            "TrainingTimesLastYear": [training],
            "WorkLifeBalance": [work_life],
            "YearsAtCompany": [years_company],
            "YearsInCurrentRole": [years_role],
            "YearsSinceLastPromotion": [promotion],
            "YearsWithCurrManager": [manager]
        })

        # Match training column order
        input_data = input_data[feature_columns]

        # Scale
        input_scaled = scaler.transform(input_data)

        # Predict
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]

        # Explain
        input_scaled_df = pd.DataFrame(input_scaled, columns=feature_columns)
        shap_values = explainer(input_scaled_df)

        importance = pd.DataFrame({
            "Feature": feature_columns,
            "Impact": shap_values.values[0]
        })
        importance["Absolute"] = importance["Impact"].abs()
        importance = importance.sort_values(by="Absolute", ascending=False)

        confidence = probability * 100

        # store results in session_state so both tabs and the download
        # button can use them without recomputing
        st.session_state["prediction"] = prediction
        st.session_state["confidence"] = confidence
        st.session_state["importance"] = importance
        st.session_state["shap_values"] = shap_values

        tab1, tab2 = st.tabs(["Prediction", "Explain Prediction"])

        # ==========================
        # TAB 1 - Prediction
        # ==========================
        with tab1:

            st.subheader("Prediction Result")

            if prediction == 1:
                st.error("⚠️ Employee is likely to leave the company.")
            else:
                st.success("✅ Employee is likely to stay in the company.")

            st.metric(
                label="Attrition Probability",
                value=f"{confidence:.2f}%"
            )

            st.progress(float(probability))

            if confidence >= 80:
                risk_level = "High"
                st.warning("🔴 Risk Level : High")
            elif confidence >= 50:
                risk_level = "Medium"
                st.info("🟡 Risk Level : Medium")
            else:
                risk_level = "Low"
                st.success("🟢 Risk Level : Low")

            st.divider()

            st.subheader("Recommendations")

            if prediction == 1:
                # only features that pushed risk UP (positive impact),
                # ranked by how much they pushed it up
                risk_drivers = importance[importance["Impact"] > 0].sort_values(
                    by="Impact", ascending=False
                ).head(5)

                if len(risk_drivers) == 0:
                    st.markdown("- Conduct a one-on-one retention interview to understand risk factors.")
                else:
                    for _, row in risk_drivers.iterrows():
                        action = recommendation_map.get(row["Feature"], DEFAULT_RECOMMENDATION)
                        st.markdown(f"- **{row['Feature']}**: {action}")
            else:
                # what's currently working in this employee's favor -
                # the features pulling risk DOWN the most
                protective_factors = importance[importance["Impact"] < 0].sort_values(
                    by="Impact", ascending=True
                ).head(3)

                st.markdown("- Continue current employee engagement practices.")
                st.markdown("- Offer training and development opportunities.")
                st.markdown("- Monitor satisfaction periodically.")

                if len(protective_factors) > 0:
                    st.caption(
                        "Factors currently keeping attrition risk low for this employee: "
                        + ", ".join(protective_factors["Feature"].tolist())
                    )

            st.divider()

            st.subheader("Download Report")

            report_text = f"""Employee Attrition Prediction Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

Prediction: {"Will Leave" if prediction == 1 else "Will Stay"}
Attrition Probability: {confidence:.2f}%
Risk Level: {risk_level}

Top Contributing Features:
"""
            for _, row in importance.head(5).iterrows():
                direction = "increased" if row["Impact"] > 0 else "decreased"
                report_text += f"- {row['Feature']} {direction} attrition risk\n"

            report_text += "\nRecommendations:\n"
            if prediction == 1:
                risk_drivers_report = importance[importance["Impact"] > 0].sort_values(
                    by="Impact", ascending=False
                ).head(5)
                if len(risk_drivers_report) == 0:
                    report_text += "- Conduct a one-on-one retention interview to understand risk factors.\n"
                else:
                    for _, row in risk_drivers_report.iterrows():
                        action = recommendation_map.get(row["Feature"], DEFAULT_RECOMMENDATION)
                        report_text += f"- {row['Feature']}: {action}\n"
            else:
                report_text += "- Continue current employee engagement practices.\n"
                report_text += "- Offer training and development opportunities.\n"
                report_text += "- Monitor satisfaction periodically.\n"

            st.download_button(
                label="Download Report as TXT",
                data=report_text,
                file_name="attrition_prediction_report.txt",
                mime="text/plain",
                use_container_width=True
            )

        # ==========================
        # TAB 2 - Explain Prediction
        # ==========================
        with tab2:

            st.subheader("Top Features Driving This Prediction")

            st.dataframe(
                importance.head(10),
                use_container_width=True
            )

            st.subheader("Explanation")

            top = importance.head(5)

            for _, row in top.iterrows():
                if row["Impact"] > 0:
                    st.write(f"🔺 **{row['Feature']}** increased the likelihood of attrition.")
                else:
                    st.write(f"🔻 **{row['Feature']}** decreased the likelihood of attrition.")

            st.divider()

            st.subheader("SHAP Bar Plot")

            fig_bar = plt.figure(figsize=(8, 5))
            shap.plots.bar(
                shap_values[0],
                show=False
            )
            st.pyplot(fig_bar)

            st.divider()

            st.subheader("SHAP Waterfall Plot")

            fig_waterfall = plt.figure(figsize=(10, 6))
            shap.plots.waterfall(
                shap_values[0],
                show=False
            )
            st.pyplot(fig_waterfall)


# ==============================
# About Model Page
# ==============================

elif page == "About Model":

    st.header("About This Model")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Dataset")
        st.markdown(
            """
            - **Source:** IBM HR Employee Attrition Dataset
            - **Employees:** 1,470
            - **Features:** 34
            - **Task:** Binary Classification (Attrition: Yes / No)
            """
        )

        st.subheader("Model")
        st.markdown(
            """
            - **Algorithm:** Logistic Regression
            - **Class weighting:** Balanced (to reduce missed attrition cases)
            - **Preprocessing:** Label Encoding + Standard Scaling
            - **Explainability:** SHAP (LinearExplainer)
            """
        )

    with col2:

        st.subheader("Performance (test set, 20% holdout)")
        st.markdown(
            """
            - **Accuracy:** 75.2%
            - **ROC AUC:** 0.81
            - **Precision (Attrition = Yes):** 0.37
            - **Recall (Attrition = Yes):** 0.77
            - **F1 Score (Attrition = Yes):** 0.50
            """
        )
        st.caption(
            "Recall was prioritized over precision: for attrition, missing an "
            "employee who is about to leave is costlier than a false alarm."
        )

        st.subheader("Libraries")
        st.markdown(
            """
            - scikit-learn
            - pandas / numpy
            - matplotlib
            - shap
            - streamlit
            """
        )

    st.divider()

    st.markdown(
        """
        This app uses a trained Machine Learning model to predict employee attrition
        (whether an employee is likely to leave the company) based on HR data such as
        job role, income, satisfaction levels, and work experience. Predictions are
        paired with SHAP explanations so the reasoning behind each prediction is
        transparent rather than a black box.
        """
    )


# ==============================
# Footer
# ==============================

st.divider()


st.markdown(
    """
    <div style="text-align:center; color:gray; font-size:15px;">
        <b>Employee Attrition Prediction System</b><br>
        Developed by <b>Tuba Younas</b><br>
        BS Computer Science | Machine Learning Project<br>
        © 2026 All Rights Reserved
    </div>
    """,
    unsafe_allow_html=True
)

