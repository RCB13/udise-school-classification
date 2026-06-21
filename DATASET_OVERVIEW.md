# UDISE+ School Classification Dataset Overview & Data Dictionary

This document provides a detailed overview of the dataset used in this machine learning project. Anyone viewing your GitHub repository will be able to understand the source data, the filtering criteria, the features, and how the target classification label was constructed.

---

## 1. About the Dataset Source
The raw data is derived from the **Unified District Information System for Education Plus (UDISE+)**, under the Ministry of Education, Government of India. It is one of the largest education databases in the world, covering details of over **1.48 million schools**, 9.5 million teachers, and 265 million students.

This project uses a cleaned and merged subset of the dataset:
*   **Raw Rows**: 1,480,000+
*   **Target Scope**: Government-managed schools with standard grade cycles.
*   **Final Filtered Rows**: ~700,000+ schools.

---

## 2. Data Scope & Filtering Criteria
Before model training, the dataset was filtered according to Samagra Shiksha policy guidelines to ensure we compare like-for-like government schools:

1.  **Government Management Filter**: Only schools managed by government departments are included (e.g. Dept of Education, Local Body, Tribal Welfare, Central Govt). Private, un-aided, and private-aided schools are excluded.
2.  **Standard Grade Cycle Filter**: Only standard grade structures are retained:
    *   Primary: `(1, 5)` or `(1, 8)`
    *   Upper Primary: `(6, 8)`
    *   Secondary/Higher Secondary: `(1, 10)`, `(1, 12)`, `(6, 10)`, `(6, 12)`, `(9, 10)`, `(9, 12)`, `(11, 12)`.
    Schools with non-standard, fragmented cycles are filtered out.

---

## 3. Data Dictionary (Feature Schema)

The features used by the machine learning model fall into three main categories:

### 3.1 School Demographics & Management
*   `state` (Categorical): The state or Union Territory name where the school is located.
*   `district` (Categorical): The local administrative district.
*   `pincode` (Numeric): Geographic ZIP/postal code (missing values Mode-imputed by state-district grouping).
*   `managment` (Categorical/Int): Code indicating the specific government body managing the school (e.g., Local Body, Social Welfare).
*   `school_category` (Int): The level of the school (1 = Primary, 2 = Upper Primary, 3 = Secondary/Higher).

### 3.2 Teacher & Student Capacity Metrics
*   `total_student_count` (Int): Total enrollment of students.
*   `total_teacher` (Int): Total number of teachers on staff.
*   `regular` (Int): Count of regular (permanent) teachers.
*   `graduate` (Int): Count of teachers with a Bachelor's degree.
*   `post_graduate_and_above` (Int): Count of teachers with Master's/Ph.D. degrees.
*   `total_teacher_trained_computer` (Int): Count of teachers trained in computer operations.
*   `teacher_received_service_training` (Int): Count of teachers who received service training.
*   `med_equivalent` (Int): Count of teachers with professional education degrees (M.Ed. equivalent).

### 3.3 School Infrastructure Indicators
*   `classrooms_in_good_condition` (Int): Classrooms requiring no structural repairs.
*   `classrooms_needs_minor_repair` (Int): Classrooms needing minor structural fixes.
*   `classrooms_needs_major_repair` (Int): Classrooms needing major structural restoration.
*   `drinking_water_available` (Int Code): 1 = Yes, 2 = No.
*   `drinking_water_functional` (Int Code): 1 = Yes, 2 = No.
*   `rain_water_harvesting` (Int Code): 0 = No, 1 = Yes, 2 = Not Functional.
*   `handwash_facility_for_meal` (Int Code): 0 = No, 1 = Yes, 2 = Not Functional.
*   `electricity_availability` (Int Code): 1 = Yes, 2 = No.
*   `solar_panel` (Int Code): 0 = No, 1 = Yes, 2 = Not Functional.
*   `library_availability` (Int Code): 1 = Yes, 2 = No.
*   `playground_available` (Int Code): 1 = Yes, 2 = No.
*   `medical_checkups` (Int Code): 0 = No, 1 = Yes, 2 = Not Functional.

---

## 4. Rule-Based Target Label Construction
Because UDISE+ does not contain a pre-defined "vulnerability" target column, a target variable `school_label` was constructed using **Government-Tuned Quality Indicators**:

### 4.1 Teacher Effectiveness Score (0–100)
Tuned to align with the Right to Education (RTE) Act:
*   **70% Weight**: Student-Teacher Ratio (STR) deviation. A primary school has an ideal ratio of 40:1, while secondary has 45:1. Deviation from this ideal reduces this component.
*   **30% Weight**: Teacher Qualification index (percentage of regular teachers, computer-literate staff, and professional degrees).

### 4.2 Infrastructure Score (0–100)
A weighted index of the 12 core infrastructure indicators (listed in section 3.3). Cleanliness, drinking water, electricity, and functional toilets are weighted highest (5% each) in accordance with Samagra Shiksha policies.

### 4.3 Final Classification Rule
To separate normal functioning government schools from severely struggling ones under resource constraints:
*   `Standardized (0)`: Meet the baseline parameters (Teacher Effectiveness Score $\geq 30$ **AND** Infrastructure Score $\geq 15$).
*   `Odd (1)`: Classified as **Odd** if Teacher Effectiveness Score $< 30$ **OR** Infrastructure Score $< 15$.

**Target Class Distribution**:
*   `0` (Standardized): ~76% of government schools.
*   `1` (Odd): ~24% of government schools.

---

## 5. Machine Learning Objective
The model's objective is to **predict `school_label` (0 or 1) using the 20 raw features** (excluding the calculated composite scores to prevent target leakage). This trains the classifier to learn the complex non-linear combinations of raw school characteristics that represent a vulnerable school, serving as a diagnostic tool for policymakers.
