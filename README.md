# UDISE+ School Quality Diagnostics & ML Pipeline

A machine learning classification pipeline and Streamlit dashboard built on the UDISE+ dataset to identify under-resourced ("Odd") government schools for targeted policy interventions.

## Project Structure

*   `udise_school_classification.ipynb`: Structured Jupyter Notebook containing the full training pipeline (Weeks 1-9).
*   `udise_school_classification.py`: Standard Python script equivalent of the training pipeline.
*   `app.py`: Sleek, modern Streamlit dashboard for single-school diagnostics and batch prediction.
*   `requirements.txt`: List of dependencies required to run the project.
*   `sample_test_schools.csv`: Reference dataset to test the batch upload prediction tool in the web app.

## Getting Started

### 1. Prerequisites
Ensure you have Python 3.8+ installed. Install the dependencies:
```bash
pip install -r requirements.txt
```

### 2. Train the Model (Optional)
To regenerate the model serialization pipeline:
1. Place the raw dataset file `final_merged_data.csv` in this directory.
2. Run the training script:
```bash
python udise_school_classification.py
```
This generates `school_classifier_pipeline.joblib`, which contains the final Voting Classifier model along with all the fitted preprocessing transformers (Scalers, PCA projections, and Label Encoders).

### 3. Launch the Streamlit App
Run the web application locally:
```bash
streamlit run app.py
```
This will start a local server and automatically open the dashboard in your web browser (usually at `http://localhost:8501`).

## Streamlit Cloud Deployment

To host this dashboard on the web:
1. Create a repository on **GitHub** and push all project files (including `app.py`, `requirements.txt`, and the generated `school_classifier_pipeline.joblib`).
2. Go to **[streamlit.io/cloud](https://streamlit.io/cloud)** and sign in with your GitHub account.
3. Click **New app**, select your repository, branch, and specify the main file path as `app.py`.
4. Click **Deploy!** Your app will be live and shareable.
