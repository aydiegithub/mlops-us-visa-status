# MLOps Project: U.S. Visa Status Prediction

This project is a complete MLOps implementation for a U.S. Visa Status Prediction system. It includes a full CI/CD pipeline for automated training, deployment, and monitoring of a machine learning model. The model predicts whether a visa application will be approved or denied based on various applicant attributes.

## Table of Contents
- [Project Overview](#project-overview)
- [Workflow Architecture](#workflow-architecture)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [ML Pipeline Stages](#ml-pipeline-stages)
  - [1. Data Ingestion](#1-data-ingestion)
  - [2. Data Validation](#2-data-validation)
  - [3. Data Transformation](#3-data-transformation)
  - [4. Model Trainer](#4-model-trainer)
  - [5. Model Evaluation](#5-model-evaluation)
  - [6. Model Pusher](#6-model-pusher)
- [CI/CD Pipeline](#cicd-pipeline)
- [How to Run](#how-to-run)
  - [Prerequisites](#prerequisites)
  - [Setup](#setup)
  - [Running the Training Pipeline](#running-the-training-pipeline)
  - [Running the Prediction Server](#running-the-prediction-server)
- [API Endpoints](#api-endpoints)
- [Connect with me](#connect-with-me)

## Project Overview

The primary goal of this project is to build an end-to-end machine learning system that can predict the outcome of U.S. visa applications. It demonstrates best practices in MLOps, including:
- **Modular Code Structure:** The project is organized into distinct components for each step of the ML lifecycle.
- **Automated Pipelines:** Both training and prediction are handled by automated pipelines.
- **CI/CD Integration:** Continuous Integration and Continuous Deployment are set up using GitHub Actions to automatically train and deploy the model on every push to the main branch.
- **Cloud Integration:** The system uses MongoDB for data storage and AWS S3/Cloudflare R2 for storing model artifacts.
- **Model Monitoring:** The pipeline includes a data validation step to detect data drift between training and new data.

## Workflow Architecture

The project is composed of two main pipelines: the **Training Pipeline** and the **Prediction Pipeline**. The CI/CD workflow automates the execution of the training pipeline and the deployment of the prediction service.

![Overall Workflow](http://googleusercontent.com/file_content/3)

## Technology Stack

- **Programming Language:** Python 3.8+
- **Web Framework:** Flask
- **ML Libraries:** Scikit-learn, Pandas, NumPy, CatBoost
- **Database:** MongoDB
- **Cloud Storage:** AWS S3, Cloudflare R2
- **CI/CD:** GitHub Actions
- **Containerization:** Docker
- **Data Validation:** Evidently AI

## Project Structure

The project follows a modular structure to ensure scalability and maintainability.

![Folder Structure](http://googleusercontent.com/file_content/4)

```
├── .github/workflows/         # CI/CD pipeline configuration
│   └── cicd.yml
├── config/                    # Configuration files
│   ├── model.yaml
│   └── schema.yaml
├── notebook/                  # Jupyter notebooks for experimentation
│   ├── EDA_US_Visa.ipynb
│   └── ...
├── static/                    # Static files for web UI
│   └── css/style.css
├── templates/                 # HTML templates for Flask app
│   └── index.html
├── usa_visa/                  # Core source code package
│   ├── __init__.py
│   ├── components/            # Modules for each pipeline stage
│   │   ├── __init__.py
│   │   ├── data_ingestion.py
│   │   ├── data_validation.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   │   ├── model_evaluation.py
│   │   └── model_pusher.py
│   ├── configuration/         # Configuration for external services
│   │   ├── __init__.py
│   │   ├── aws_connection.py
│   │   └── mongo_db_connection.py
│   ├── constants/             # Project-wide constants
│   │   └── __init__.py
│   ├── data_access/           # Data access layer for MongoDB
│   │   ├── __init__.py
│   │   └── usvisa_data.py
│   ├── entity/                # Entity definitions (data classes)
│   │   ├── __init__.py
│   │   ├── artifact_entity.py
│   │   └── config_entity.py
│   ├── exception/             # Custom exception handling
│   │   └── __init__.py
│   ├── logger/                # Custom logging
│   │   └── __init__.py
│   ├── pipeline/              # Pipeline orchestration
│   │   ├── __init__.py
│   │   ├── training_pipeline.py
│   │   └── prediction_pipeline.py
│   └── utils/                 # Utility functions
│       ├── __init__.py
│       └── main_utils.py
├── .gitignore
├── app.py                     # Main Flask application
├── demo.py                    # Script to run training pipeline
├── Dockerfile
├── LICENSE
├── requirements.txt
└── setup.py                   # Setup script for the package
```

---

## ML Pipeline Stages

The training pipeline is a sequence of components, each responsible for a specific task.

### 1. Data Ingestion
This stage is responsible for fetching the dataset from the source (MongoDB), splitting it into training and testing sets, and storing them as artifacts for the next stages.

![Data Ingestion Flow](http://googleusercontent.com/file_content/0)

### 2. Data Validation
This stage validates the ingested data to ensure its quality and integrity. It performs two key checks:
- **Schema Validation:** Verifies that the data conforms to a predefined schema (`schema.yaml`), checking for correct column names, data types, and value ranges.
- **Data Drift Detection:** Compares the statistical properties of the new training data with a baseline (or previous batch) to detect any significant changes (drift). This is crucial for maintaining model performance over time.

![Data Validation Flow](http://googleusercontent.com/file_content/2)

### 3. Data Transformation
This stage preprocesses the validated data to make it suitable for model training. It involves applying a series of transformations like one-hot encoding for categorical features and scaling for numerical features. The transformation pipeline object is saved to be used later during prediction.

![Data Transformation Flow](http://googleusercontent.com/file_content/1)

### 4. Model Trainer
This stage takes the transformed data and trains a machine learning model. It uses the CatBoost Classifier algorithm. After training, the model is saved as a `.pkl` file.

![Model Trainer Flow](http://googleusercontent.com/file_content/7)

### 5. Model Evaluation
This stage evaluates the newly trained model's performance against the model currently in production. If no production model exists, the new model is accepted by default. If a production model exists, the new model is accepted only if its performance (e.g., F1-score) is better by a predefined margin.

![Model Evaluation Flow](http://googleusercontent.com/file_content/5)

### 6. Model Pusher
If the model evaluation stage accepts the new model, this stage pushes the model artifact to a production-ready cloud storage location (AWS S3 or Cloudflare R2). This makes the model available for the prediction service.

![Model Pusher Flow](http://googleusercontent.com/file_content/6)

---

## CI/CD Pipeline

The project is configured with a CI/CD pipeline using **GitHub Actions** (`.github/workflows/cicd.yml`). This pipeline automates the entire process from code commit to deployment.

**Workflow:**
1.  **Trigger:** The workflow is triggered on every `push` to the `main` branch.
2.  **Setup:** It sets up a virtual environment with the required Python version.
3.  **Install Dependencies:** It installs all the project dependencies from `requirements.txt`.
4.  **Run Training Pipeline:** It executes the entire ML training pipeline by running the `demo.py` script. This ensures the code is working and produces a new model if necessary.
5.  **Build Docker Image:** It builds a Docker image of the Flask application.
6.  **Push to Docker Hub:** It pushes the newly built image to Docker Hub, making it ready for deployment.

This automated process ensures that every change is tested and a new, potentially better model is deployed seamlessly.

## How to Run

### Prerequisites
- Python 3.8 or higher
- MongoDB account and connection URI
- AWS account with S3 bucket (or Cloudflare account with R2 bucket)
- Docker installed locally (for building the image)

### Setup
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/mlops-us-visa-status.git](https://github.com/your-username/mlops-us-visa-status.git)
    cd mlops-us-visa-status
    ```

2.  **Create a virtual environment and activate it:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up Environment Variables:**
    Create a `.env` file in the root directory and add your credentials. The application will automatically load them.
    ```
    MONGO_DB_URL="your_mongodb_uri"
    AWS_ACCESS_KEY_ID="your_aws_access_key"
    AWS_SECRET_ACCESS_KEY="your_aws_secret_key"
    # Or for Cloudflare R2
    CLOUDFLARE_ENDPOINT_URL="your_r2_endpoint"
    CLOUDFLARE_ACCESS_KEY_ID="your_r2_access_key"
    CLOUDFLARE_SECRET_ACCESS_KEY="your_r2_secret_key"
    ```

### Running the Training Pipeline
To run the training pipeline manually, execute the `demo.py` script:
```bash
python demo.py
```
This will run all the stages from data ingestion to model pusher and save the artifacts in the `artifact/` directory.

### Running the Prediction Server
To start the Flask server for predictions, run the `app.py` script:
```bash
python app.py
```
The server will start on `http://127.0.0.1:8080`.

## API Endpoints

The Flask application provides the following endpoints:

- **`GET /`**: Renders a simple web UI to input data and get a prediction.
- **`POST /predict`**: The main prediction endpoint. It accepts JSON data with the applicant's information and returns the visa status prediction.
- **`GET /train`**: Triggers a new training pipeline run.

**Example `curl` request for prediction:**
```bash
curl -X POST [http://127.0.0.1:8080/predict](http://127.0.0.1:8080/predict) \
-H "Content-Type: application/json" \
-d '{
      "continent": "Asia",
      "education_of_employee": "Master''s",
      "has_job_experience": "Y",
      "requires_job_training": "N",
      "no_of_employees": 100,
      "company_age": 20,
      "region_of_employment": "West",
      "prevailing_wage": 120000,
      "unit_of_wage": "Year",
      "full_time_position": "Y",
      "case_status": "Certified"
    }'
```

---

## 🔗 Connect With Me:

- 🌐 **Website:** [aydie.in](https://aydie.in)
- 👨‍💻 **Coding Profiles:**  
  - [LeetCode](https://leetcode.com/aydie)  
  - [HackerRank](https://hackerrank.com/aydie)
- 📸 **Instagram:** [@aydiemusic](https://instagram.com/aydiemusic)
- 💼 **LinkedIn:** [linkedin.com/in/aydiemusic](https://www.linkedin.com/in/aydiemusic)
- 🎵 **Music Channel:** [YouTube - Aydie Music](https://www.youtube.com/aydiemuisc)
