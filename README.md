# MLOPSIFY

## Introduction
This project is designed to predict product review scores based on product descriptions. It follows MLOps best practices, leveraging MLflow for experiment tracking, FastAPI for building a prediction server, and Docker for containerizing the API. The deployment is handled using Google Cloud Run.

## Problem Statement
Customer reviews play a crucial role in influencing purchasing decisions. However, many products receive limited or no reviews, making it difficult to assess their quality. 
This project aims to bridge that gap by predicting product review scores based on product descriptions. By leveraging machine learning and MLOps best practices, this solution enables businesses to estimate customer sentiment even before reviews are posted. 

## Objective
The goal of this project is to develop a machine learning model that accurately predicts product review scores based on product descriptions. By implementing MLOps best practices, the project ensures a scalable, reproducible, and efficient pipeline for model development and deployment.

## Dataset
The dataset used in this project is [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce). The dataset has information of over 100k orders made at multiple. marketplaces in Brazil including attributes such as product_weight_g, product_length_cm, product_height_cm, product_width_cm.

## Standard Training Pipeline
The training pipeline consists of the following steps:

* `ingest_data` – Loads the dataset and creates a DataFrame.
* `clean_data` – Cleans the data by handling missing values and removing unnecessary columns.
* `train_model` – Trains the model and logs it using MLflow autologging.
* `evaluation` – Evaluates the model and logs performance metrics to the artifact store via MLflow autologging.

## Deployment
The API container was deployed to Google Cloud Run.Then by leveraging Docker, the FastAPI application was containerized. Google Cloud Run automatically scales based on demand, ensuring high availability while reducing infrastructure management overhead, making the prediction service reliable and cost-effective.
You can access the API prediction [here](https://mlops-project-api-v2-36719627723.us-central1.run.app/docs)

## Future Work
Implement GitLab Actions to automate the entire pipeline, from training and testing to deployment, ensuring a seamless CI/CD workflow.



