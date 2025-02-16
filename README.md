# Opsify  

## Introduction

This project leverages machine learning to predict customer reviews based on input features. 
It follows MLOps best practices, including building a FastAPI server for predictions, containerizing the application with Docker, and deploying it to the cloud for scalability and accessibility.

## Problem Statement

Customer reviews play a crucial role in shaping a product's reputation and influencing purchasing decisions. 
However, predicting customer satisfaction based on product attributes can help businesses identify potential issues early and improve the overall customer experience. 
This project aims to predict the **Review Score** of a customer using machine learning techniques. 
Given a dataset containing customer order details and product attributes that impact customer ratings, aim is to accurately predict the review scores through which businesses can gain valuable insights into product performance and enhance customer satisfaction strategies.

## Aim and Objective

This project aims to predict customer review scores using machine learning while following MLOps best practices for deployment and scalability. A FastAPI server is built for real-time predictions, containerized with Docker, and deployed on the cloud for accessibility.

## Dataset Description

The dataset used in this project is the [**Brazilian E-Commerce Public Dataset by Olist**](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce/data?select=olist_customers_dataset.csv), sourced from **Kaggle**. It contains **100,000+ orders** made between **2016 and 2018** across multiple online marketplaces in Brazil. The dataset provides a comprehensive view of e-commerce transactions, including **order details, product attributes, customer information, payment data, delivery performance, and customer reviews**. The review scores, given by customers after receiving their orders, serve as the target variable for this machine learning model.

## Solution

This project leverages **machine learning** to predict customer review scores based on product and order attributes. A **FastAPI server** is built to serve real-time predictions, which is **containerized using Docker** and **deployed on the cloud** for scalability. By following **MLOps best practices**, the solution ensures efficient model deployment, automation, and monitoring, making it robust and production-ready.

## Training Pipeline
The standard training pipeline consists of several steps:

* `ingest_data`: This step will ingest the data and create a DataFrame.
* `clean_data`: This step will clean the data and remove the unwanted columns.
* `train_model`: This step will train the model and save the model using MLflow autologging.
* `evaluation`: This step will evaluate the model and save the metrics -- using MLflow autologging -- into the artifact store.

