# image of python
FROM python:3.10

# make working dir
WORKDIR /app

# copy the api code
COPY api.py .
COPY models/XGBoostRegressor.pkl models/XGBoostRegressor.pkl

# copy requirements.txt file and run it
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# exposing port 8000
EXPOSE 8080

# running the container
CMD ["uvicorn", "api:app","--host", "0.0.0.0","--port", "8080","--proxy-headers"]