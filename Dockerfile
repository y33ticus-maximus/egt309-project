
# docker build -t gas-monitoring-pipeline .
# docker run --rm gas-monitoring-pipeline

FROM python:3.11-slim

WORKDIR /app

# install the Python libraries
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy the project files into the image
COPY . .

# run all model pipelines when the container starts
CMD ["bash", "run.sh"]
