# Customer Segmentation ML Project

**A comprehensive machine learning solution for customer segmentation using clustering algorithms with FastAPI web interface and automated CI/CD deployment.**

## Overview

This project implements a **customer segmentation system** using advanced machine learning clustering algorithms. It analyzes customer behavior patterns to automatically classify customers into distinct segments, helping businesses make data-driven decisions for targeted marketing and personalized customer experiences.

## User Interface
![Customer Segmentation Web Interface](ui.png)

### Key Capabilities:
- **6 Customer Segments**: Budget Conscious, High Value, Regular Customers, Premium Shoppers, Occasional Buyers, Loyal Customers
- **Multiple Clustering Models**: KMeans, AgglomerativeClustering, DBSCAN with automatic model selection
- **Real-time Predictions**: FastAPI-powered web interface and REST API
- **Automated Deployment**: CI/CD pipeline with Docker containerization
- **Scalable Architecture**: Cloud-native design with AWS integration

## Features

### Machine Learning
- **Advanced Clustering**: Multiple algorithms with hyperparameter optimization
- **Model Evaluation**: Silhouette score, Calinski-Harabasz score, Davies-Bouldin score
- **Feature Engineering**: Automated data preprocessing and transformation
- **Model Persistence**: Automated model saving and loading with versioning

### Web Application
- **Interactive UI**: Beautiful, responsive web interface for predictions
- **REST API**: RESTful endpoints for programmatic access
- **Real-time Results**: Instant customer segmentation predictions
- **Batch Processing**: Support for multiple customer predictions

### Cloud Integration
- **AWS Deployment**: Automated deployment to AWS EC2
- **Docker Containerization**: Consistent deployment across environments
- **MongoDB Atlas**: Cloud database for customer data storage
- **ECR Registry**: Container image management

### DevOps
- **CI/CD Pipeline**: Automated testing, building, and deployment
- **GitHub Actions**: Continuous integration and deployment
- **Environment Management**: Separate configurations for dev/prod
- **Monitoring**: Comprehensive logging and health checks

## Architecture

```mermaid
graph TB
    A[Customer Data] --> B[Data Ingestion]
    B --> C[Data Validation]
    C --> D[Data Transformation]
    D --> E[Model Training]
    E --> F[Model Evaluation]
    F --> G[Model Deployment]
    G --> H[FastAPI Web Service]
    H --> I[Customer Predictions]
    
    J[MongoDB Atlas] --> B
    K[AWS S3] --> G
    L[Docker Container] --> H
    M[GitHub Actions] --> N[AWS ECR]
    N --> O[EC2 Deployment]
```

## Project Structure

```
customer_segmentation/
├── src/                          # Source code
│   ├── components/               # ML pipeline components
│   │   ├── data_ingestion.py        # Data loading and preprocessing
│   │   ├── data_transformation.py   # Feature engineering
│   │   ├── data_validation.py       # Data quality checks
│   │   ├── model_trainer.py         # Model training logic
│   │   ├── model_evaluation.py      # Model assessment
│   │   └── model_pusher.py          # Model deployment
│   ├── pipline/                  # Pipeline orchestration
│   │   ├── training_pipeline.py     # End-to-end training
│   │   └── prediction_pipeline.py   # Real-time predictions
│   ├── entity/                   # Data structures
│   │   ├── config_entity.py         # Configuration classes
│   │   ├── artifact_entity.py       # Pipeline artifacts
│   │   ├── estimator.py             # Model wrapper
│   │   └── s3_estimator.py          # Cloud model handler
│   ├── configuration/            # External connections
│   │   ├── mongo_db_connection.py   # Database connectivity
│   │   └── aws_connection.py        # AWS services
│   ├── utils/                    # Utility functions
│   ├── logger/                   # Logging configuration
│   └── exception/                # Error handling
├── templates/                    # Web interface
│   └── index.html                   # Main UI template
├── notebook/                     # Jupyter notebooks
│   ├── analysis.ipynb               # Exploratory data analysis
│   └── segmentation.py             # Analysis scripts
├── onfig/                       # Configuration files
│   ├── model.yaml                   # ML model parameters
│   └── schema.yaml                  # Data schema definition
├── Dockerfile                    # Container configuration
├── app.py                        # FastAPI application
├── demo.py                       # Pipeline demonstration
├── requirements.txt              # Python dependencies
└── .github/workflows/           # CI/CD pipeline
    └── cicd.yaml                    # GitHub Actions workflow
```

## Installation

### Prerequisites
- Python 3.10+
- Docker
- AWS CLI (for deployment)
- Git

### Local Development Setup

```bash
# Clone the repository
git clone https://github.com/beniamine3155/customer_segmentation.git
cd customer_segmentation

# Create virtual environment
python -m venv segvenv
source segvenv/bin/activate  # On Windows: segvenv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export MONGODB_URL="your_mongodb_connection_string"
export AWS_ACCESS_KEY_ID="your_aws_access_key"
export AWS_SECRET_ACCESS_KEY="your_aws_secret_key"
export AWS_DEFAULT_REGION="us-east-1"
```

### Docker Setup

```bash
# Build Docker image
docker build -t customer-segmentation .

# Run container
docker run -d \
  -e MONGODB_URL="your_mongodb_connection_string" \
  -e AWS_ACCESS_KEY_ID="your_aws_access_key" \
  -e AWS_SECRET_ACCESS_KEY="your_aws_secret_key" \
  -e AWS_DEFAULT_REGION="us-east-1" \
  -p 5001:5001 \
  customer-segmentation
```

## Usage

### Training the Model

```bash
# Run the complete ML pipeline
python demo.py
```

This will execute:
1. **Data Ingestion**: Load customer data from MongoDB
2. **Data Validation**: Verify data quality and schema
3. **Data Transformation**: Feature engineering and preprocessing
4. **Model Training**: Train multiple clustering models
5. **Model Evaluation**: Select best performing model
6. **Model Deployment**: Save model artifacts

### Web Application

```bash
# Start FastAPI server
python app.py

# Access web interface
# Open browser: http://localhost:5001
```

### Prediction Pipeline

```python
from src.pipline.prediction_pipeline import predict_customer_segment

# Single customer prediction
result = predict_customer_segment(
    age=35,
    income=50000,
    total_spend=1200,
    recency=30,
    num_web_purchases=5,
    num_store_purchases=3,
    num_web_visits_month=4
)

print(f"Customer Segment: {result['segment_details']['Segment_Name']}")
```

## API Documentation

### FastAPI Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Web interface |
| `POST` | `/predict` | JSON API prediction |
| `POST` | `/predict-form` | HTML form submission |
| `POST` | `/predict-batch` | Batch predictions |
| `GET` | `/health` | Health check |
| `GET` | `/segments` | Segment information |
| `GET` | `/docs` | Interactive API docs |

### Example API Usage

```bash
# Single prediction
curl -X POST "http://localhost:5001/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 35,
    "income": 50000,
    "total_spend": 1200,
    "recency": 30,
    "num_web_purchases": 5,
    "num_store_purchases": 3,
    "num_web_visits_month": 4
  }'

# Health check
curl -X GET "http://localhost:5001/health"
```

## Machine Learning Pipeline

### Data Features
The model uses 7 key features for customer segmentation:

| Feature | Description | Type |
|---------|-------------|------|
| `Age` | Customer age (2025 - birth_year) | Numeric |
| `Income` | Annual income in dollars | Numeric |
| `Total_Spend` | Total annual spending | Numeric |
| `Recency` | Days since last purchase | Numeric |
| `NumWebPurchases` | Number of web purchases | Numeric |
| `NumStorePurchases` | Number of store purchases | Numeric |
| `NumWebVisitsMonth` | Monthly website visits | Numeric |

### Clustering Models
- **KMeans**: Centroid-based clustering
- **AgglomerativeClustering**: Hierarchical clustering
- **DBSCAN**: Density-based clustering

### Model Selection
Models are evaluated using:
- **Silhouette Score**: Measure cluster cohesion and separation
- **Calinski-Harabasz Score**: Ratio of between-cluster to within-cluster dispersion
- **Davies-Bouldin Score**: Average similarity between clusters

### Customer Segments

| Segment | ID | Description |
|---------|----| ------------|
| Budget Conscious | 0 | Price-sensitive customers focusing on value |
| High Value | 1 | High income with significant spending power |
| Regular Customers | 2 | Consistent moderate spending patterns |
| Premium Shoppers | 3 | Quality-focused, premium product buyers |
| Occasional Buyers | 4 | Infrequent but meaningful purchases |
| Loyal Customers | 5 | Highly engaged with frequent interactions |

## Deployment

### GitHub Actions CI/CD

The project includes automated deployment pipeline:

1. **Continuous Integration**:
   - Code checkout
   - AWS credentials configuration
   - Docker image build
   - Push to Amazon ECR

2. **Continuous Deployment**:
   - Pull image from ECR
   - Deploy to EC2 instance
   - Health checks

### Required GitHub Secrets

```yaml
# AWS Configuration
AWS_ACCESS_KEY_ID: your_aws_access_key
AWS_SECRET_ACCESS_KEY: your_aws_secret_key
AWS_DEFAULT_REGION: us-east-1

# ECR Repository
ECR_REPO: customer-seg-proj

# Database
MONGODB_URL: mongodb+srv://user:pass@cluster.mongodb.net/db
```

### Manual Deployment

```bash
# Build and push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 939966403904.dkr.ecr.us-east-1.amazonaws.com
docker build -t customer-seg-proj .
docker tag customer-seg-proj:latest 939966403904.dkr.ecr.us-east-1.amazonaws.com/customer-seg-proj:latest
docker push 939966403904.dkr.ecr.us-east-1.amazonaws.com/customer-seg-proj:latest

# Deploy to EC2
docker run -d \
  -e MONGODB_URL="your_mongodb_url" \
  -e AWS_ACCESS_KEY_ID="your_access_key" \
  -e AWS_SECRET_ACCESS_KEY="your_secret_key" \
  -e AWS_DEFAULT_REGION="us-east-1" \
  -p 5001:5001 \
  939966403904.dkr.ecr.us-east-1.amazonaws.com/customer-seg-proj:latest
```

## Configuration

### Environment Variables

```bash
# Database
MONGODB_URL=mongodb+srv://user:pass@cluster.mongodb.net/db

# AWS Services
AWS_ACCESS_KEY_ID=your_access_key_id
AWS_SECRET_ACCESS_KEY=your_secret_access_key
AWS_DEFAULT_REGION=us-east-1

# Model Configuration
MODEL_BUCKET_NAME=customer-seg-55
MODEL_FILE_NAME=model.pkl
```

### Model Configuration (`config/model.yaml`)

```yaml
kmeans:
  n_clusters: [2, 3, 4, 5, 6]
  random_state: 42

agglomerative:
  n_clusters: [2, 3, 4, 5, 6]
  linkage: ['ward', 'complete', 'average']

dbscan:
  eps: [0.1, 0.3, 0.5, 0.7, 1.0]
  min_samples: [3, 5, 8, 10]
```