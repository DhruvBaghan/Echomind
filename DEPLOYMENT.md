# EchoMind Deployment Guide

This guide covers deploying EchoMind to various cloud platforms.

## Table of Contents
- [Local Development](#local-development)
- [Heroku Deployment](#heroku-deployment)
- [AWS Deployment](#aws-deployment)
- [Google Cloud Deployment](#google-cloud-deployment)
- [Docker Deployment](#docker-deployment)

---

## Local Development

### Prerequisites
- Python 3.8+
- Git

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/echomind.git
cd echomind/EchoMind

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup environment
cp .env.example .env
# Edit .env with your configuration

# Initialize database
python scripts/setup_database.py --init --seed --yes

# Generate/process datasets
python scripts/download_datasets.py --process

# Train models
python scripts/train_models.py

# Run application
python -c "import sys; sys.path.insert(0, '..'); from EchoMind.backend.app import create_app; app = create_app(); app.run(debug=True, port=5000)"
```

Access: http://127.0.0.1:5000

---

## Heroku Deployment

### Prerequisites
- Heroku account
- Heroku CLI installed

### Steps

1. **Create Procfile:**
```
web: gunicorn -w 4 -b 0.0.0.0:$PORT "EchoMind.backend.app:create_app()"
```

2. **Update requirements.txt:**
```bash
pip freeze > requirements.txt
# Add gunicorn
echo "gunicorn" >> requirements.txt
```

3. **Initialize Heroku app:**
```bash
heroku login
heroku create your-echomind-app
```

4. **Configure environment variables:**
```bash
heroku config:set FLASK_ENV=production
heroku config:set SECRET_KEY=your-production-secret-key
heroku config:set DATABASE_URL=postgresql://...
```

5. **Deploy:**
```bash
git push heroku main
```

6. **Initialize database on Heroku:**
```bash
heroku run python EchoMind/scripts/setup_database.py --init --seed --yes
```

7. **Train models on Heroku:**
```bash
heroku run python EchoMind/scripts/train_models.py
```

**Access:** https://your-echomind-app.herokuapp.com

---

## AWS Deployment

### Option A: AWS Elastic Beanstalk (Recommended for Flask)

1. **Install EB CLI:**
```bash
pip install awsebcli
```

2. **Initialize Elastic Beanstalk:**
```bash
eb init -p python-3.11 echomind --region us-east-1
```

3. **Create Procfile:**
```
web: gunicorn --chdir EchoMind -w 4 -b 0.0.0.0:5000 backend.app:create_app()
```

4. **Create environment:**
```bash
eb create echomind-env
```

5. **Deploy:**
```bash
eb deploy
```

6. **Open app:**
```bash
eb open
```

### Option B: AWS EC2 with Nginx + Gunicorn

1. **Launch EC2 instance** (Ubuntu 22.04 LTS, t2.micro for free tier)

2. **SSH into instance:**
```bash
ssh -i your-key.pem ubuntu@your-instance-ip
```

3. **Install dependencies:**
```bash
sudo apt update
sudo apt install python3-pip python3-venv nginx git -y
```

4. **Clone and setup:**
```bash
git clone https://github.com/yourusername/echomind.git
cd echomind/EchoMind
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

5. **Setup Gunicorn:**
```bash
pip install gunicorn
gunicorn -w 4 -b 127.0.0.1:8000 "EchoMind.backend.app:create_app()" --chdir ..
```

6. **Configure Nginx:**
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

---

## Google Cloud Deployment

### Option A: Google Cloud Run (Serverless, Recommended)

1. **Install Google Cloud SDK**

2. **Create Dockerfile:**
Already provided in the project. Ensure it references correct paths.

3. **Build and deploy:**
```bash
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/echomind
gcloud run deploy echomind --image gcr.io/YOUR_PROJECT_ID/echomind --platform managed --region us-central1 --allow-unauthenticated
```

### Option B: Google App Engine

1. **Create app.yaml:**
```yaml
runtime: python311

env: standard
entrypoint: gunicorn -w 4 -b 0.0.0.0:8080 "EchoMind.backend.app:create_app()"

env_variables:
  FLASK_ENV: "production"
```

2. **Deploy:**
```bash
gcloud app deploy
```

---

## Docker Deployment

### Local Docker

1. **Build image:**
```bash
docker build -t echomind:latest .
```

2. **Run container:**
```bash
docker run -p 5000:5000 \
  -e FLASK_ENV=production \
  -e SECRET_KEY=your-key \
  echomind:latest
```

### Docker Compose

```bash
docker-compose up -d
```

### Push to Container Registry

```bash
# Google Container Registry
docker tag echomind gcr.io/YOUR_PROJECT_ID/echomind
docker push gcr.io/YOUR_PROJECT_ID/echomind

# Docker Hub
docker tag echomind yourusername/echomind
docker push yourusername/echomind
```

---

## Production Checklist

- [ ] Set `FLASK_DEBUG=False`
- [ ] Use strong `SECRET_KEY` (generate: `python -c "import secrets; print(secrets.token_hex(32))"`)
- [ ] Use PostgreSQL instead of SQLite
- [ ] Configure proper logging
- [ ] Set up monitoring (e.g., Sentry for errors)
- [ ] Enable HTTPS/SSL
- [ ] Configure CORS for your domain
- [ ] Set up automated backups
- [ ] Configure rate limiting
- [ ] Add environment-specific configurations
- [ ] Document API endpoints
- [ ] Set up CI/CD pipeline (GitHub Actions, GitLab CI, etc.)

---

## Troubleshooting

### Models not loading
- Ensure `ml_models/` directory exists
- Run training script before deployment
- Check file permissions

### Database errors
- Verify DATABASE_URL is correct
- Run migrations: `python EchoMind/scripts/setup_database.py --init`
- Check disk space on server

### Performance issues
- Increase workers: `gunicorn -w 8`
- Enable caching in Flask
- Use PostgreSQL with proper indexing
- Implement Redis for sessions

### Out of memory
- Reduce number of Gunicorn workers
- Optimize model loading
- Implement pagination for large datasets

---

## Scaling Recommendations

1. **Database**: Use managed PostgreSQL (AWS RDS, Google Cloud SQL)
2. **Cache**: Add Redis for session/data caching
3. **Storage**: Use S3/GCS for model files and datasets
4. **Queue**: Use Celery + RabbitMQ for background tasks (model training)
5. **CDN**: Use CloudFront/Cloud CDN for static files
6. **Monitoring**: Set up CloudWatch/Stackdriver

---

For questions or issues, refer to the [README.md](README.md) or create an issue on GitHub.
