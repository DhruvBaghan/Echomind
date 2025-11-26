# EchoMind - AI-Powered Resource Consumption Predictor

## Overview
EchoMind is an intelligent system that predicts electricity and water consumption based on historical usage data. The application uses pre-trained machine learning models (Prophet for time-series forecasting) to provide accurate predictions and cost estimations.

## Project Status
✅ **Ready to use** - The application has been successfully set up in Replit and is running.

## Recent Changes (November 26, 2025)
- ✅ Installed Python 3.11 and all required dependencies
- ✅ Configured environment variables for development
- ✅ Initialized SQLite database with demo data
- ✅ Set up Flask development server on port 5000
- ✅ Created WSGI entry point for production deployment
- ✅ Configured deployment with Gunicorn for autoscale
- ✅ Verified all API endpoints are working correctly

## Tech Stack
- **Backend**: Flask (Python 3.11)
- **ML Models**: Prophet (Facebook's time-series forecasting)
- **Frontend**: HTML, CSS, JavaScript, Chart.js
- **Database**: SQLite (Development) / PostgreSQL (Production)
- **Production Server**: Gunicorn

## Project Structure
```
├── backend/                 # Backend application code
│   ├── api/                # API route blueprints
│   ├── database/           # Database models and utilities
│   ├── ml_training/        # ML training scripts
│   ├── models/             # ML prediction models
│   ├── services/           # Business logic services
│   ├── utils/              # Utility functions
│   ├── app.py              # Flask application factory
│   └── config.py           # Configuration management
├── frontend/               # Frontend assets
│   ├── static/            # CSS, JS, images
│   └── templates/         # HTML templates
├── ml_models/             # Pre-trained ML models
│   ├── electricity_model.pkl
│   ├── water_model.pkl
│   └── model_metadata.json
├── scripts/               # Utility scripts
│   ├── setup_database.py  # Database initialization
│   ├── train_models.py    # Model training
│   └── download_datasets.py
├── wsgi.py                # WSGI entry point
└── requirements.txt       # Python dependencies
```

## Running the Application

### Development
The Flask development server is configured to run automatically via the "Flask Server" workflow.
- **URL**: Available via the Replit webview
- **Port**: 5000
- **Host**: 0.0.0.0

### Production Deployment
The application is configured for autoscale deployment using Gunicorn:
```bash
gunicorn --bind=0.0.0.0:5000 --reuse-port --workers=4 wsgi:app
```

## Environment Variables
All environment variables are configured in the Replit secrets/environment:
- `FLASK_APP`: backend.app
- `FLASK_ENV`: development
- `FLASK_DEBUG`: True
- `SECRET_KEY`: (configured)
- `DATABASE_URL`: sqlite:///echomind.db
- `HOST`: 0.0.0.0
- `PORT`: 5000
- Plus various configuration for costs, predictions, and models

## Database
- **Development**: SQLite (echomind.db)
- **Tables**: users, usage_history, preferences, predictions, alerts
- **Demo User**: demo@echomind.io / demo123
- **Demo Data**: 30 days of hourly electricity and water consumption data

## API Endpoints

### Health Check
- `GET /health` - System health status

### Frontend Routes
- `GET /` - Dashboard (default)
- `GET /dashboard` - Main dashboard
- `GET /input` - User input page
- `GET /predictions` - Predictions page
- `GET /electricity` - Electricity-specific page
- `GET /water` - Water-specific page

### API Routes (all under `/api`)
- `/api/dashboard/overview` - Dashboard overview data
- `/api/dashboard/predictions-summary` - Prediction summaries
- `/api/dashboard/alerts` - User alerts
- `/api/dashboard/sustainability` - Sustainability metrics
- `/api/predict/*` - Prediction endpoints
- `/api/electricity/*` - Electricity management
- `/api/water/*` - Water management
- `/api/user/*` - User management

## Features
- ⚡ **Electricity Prediction**: Predict future electricity consumption (kWh)
- 💧 **Water Prediction**: Predict future water consumption (liters)
- 📊 **Visual Analytics**: Interactive charts and graphs
- 💰 **Cost Estimation**: Calculate projected utility costs
- 🎯 **Pre-trained Models**: Ready-to-use ML models trained on real datasets
- 📱 **User-Friendly Interface**: Simple form-based input system

## Maintenance Scripts

### Database Management
```bash
# Initialize database
python scripts/setup_database.py --init --seed --yes

# Show database info
python scripts/setup_database.py --info

# Backup database
python scripts/setup_database.py --backup

# Reset database (caution!)
python scripts/setup_database.py --reset --yes
```

### Model Training
```bash
# Train models with data generation
python scripts/train_models.py --generate-data

# Generate datasets
python scripts/download_datasets.py --generate --process --days 90
```

## Known Issues
- **Plotly import warning**: Prophet (the ML library) shows "Importing plotly failed. Interactive plots will not work." This is a non-critical warning. Plotly is an optional dependency for Prophet's visualization features, which are not used in this application. All core prediction and API functionality works correctly.
- **Font Awesome icons**: Icons may not render properly due to CDN loading. This is a non-critical UI issue that does not affect functionality.

## Production Checklist (Before deploying to production)
- [ ] Set `FLASK_DEBUG=False`
- [ ] Generate strong `SECRET_KEY`
- [ ] Switch to PostgreSQL from SQLite
- [ ] Configure proper logging
- [ ] Set up monitoring
- [ ] Enable HTTPS/SSL
- [ ] Configure CORS for domain
- [ ] Set up automated backups
- [ ] Configure rate limiting

## Support
For questions or issues, refer to README.md or DEPLOYMENT.md
