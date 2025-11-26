-- ============================================
-- EchoMind - Database Initialization Script
-- ============================================
-- This script initializes the PostgreSQL database
-- with required tables, indexes, and seed data.

-- ===========================================
-- Create Extensions
-- ===========================================

-- UUID support
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Improved text search
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- ===========================================
-- Create Schema
-- ===========================================

-- Main schema for application tables
CREATE SCHEMA IF NOT EXISTS echomind;

-- Set search path
SET search_path TO echomind, public;

-- ===========================================
-- Users Table
-- ===========================================

CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    name VARCHAR(255) NOT NULL,
    household_size INTEGER DEFAULT 4,
    location VARCHAR(255),
    is_active BOOLEAN DEFAULT TRUE,
    is_verified BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP WITH TIME ZONE
);

-- Indexes for users
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_active ON users(is_active) WHERE is_active = TRUE;

-- ===========================================
-- Usage History Table
-- ===========================================

CREATE TABLE IF NOT EXISTS usage_history (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    resource_type VARCHAR(50) NOT NULL CHECK (resource_type IN ('electricity', 'water')),
    consumption DECIMAL(12, 4) NOT NULL,
    unit VARCHAR(20) NOT NULL,
    recorded_at TIMESTAMP WITH TIME ZONE NOT NULL,
    notes TEXT,
    source VARCHAR(50) DEFAULT 'manual',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for usage_history
CREATE INDEX IF NOT EXISTS idx_usage_user_id ON usage_history(user_id);
CREATE INDEX IF NOT EXISTS idx_usage_resource ON usage_history(resource_type);
CREATE INDEX IF NOT EXISTS idx_usage_recorded_at ON usage_history(recorded_at);
CREATE INDEX IF NOT EXISTS idx_usage_user_resource ON usage_history(user_id, resource_type);
CREATE INDEX IF NOT EXISTS idx_usage_user_date ON usage_history(user_id, recorded_at);

-- Partition by month for large datasets (optional, for production)
-- CREATE TABLE usage_history_y2024m01 PARTITION OF usage_history
--     FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

-- ===========================================
-- Preferences Table
-- ===========================================

CREATE TABLE IF NOT EXISTS preferences (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL UNIQUE REFERENCES users(id) ON DELETE CASCADE,
    electricity_rate DECIMAL(10, 6) DEFAULT 0.12,
    water_rate DECIMAL(10, 6) DEFAULT 0.002,
    currency VARCHAR(10) DEFAULT 'USD',
    notifications_enabled BOOLEAN DEFAULT TRUE,
    email_reports BOOLEAN DEFAULT FALSE,
    alert_threshold_electricity DECIMAL(10, 2) DEFAULT 50.0,
    alert_threshold_water DECIMAL(10, 2) DEFAULT 500.0,
    prediction_periods INTEGER DEFAULT 24,
    theme VARCHAR(20) DEFAULT 'light',
    language VARCHAR(10) DEFAULT 'en',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Index for preferences
CREATE INDEX IF NOT EXISTS idx_preferences_user ON preferences(user_id);

-- ===========================================
-- Predictions Table
-- ===========================================

CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    resource_type VARCHAR(50) NOT NULL,
    prediction_date TIMESTAMP WITH TIME ZONE NOT NULL,
    periods INTEGER NOT NULL,
    frequency VARCHAR(10) DEFAULT 'H',
    total_predicted DECIMAL(12, 4) NOT NULL,
    average_predicted DECIMAL(12, 4) NOT NULL,
    min_predicted DECIMAL(12, 4) NOT NULL,
    max_predicted DECIMAL(12, 4) NOT NULL,
    total_cost DECIMAL(12, 4) NOT NULL,
    model_version VARCHAR(50) DEFAULT '1.0.0',
    confidence_interval DECIMAL(4, 2) DEFAULT 0.95,
    predictions_json JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for predictions
CREATE INDEX IF NOT EXISTS idx_predictions_user ON predictions(user_id);
CREATE INDEX IF NOT EXISTS idx_predictions_resource ON predictions(resource_type);
CREATE INDEX IF NOT EXISTS idx_predictions_date ON predictions(prediction_date);
CREATE INDEX IF NOT EXISTS idx_predictions_created ON predictions(created_at);

-- ===========================================
-- Alerts Table
-- ===========================================

CREATE TABLE IF NOT EXISTS alerts (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    alert_type VARCHAR(50) NOT NULL,
    resource_type VARCHAR(50),
    priority VARCHAR(20) DEFAULT 'medium' CHECK (priority IN ('low', 'medium', 'high', 'critical')),
    title VARCHAR(255) NOT NULL,
    message TEXT NOT NULL,
    is_read BOOLEAN DEFAULT FALSE,
    is_dismissed BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    read_at TIMESTAMP WITH TIME ZONE
);

-- Indexes for alerts
CREATE INDEX IF NOT EXISTS idx_alerts_user ON alerts(user_id);
CREATE INDEX IF NOT EXISTS idx_alerts_unread ON alerts(user_id, is_read) WHERE is_read = FALSE;
CREATE INDEX IF NOT EXISTS idx_alerts_priority ON alerts(priority);
CREATE INDEX IF NOT EXISTS idx_alerts_created ON alerts(created_at);

-- ===========================================
-- Devices Table (for IoT integration)
-- ===========================================

CREATE TABLE IF NOT EXISTS devices (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    device_id VARCHAR(100) NOT NULL UNIQUE,
    device_type VARCHAR(50) NOT NULL CHECK (device_type IN ('electricity_meter', 'water_meter', 'smart_plug', 'sensor')),
    name VARCHAR(255),
    location VARCHAR(255),
    is_active BOOLEAN DEFAULT TRUE,
    last_seen TIMESTAMP WITH TIME ZONE,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for devices
CREATE INDEX IF NOT EXISTS idx_devices_user ON devices(user_id);
CREATE INDEX IF NOT EXISTS idx_devices_device_id ON devices(device_id);
CREATE INDEX IF NOT EXISTS idx_devices_type ON devices(device_type);
CREATE INDEX IF NOT EXISTS idx_devices_active ON devices(is_active) WHERE is_active = TRUE;

-- ===========================================
-- API Keys Table
-- ===========================================

CREATE TABLE IF NOT EXISTS api_keys (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    key_hash VARCHAR(255) NOT NULL,
    name VARCHAR(100),
    permissions JSONB DEFAULT '["read"]',
    is_active BOOLEAN DEFAULT TRUE,
    last_used TIMESTAMP WITH TIME ZONE,
    expires_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for api_keys
CREATE INDEX IF NOT EXISTS idx_api_keys_user ON api_keys(user_id);
CREATE INDEX IF NOT EXISTS idx_api_keys_active ON api_keys(is_active) WHERE is_active = TRUE;

-- ===========================================
-- Audit Log Table
-- ===========================================

CREATE TABLE IF NOT EXISTS audit_log (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
    action VARCHAR(100) NOT NULL,
    table_name VARCHAR(100),
    record_id INTEGER,
    old_values JSONB,
    new_values JSONB,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Index for audit_log
CREATE INDEX IF NOT EXISTS idx_audit_user ON audit_log(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_action ON audit_log(action);
CREATE INDEX IF NOT EXISTS idx_audit_created ON audit_log(created_at);

-- ===========================================
-- Functions and Triggers
-- ===========================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply updated_at trigger to relevant tables
DROP TRIGGER IF EXISTS update_users_updated_at ON users;
CREATE TRIGGER update_users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_preferences_updated_at ON preferences;
CREATE TRIGGER update_preferences_updated_at
    BEFORE UPDATE ON preferences
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_devices_updated_at ON devices;
CREATE TRIGGER update_devices_updated_at
    BEFORE UPDATE ON devices
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ===========================================
-- Views
-- ===========================================

-- Daily consumption summary view
CREATE OR REPLACE VIEW daily_consumption AS
SELECT 
    user_id,
    resource_type,
    DATE(recorded_at) as date,
    SUM(consumption) as total_consumption,
    AVG(consumption) as avg_consumption,
    MIN(consumption) as min_consumption,
    MAX(consumption) as max_consumption,
    COUNT(*) as reading_count
FROM usage_history
GROUP BY user_id, resource_type, DATE(recorded_at);

-- User statistics view
CREATE OR REPLACE VIEW user_statistics AS
SELECT 
    u.id as user_id,
    u.email,
    u.name,
    COUNT(DISTINCT uh.id) as total_readings,
    COUNT(DISTINCT CASE WHEN uh.resource_type = 'electricity' THEN uh.id END) as electricity_readings,
    COUNT(DISTINCT CASE WHEN uh.resource_type = 'water' THEN uh.id END) as water_readings,
    MAX(uh.recorded_at) as last_reading,
    COUNT(DISTINCT a.id) FILTER (WHERE a.is_read = FALSE) as unread_alerts
FROM users u
LEFT JOIN usage_history uh ON u.id = uh.user_id
LEFT JOIN alerts a ON u.id = a.user_id
GROUP BY u.id, u.email, u.name;

-- ===========================================
-- Seed Data
-- ===========================================

-- Insert demo user (password: demo123)
INSERT INTO users (email, password_hash, name, household_size, location, is_active, is_verified)
VALUES (
    'demo@echomind.io',
    'pbkdf2:sha256:260000$demo$c0e31e5a1b9b3f8d4e5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e',
    'Demo User',
    4,
    'San Francisco, CA',
    TRUE,
    TRUE
)
ON CONFLICT (email) DO NOTHING;

-- Insert preferences for demo user
INSERT INTO preferences (user_id, electricity_rate, water_rate, currency)
SELECT id, 0.12, 0.002, 'USD'
FROM users WHERE email = 'demo@echomind.io'
ON CONFLICT (user_id) DO NOTHING;

-- ===========================================
-- Permissions
-- ===========================================

-- Grant permissions to application user
-- (Adjust 'echomind' to your actual application user)
-- GRANT ALL PRIVILEGES ON SCHEMA echomind TO echomind;
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA echomind TO echomind;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA echomind TO echomind;
-- GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA echomind TO echomind;

-- ===========================================
-- Completion Message
-- ===========================================

DO $$
BEGIN
    RAISE NOTICE 'EchoMind database initialization complete!';
    RAISE NOTICE 'Tables created: users, usage_history, preferences, predictions, alerts, devices, api_keys, audit_log';
    RAISE NOTICE 'Demo user created: demo@echomind.io';
END $$;