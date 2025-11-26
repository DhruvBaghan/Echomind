/* ============================================
   EchoMind - API Client
   ============================================ */

/**
 * API Client for EchoMind backend
 * Handles all HTTP requests to the API
 */

const API = {
    // Base URL for API endpoints
    baseUrl: '/api',
    
    // Default headers for requests
    defaultHeaders: {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    },
    
    /**
     * Make a generic API request
     * @param {string} endpoint - API endpoint
     * @param {object} options - Fetch options
     * @returns {Promise} Response data
     */
    async request(endpoint, options = {}) {
        const url = `${this.baseUrl}${endpoint}`;
        
        const config = {
            headers: { ...this.defaultHeaders, ...options.headers },
            ...options
        };
        
        try {
            const response = await fetch(url, config);
            const data = await response.json();
            
            if (!response.ok) {
                throw new APIError(
                    data.message || data.error || 'Request failed',
                    response.status,
                    data
                );
            }
            
            return data;
        } catch (error) {
            if (error instanceof APIError) {
                throw error;
            }
            throw new APIError('Network error', 0, { originalError: error.message });
        }
    },
    
    /**
     * GET request
     */
    async get(endpoint, params = {}) {
        const queryString = new URLSearchParams(params).toString();
        const url = queryString ? `${endpoint}?${queryString}` : endpoint;
        return this.request(url, { method: 'GET' });
    },
    
    /**
     * POST request
     */
    async post(endpoint, data = {}) {
        return this.request(endpoint, {
            method: 'POST',
            body: JSON.stringify(data)
        });
    },
    
    /**
     * PUT request
     */
    async put(endpoint, data = {}) {
        return this.request(endpoint, {
            method: 'PUT',
            body: JSON.stringify(data)
        });
    },
    
    /**
     * DELETE request
     */
    async delete(endpoint) {
        return this.request(endpoint, { method: 'DELETE' });
    },
    
    // ----- Prediction Endpoints -----
    
    predictions: {
        /**
         * Get electricity predictions
         * @param {array} data - Consumption data array
         * @param {number} periods - Number of periods to predict
         * @param {string} frequency - Prediction frequency (H, D, W)
         */
        async electricity(data, periods = 24, frequency = 'H') {
            return API.post('/predict/electricity', { data, periods, frequency });
        },
        
        /**
         * Get water predictions
         */
        async water(data, periods = 24, frequency = 'H') {
            return API.post('/predict/water', { data, periods, frequency });
        },
        
        /**
         * Get predictions for both resources
         */
        async both(electricityData, waterData, periods = 24, frequency = 'H') {
            return API.post('/predict/both', {
                electricity: { data: electricityData, periods, frequency },
                water: { data: waterData, periods, frequency }
            });
        },
        
        /**
         * Get demo predictions
         */
        async demo(resource = 'both', periods = 24) {
            return API.get('/predict/demo', { resource, periods });
        },
        
        /**
         * Quick predict based on average
         */
        async quick(resourceType, recentAverage, periods = 24) {
            return API.post('/predict/quick-predict', {
                resource_type: resourceType,
                recent_average: recentAverage,
                periods
            });
        },
        
        /**
         * Get sustainability score
         */
        async sustainabilityScore(electricityDaily, waterDaily, householdSize = 4) {
            return API.post('/predict/sustainability-score', {
                electricity: { daily_consumption_kwh: electricityDaily },
                water: { daily_consumption_liters: waterDaily },
                household_size: householdSize
            });
        }
    },
    
    // ----- Electricity Endpoints -----
    
    electricity: {
        /**
         * Get module info
         */
        async info() {
            return API.get('/electricity/');
        },
        
        /**
         * Analyze consumption
         */
        async analyze(data) {
            return API.post('/electricity/analyze', { data });
        },
        
        /**
         * Get recommendations
         */
        async recommendations() {
            return API.get('/electricity/recommendations');
        },
        
        /**
         * Get cost estimate
         */
        async costEstimate(consumptionKwh, ratePerKwh = null) {
            const payload = { consumption_kwh: consumptionKwh };
            if (ratePerKwh) payload.rate_per_kwh = ratePerKwh;
            return API.post('/electricity/cost-estimate', payload);
        },
        
        /**
         * Get peak hours info
         */
        async peakHours() {
            return API.get('/electricity/peak-hours');
        }
    },
    
    // ----- Water Endpoints -----
    
    water: {
        /**
         * Get module info
         */
        async info() {
            return API.get('/water/');
        },
        
        /**
         * Analyze consumption
         */
        async analyze(data) {
            return API.post('/water/analyze', { data });
        },
        
        /**
         * Detect leaks
         */
        async detectLeaks(data, sensitivity = 'medium') {
            return API.post('/water/leak-detection', { data, sensitivity });
        },
        
        /**
         * Get recommendations
         */
        async recommendations() {
            return API.get('/water/recommendations');
        },
        
        /**
         * Get cost estimate
         */
        async costEstimate(consumptionLiters, ratePerLiter = null) {
            const payload = { consumption_liters: consumptionLiters };
            if (ratePerLiter) payload.rate_per_liter = ratePerLiter;
            return API.post('/water/cost-estimate', payload);
        },
        
        /**
         * Get usage patterns
         */
        async usagePatterns() {
            return API.get('/water/usage-patterns');
        }
    },
    
    // ----- Dashboard Endpoints -----
    
    dashboard: {
        /**
         * Get dashboard overview
         */
        async overview(period = 'today') {
            return API.get('/dashboard/overview', { period });
        },
        
        /**
         * Get statistics
         */
        async stats(period = 'month', resource = 'both') {
            return API.get('/dashboard/stats', { period, resource });
        },
        
        /**
         * Get recent activity
         */
        async recentActivity(limit = 10) {
            return API.get('/dashboard/recent', { limit });
        },
        
        /**
         * Get predictions summary
         */
        async predictionsSummary(periods = 24) {
            return API.get('/dashboard/predictions-summary', { periods });
        },
        
        /**
         * Get cost summary
         */
        async costSummary(period = 'month') {
            return API.get('/dashboard/cost-summary', { period });
        },
        
        /**
         * Get sustainability metrics
         */
        async sustainability() {
            return API.get('/dashboard/sustainability');
        },
        
        /**
         * Get alerts
         */
        async alerts() {
            return API.get('/dashboard/alerts');
        },
        
        /**
         * Get tips
         */
        async tips(category = 'all', limit = 5) {
            return API.get('/dashboard/tips', { category, limit });
        },
        
        /**
         * Get chart data
         */
        async chartData(chartType, period = 'week', resource = 'both') {
            return API.get('/dashboard/chart-data', {
                chart_type: chartType,
                period,
                resource
            });
        }
    },
    
    // ----- User Endpoints -----
    
    user: {
        /**
         * Register new user
         */
        async register(email, name, password, householdSize = 4, location = null) {
            return API.post('/user/register', {
                email,
                name,
                password,
                household_size: householdSize,
                location
            });
        },
        
        /**
         * Login
         */
        async login(email, password) {
            return API.post('/user/login', { email, password });
        },
        
        /**
         * Logout
         */
        async logout() {
            return API.post('/user/logout');
        },
        
        /**
         * Get profile
         */
        async profile() {
            return API.get('/user/profile');
        },
        
        /**
         * Update profile
         */
        async updateProfile(data) {
            return API.put('/user/profile', data);
        },
        
        /**
         * Save usage entry
         */
        async saveUsage(resourceType, consumption, datetime = null, notes = null) {
            const payload = {
                resource_type: resourceType,
                consumption
            };
            if (datetime) payload.datetime = datetime;
            if (notes) payload.notes = notes;
            return API.post('/user/save-usage', payload);
        },
        
        /**
         * Save bulk usage entries
         */
        async saveUsageBulk(entries) {
            return API.post('/user/save-usage/bulk', { entries });
        },
        
        /**
         * Get usage history
         */
        async history(resourceType = null, startDate = null, endDate = null, limit = 100) {
            const params = { limit };
            if (resourceType) params.resource_type = resourceType;
            if (startDate) params.start_date = startDate;
            if (endDate) params.end_date = endDate;
            return API.get('/user/history', params);
        },
        
        /**
         * Delete history entry
         */
        async deleteHistory(entryId) {
            return API.delete(`/user/history/${entryId}`);
        },
        
        /**
         * Get preferences
         */
        async preferences() {
            return API.get('/user/preferences');
        },
        
        /**
         * Update preferences
         */
        async updatePreferences(preferences) {
            return API.put('/user/preferences', preferences);
        },
        
        /**
         * Get consumption summary
         */
        async summary(period = 'month') {
            return API.get('/user/summary', { period });
        }
    }
};


/**
 * Custom API Error class
 */
class APIError extends Error {
    constructor(message, status, data) {
        super(message);
        this.name = 'APIError';
        this.status = status;
        this.data = data;
    }
}


/**
 * Utility functions for API responses
 */
const APIUtils = {
    /**
     * Format consumption value
     */
    formatConsumption(value, resourceType) {
        const unit = resourceType === 'electricity' ? 'kWh' : 'L';
        return `${value.toFixed(2)} ${unit}`;
    },
    
    /**
     * Format currency
     */
    formatCurrency(amount, currency = 'USD') {
        return new Intl.NumberFormat('en-US', {
            style: 'currency',
            currency: currency
        }).format(amount);
    },
    
    /**
     * Format datetime for API
     */
    formatDateTime(date) {
        return date.toISOString();
    },
    
    /**
     * Parse datetime from API
     */
    parseDateTime(dateString) {
        return new Date(dateString);
    },
    
    /**
     * Create consumption data array from user input
     */
    createConsumptionData(values, startDate, frequency = 'H') {
        const data = [];
        const date = new Date(startDate);
        
        const incrementHours = frequency === 'H' ? 1 : frequency === 'D' ? 24 : 168;
        
        values.forEach((value, index) => {
            const entryDate = new Date(date.getTime() + index * incrementHours * 60 * 60 * 1000);
            data.push({
                datetime: entryDate.toISOString(),
                consumption: parseFloat(value)
            });
        });
        
        return data;
    }
};


// Make available globally
window.API = API;
window.APIError = APIError;
window.APIUtils = APIUtils;