/* ============================================
   EchoMind - Predictions JavaScript
   ============================================ */

/**
 * Predictions page controller for EchoMind
 * Handles prediction display, visualization, and analysis
 */

const Predictions = {
    // Current prediction data
    predictionData: null,
    
    // Current resource type
    resourceType: 'electricity',
    
    // Chart instances
    charts: {},
    
    /**
     * Initialize predictions page
     */
    init() {
        console.log('Initializing predictions page...');
        
        // Load prediction data from session
        this.loadPredictionData();
        
        // Setup event listeners
        this.setupEventListeners();
        
        // Display predictions if available
        if (this.predictionData) {
            this.displayPredictions();
        } else {
            this.showNoPredictions();
        }
        
        console.log('Predictions page initialized');
    },
    
    /**
     * Load prediction data from session storage
     */
    loadPredictionData() {
        const stored = sessionStorage.getItem('predictionResult');
        const resource = sessionStorage.getItem('predictionResource');
        
        if (stored) {
            try {
                this.predictionData = JSON.parse(stored);
                this.resourceType = resource || 'electricity';
            } catch (e) {
                console.error('Error parsing prediction data:', e);
            }
        }
    },
    
    /**
     * Setup event listeners
     */
    setupEventListeners() {
        // Resource type tabs
        document.querySelectorAll('[data-resource]').forEach(tab => {
            tab.addEventListener('click', (e) => {
                this.switchResource(e.target.dataset.resource);
            });
        });
        
        // Download buttons
        const downloadCSV = document.getElementById('download-csv');
        if (downloadCSV) {
            downloadCSV.addEventListener('click', () => this.downloadCSV());
        }
        
        const downloadPDF = document.getElementById('download-pdf');
        if (downloadPDF) {
            downloadPDF.addEventListener('click', () => this.downloadPDF());
        }
        
        // New prediction button
        const newPrediction = document.getElementById('new-prediction');
        if (newPrediction) {
            newPrediction.addEventListener('click', () => {
                window.location.href = '/input';
            });
        }
        
        // Demo predictions button
        const demoBtn = document.getElementById('load-demo');
        if (demoBtn) {
            demoBtn.addEventListener('click', () => this.loadDemoPredictions());
        }
        
        // Period selector
        document.querySelectorAll('[data-periods]').forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.changePeriods(parseInt(e.target.dataset.periods));
            });
        });
    },
    
    /**
     * Switch resource type
     */
    async switchResource(resource) {
        this.resourceType = resource;
        
        // Update tabs
        document.querySelectorAll('[data-resource]').forEach(tab => {
            tab.classList.toggle('active', tab.dataset.resource === resource);
        });
        
        // Load demo predictions for the new resource
        await this.loadDemoPredictions();
    },
    
    /**
     * Load demo predictions
     */
    async loadDemoPredictions() {
        try {
            this.showLoading();
            
            const result = await API.predictions.demo(this.resourceType, 24);
            
            if (result.success) {
                this.predictionData = result[this.resourceType] || result;
                this.displayPredictions();
            } else {
                this.showError('Failed to load demo predictions');
            }
            
            this.hideLoading();
        } catch (error) {
            this.hideLoading();
            this.showError(error.message);
        }
    },
    
    /**
     * Change prediction periods
     */
    async changePeriods(periods) {
        // Update button states
        document.querySelectorAll('[data-periods]').forEach(btn => {
            btn.classList.toggle('active', parseInt(btn.dataset.periods) === periods);
        });
        
        try {
            this.showLoading();
            
            const result = await API.predictions.demo(this.resourceType, periods);
            
            if (result.success) {
                this.predictionData = result[this.resourceType] || result;
                this.displayPredictions();
            }
            
            this.hideLoading();
        } catch (error) {
            this.hideLoading();
            this.showError(error.message);
        }
    },
    
    /**
     * Display predictions
     */
    displayPredictions() {
        if (!this.predictionData) {
            this.showNoPredictions();
            return;
        }
        
        // Show predictions container
        const container = document.getElementById('predictions-container');
        if (container) container.classList.remove('hidden');
        
        const noData = document.getElementById('no-predictions');
        if (noData) noData.classList.add('hidden');
        
        // Update all components
        this.updateSummary();
        this.updateChart();
        this.updateTable();
        this.updateRecommendations();
        this.updateCostAnalysis();
    },
    
    /**
     * Update summary cards
     */
    updateSummary() {
        const summary = this.predictionData.summary;
        if (!summary) return;
        
        const unit = this.resourceType === 'electricity' ? 'kWh' : 'L';
        
        // Update summary values
        this.updateElement('total-predicted', 
            `${summary.total_predicted?.toFixed(2) || 0} ${unit}`);
        this.updateElement('average-predicted', 
            `${summary.average_predicted?.toFixed(2) || 0} ${unit}`);
        this.updateElement('min-predicted', 
            `${summary.min_predicted?.toFixed(2) || 0} ${unit}`);
        this.updateElement('max-predicted', 
            `${summary.max_predicted?.toFixed(2) || 0} ${unit}`);
        this.updateElement('total-cost', 
            `$${summary.total_estimated_cost?.toFixed(2) || 0}`);
        this.updateElement('prediction-periods', 
            `${summary.periods || 0} hours`);
    },
    
    /**
     * Update prediction chart
     */
    updateChart() {
        const ctx = document.getElementById('prediction-chart');
        if (!ctx) return;
        
        const predictions = this.predictionData.predictions;
        if (!predictions || predictions.length === 0) return;
        
        // Destroy existing chart
        if (this.charts.prediction) {
            this.charts.prediction.destroy();
        }
        
        // Prepare data
        const labels = predictions.map(p => this.formatChartLabel(p.datetime));
        const values = predictions.map(p => p.predicted_value);
        const upperBounds = predictions.map(p => p.upper_bound);
        const lowerBounds = predictions.map(p => p.lower_bound);
        
        // Get colors based on resource type
        const colors = this.resourceType === 'electricity' 
            ? { main: '#f59e0b', light: 'rgba(245, 158, 11, 0.2)' }
            : { main: '#06b6d4', light: 'rgba(6, 182, 212, 0.2)' };
        
        this.charts.prediction = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Predicted',
                        data: values,
                        borderColor: colors.main,
                        backgroundColor: colors.light,
                        fill: false,
                        tension: 0.4,
                        pointRadius: 3,
                        pointHoverRadius: 6
                    },
                    {
                        label: 'Upper Bound',
                        data: upperBounds,
                        borderColor: 'rgba(156, 163, 175, 0.5)',
                        backgroundColor: 'transparent',
                        borderDash: [5, 5],
                        fill: false,
                        tension: 0.4,
                        pointRadius: 0
                    },
                    {
                        label: 'Lower Bound',
                        data: lowerBounds,
                        borderColor: 'rgba(156, 163, 175, 0.5)',
                        backgroundColor: colors.light,
                        borderDash: [5, 5],
                        fill: '-1',
                        tension: 0.4,
                        pointRadius: 0
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    mode: 'index',
                    intersect: false
                },
                plugins: {
                    legend: {
                        position: 'top'
                    },
                    tooltip: {
                        callbacks: {
                            label: (context) => {
                                const unit = this.resourceType === 'electricity' ? 'kWh' : 'L';
                                return `${context.dataset.label}: ${context.parsed.y.toFixed(2)} ${unit}`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        display: true,
                        title: {
                            display: true,
                            text: 'Time'
                        },
                        ticks: {
                            maxTicksLimit: 12
                        }
                    },
                    y: {
                        display: true,
                        title: {
                            display: true,
                            text: this.resourceType === 'electricity' ? 'Consumption (kWh)' : 'Consumption (L)'
                        },
                        beginAtZero: true
                    }
                }
            }
        });
    },
    
    /**
     * Update predictions table
     */
    updateTable() {
        const container = document.getElementById('predictions-table');
        if (!container) return;
        
        const predictions = this.predictionData.predictions;
        if (!predictions || predictions.length === 0) {
            container.innerHTML = '<p class="text-muted">No prediction data available</p>';
            return;
        }
        
        const unit = this.resourceType === 'electricity' ? 'kWh' : 'L';
        
        // Show first 24 entries, with option to show more
        const displayPredictions = predictions.slice(0, 24);
        
        container.innerHTML = `
            <div class="table-container">
                <table class="table">
                    <thead>
                        <tr>
                            <th>Date/Time</th>
                            <th>Predicted</th>
                            <th>Range</th>
                            <th>Est. Cost</th>
                            ${this.resourceType === 'electricity' ? '<th>Peak</th>' : '<th>Period</th>'}
                        </tr>
                    </thead>
                    <tbody>
                        ${displayPredictions.map(p => `
                            <tr>
                                <td>${this.formatDateTime(p.datetime)}</td>
                                <td><strong>${p.predicted_value.toFixed(2)} ${unit}</strong></td>
                                <td class="text-muted">${p.lower_bound.toFixed(2)} - ${p.upper_bound.toFixed(2)}</td>
                                <td>$${p.estimated_cost?.toFixed(4) || '0.00'}</td>
                                <td>${this.getPeriodBadge(p)}</td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            </div>
            ${predictions.length > 24 ? `
                <button class="btn btn-secondary mt-md" onclick="Predictions.showAllPredictions()">
                    Show All ${predictions.length} Predictions
                </button>
            ` : ''}
        `;
    },
    
    /**
     * Show all predictions
     */
    showAllPredictions() {
        const container = document.getElementById('predictions-table');
        if (!container) return;
        
        const predictions = this.predictionData.predictions;
        const unit = this.resourceType === 'electricity' ? 'kWh' : 'L';
        
        container.innerHTML = `
            <div class="table-container" style="max-height: 500px; overflow-y: auto;">
                <table class="table">
                    <thead>
                        <tr>
                            <th>Date/Time</th>
                            <th>Predicted</th>
                            <th>Range</th>
                            <th>Est. Cost</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${predictions.map(p => `
                            <tr>
                                <td>${this.formatDateTime(p.datetime)}</td>
                                <td><strong>${p.predicted_value.toFixed(2)} ${unit}</strong></td>
                                <td class="text-muted">${p.lower_bound.toFixed(2)} - ${p.upper_bound.toFixed(2)}</td>
                                <td>$${p.estimated_cost?.toFixed(4) || '0.00'}</td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            </div>
        `;
    },
    
    /**
     * Update recommendations
     */
    updateRecommendations() {
        const container = document.getElementById('recommendations-container');
        if (!container) return;
        
        const recommendations = this.predictionData.recommendations;
        
        if (!recommendations || recommendations.length === 0) {
            container.innerHTML = `
                <div class="alert alert-success">
                    <i class="fas fa-check-circle"></i>
                    <span>Your usage patterns look efficient! Keep up the good work.</span>
                </div>`;
            return;
        }
        
        container.innerHTML = recommendations.map(rec => `
            <div class="recommendation-card ${rec.priority}">
                <div class="recommendation-header">
                    <span class="badge badge-${this.getPriorityClass(rec.priority)}">
                        ${rec.priority.toUpperCase()}
                    </span>
                    <span class="recommendation-type">${rec.type}</span>
                </div>
                <p class="recommendation-message">${rec.message}</p>
                ${rec.potential_savings ? `
                    <p class="recommendation-savings">
                        <i class="fas fa-piggy-bank"></i> 
                        Potential savings: ${rec.potential_savings}
                    </p>
                ` : ''}
            </div>
        `).join('');
    },
    
    /**
     * Update cost analysis
     */
    updateCostAnalysis() {
        const container = document.getElementById('cost-analysis');
        if (!container) return;
        
        const summary = this.predictionData.summary;
        if (!summary) return;
        
        const totalCost = summary.total_estimated_cost || 0;
        const periods = summary.periods || 24;
        
        // Calculate projections
        const hourlyRate = totalCost / periods;
        const dailyCost = hourlyRate * 24;
        const weeklyCost = dailyCost * 7;
        const monthlyCost = dailyCost * 30;
        
        container.innerHTML = `
            <div class="cost-grid">
                <div class="cost-item">
                    <span class="cost-label">Hourly Average</span>
                    <span class="cost-value">$${hourlyRate.toFixed(4)}</span>
                </div>
                <div class="cost-item">
                    <span class="cost-label">Daily Projection</span>
                    <span class="cost-value">$${dailyCost.toFixed(2)}</span>
                </div>
                <div class="cost-item">
                    <span class="cost-label">Weekly Projection</span>
                    <span class="cost-value">$${weeklyCost.toFixed(2)}</span>
                </div>
                <div class="cost-item highlight">
                    <span class="cost-label">Monthly Projection</span>
                    <span class="cost-value">$${monthlyCost.toFixed(2)}</span>
                </div>
            </div>
            <p class="text-muted text-center mt-md">
                Based on ${periods} hours of predicted consumption
            </p>
        `;
    },
    
    /**
     * Show no predictions message
     */
    showNoPredictions() {
        const container = document.getElementById('predictions-container');
        if (container) container.classList.add('hidden');
        
        const noData = document.getElementById('no-predictions');
        if (noData) noData.classList.remove('hidden');
    },
    
    /**
     * Download predictions as CSV
     */
    downloadCSV() {
        if (!this.predictionData?.predictions) {
            this.showError('No prediction data to download');
            return;
        }
        
        const predictions = this.predictionData.predictions;
        const unit = this.resourceType === 'electricity' ? 'kWh' : 'liters';
        
        // Create CSV content
        const headers = ['DateTime', `Predicted (${unit})`, 'Lower Bound', 'Upper Bound', 'Estimated Cost (USD)'];
        const rows = predictions.map(p => [
            p.datetime,
            p.predicted_value.toFixed(4),
            p.lower_bound.toFixed(4),
            p.upper_bound.toFixed(4),
            p.estimated_cost?.toFixed(6) || '0'
        ]);
        
        const csv = [headers.join(','), ...rows.map(r => r.join(','))].join('\n');
        
        // Download
        const blob = new Blob([csv], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `echomind_${this.resourceType}_predictions_${new Date().toISOString().split('T')[0]}.csv`;
        a.click();
        URL.revokeObjectURL(url);
        
        this.showSuccess('CSV downloaded successfully');
    },
    
    /**
     * Download predictions as PDF (simplified version)
     */
    downloadPDF() {
        // For a full implementation, you'd use a library like jsPDF
        // This is a simplified version that opens print dialog
        window.print();
    },
    
    // ----- Helper Methods -----
    
    /**
     * Update element text
     */
    updateElement(id, value) {
        const el = document.getElementById(id);
        if (el) el.textContent = value;
    },
    
    /**
     * Format datetime for display
     */
    formatDateTime(isoString) {
        const date = new Date(isoString);
        return date.toLocaleString('en-US', {
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        });
    },
    
    /**
     * Format datetime for chart labels
     */
    formatChartLabel(isoString) {
        const date = new Date(isoString);
        return date.toLocaleString('en-US', {
            month: 'short',
            day: 'numeric',
            hour: '2-digit'
        });
    },
    
    /**
     * Get period badge HTML
     */
    getPeriodBadge(prediction) {
        if (this.resourceType === 'electricity') {
            const isPeak = prediction.is_peak_hour;
            return isPeak 
                ? '<span class="badge badge-warning">Peak</span>'
                : '<span class="badge badge-success">Off-Peak</span>';
        } else {
            const period = prediction.usage_period || 'unknown';
            const badges = {
                'morning_peak': '<span class="badge badge-info">Morning</span>',
                'evening_peak': '<span class="badge badge-warning">Evening</span>',
                'midday': '<span class="badge badge-success">Midday</span>',
                'night': '<span class="badge badge-secondary">Night</span>'
            };
            return badges[period] || '<span class="badge">Unknown</span>';
        }
    },
    
    /**
     * Get priority class
     */
    getPriorityClass(priority) {
        const classes = {
            critical: 'error',
            high: 'warning',
            medium: 'info',
            low: 'success'
        };
        return classes[priority] || 'info';
    },
    
    /**
     * Show loading state
     */
    showLoading() {
        const loader = document.getElementById('predictions-loader');
        if (loader) loader.classList.remove('hidden');
    },
    
    /**
     * Hide loading state
     */
    hideLoading() {
        const loader = document.getElementById('predictions-loader');
        if (loader) loader.classList.add('hidden');
    },
    
    /**
     * Show success message
     */
    showSuccess(message) {
        this.showMessage(message, 'success');
    },
    
    /**
     * Show error message
     */
    showError(message) {
        this.showMessage(message, 'error');
    },
    
    /**
     * Show message
     */
    showMessage(message, type) {
        const container = document.getElementById('message-container');
        if (!container) return;
        
        container.innerHTML = `
            <div class="alert alert-${type}">
                <i class="fas fa-${type === 'success' ? 'check-circle' : 'exclamation-circle'}"></i>
                <span>${message}</span>
            </div>`;
        container.classList.remove('hidden');
        
        setTimeout(() => {
            container.classList.add('hidden');
        }, 5000);
    }
};


// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    Predictions.init();
});


// Export for module use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = Predictions;
}

// Make available globally
window.Predictions = Predictions;