/* ============================================
   EchoMind - Dashboard JavaScript
   ============================================ */

/**
 * Dashboard controller for EchoMind
 * Handles dashboard data loading and visualization
 */

const Dashboard = {
    // Chart instances
    charts: {},
    
    // Current period
    currentPeriod: 'today',
    
    // Refresh interval (ms)
    refreshInterval: 300000, // 5 minutes
    
    /**
     * Initialize dashboard
     */
    async init() {
        console.log('Initializing dashboard...');
        
        // Setup event listeners
        this.setupEventListeners();
        
        // Load initial data
        await this.loadDashboard();
        
        // Setup auto-refresh
        this.setupAutoRefresh();
        
        console.log('Dashboard initialized');
    },
    
    /**
     * Setup event listeners
     */
    setupEventListeners() {
        // Period selector
        document.querySelectorAll('[data-period]').forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.changePeriod(e.target.dataset.period);
            });
        });
        
        // Refresh button
        const refreshBtn = document.getElementById('refresh-dashboard');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', () => this.loadDashboard());
        }
        
        // Mobile menu toggle
        const menuToggle = document.getElementById('menu-toggle');
        if (menuToggle) {
            menuToggle.addEventListener('click', () => {
                document.querySelector('.sidebar').classList.toggle('open');
            });
        }
    },
    
    /**
     * Setup auto-refresh
     */
    setupAutoRefresh() {
        setInterval(() => {
            this.loadDashboard();
        }, this.refreshInterval);
    },
    
    /**
     * Change period and reload
     */
    async changePeriod(period) {
        this.currentPeriod = period;
        
        // Update active button
        document.querySelectorAll('[data-period]').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.period === period);
        });
        
        await this.loadDashboard();
    },
    
    /**
     * Load all dashboard data
     */
    async loadDashboard() {
        try {
            this.showLoading();
            
            // Load all data in parallel - pass period to overview which includes predictions
            const [overview, sustainability, alerts] = await Promise.all([
                API.dashboard.overview(this.currentPeriod),
                API.dashboard.sustainability(),
                API.dashboard.alerts()
            ]);
            
            // Update UI components - predictions are included in overview
            this.updateStats(overview.overview?.statistics);
            this.updatePredictions(overview.overview?.predictions);
            this.updateSustainability(sustainability);
            this.updateAlerts(alerts);
            this.updateCharts(overview);
            this.updateTips(overview.overview?.tips);
            
            this.hideLoading();
            
        } catch (error) {
            console.error('Dashboard load error:', error);
            this.showError('Failed to load dashboard data');
            this.hideLoading();
        }
    },
    
    /**
     * Update statistics cards
     */
    updateStats(stats) {
        if (!stats) return;
        
        // Electricity stats
        const elecStats = stats.electricity;
        if (elecStats) {
            this.updateElement('electricity-consumption', 
                `${elecStats.consumption?.toFixed(1) || 0} kWh`);
            this.updateElement('electricity-trend', 
                this.getTrendIcon(elecStats.trend), true);
        }
        
        // Water stats
        const waterStats = stats.water;
        if (waterStats) {
            this.updateElement('water-consumption', 
                `${waterStats.consumption?.toFixed(0) || 0} L`);
            this.updateElement('water-trend', 
                this.getTrendIcon(waterStats.trend), true);
        }
    },
    
    /**
     * Update predictions section
     */
    updatePredictions(data) {
        if (!data?.predictions_summary) return;
        
        const summary = data.predictions_summary;
        
        // Electricity prediction
        if (summary.electricity) {
            this.updateElement('electricity-predicted',
                `${summary.electricity.total_predicted?.toFixed(1) || 0} kWh`);
            this.updateElement('electricity-cost',
                `$${summary.electricity.total_estimated_cost?.toFixed(2) || 0}`);
        }
        
        // Water prediction
        if (summary.water) {
            this.updateElement('water-predicted',
                `${summary.water.total_predicted?.toFixed(0) || 0} L`);
            this.updateElement('water-cost',
                `$${summary.water.total_estimated_cost?.toFixed(2) || 0}`);
        }
    },
    
    /**
     * Update sustainability score
     */
    updateSustainability(data) {
        if (!data?.sustainability) return;
        
        const sus = data.sustainability;
        
        // Update score circle
        const scoreCircle = document.querySelector('.score-circle');
        if (scoreCircle) {
            scoreCircle.style.setProperty('--score', sus.overall_score || 0);
        }
        
        this.updateElement('sustainability-score', sus.overall_score || 0);
        this.updateElement('sustainability-grade', sus.grade || 'N/A');
        
        // Update improvement tips
        if (sus.improvement_tips) {
            const tipsContainer = document.getElementById('improvement-tips');
            if (tipsContainer) {
                tipsContainer.innerHTML = sus.improvement_tips
                    .map(tip => `<li>${tip}</li>`)
                    .join('');
            }
        }
    },
    
    /**
     * Update alerts
     */
    updateAlerts(data) {
        const alertsContainer = document.getElementById('alerts-container');
        if (!alertsContainer) return;
        
        const alerts = data?.alerts || [];
        
        if (alerts.length === 0) {
            alertsContainer.innerHTML = `
                <div class="alert alert-success">
                    <i class="fas fa-check-circle"></i>
                    <span>No active alerts. Everything looks good!</span>
                </div>`;
            return;
        }
        
        alertsContainer.innerHTML = alerts.map(alert => `
            <div class="alert alert-${this.getPriorityClass(alert.priority)}">
                <i class="fas fa-${this.getPriorityIcon(alert.priority)}"></i>
                <div>
                    <strong>${alert.title}</strong>
                    <p>${alert.message}</p>
                </div>
            </div>
        `).join('');
    },
    
    /**
     * Update charts
     */
    updateCharts(data) {
        // Update consumption chart
        this.updateConsumptionChart(data);
        
        // Update comparison chart
        this.updateComparisonChart(data);
    },
    
    /**
     * Update consumption chart
     */
    updateConsumptionChart(data) {
        const ctx = document.getElementById('consumption-chart');
        if (!ctx) return;
        
        // Destroy existing chart
        if (this.charts.consumption) {
            this.charts.consumption.destroy();
        }
        
        // Generate sample data (replace with actual data)
        const labels = this.generateTimeLabels(24);
        const electricityData = this.generateSampleData(24, 1, 3);
        const waterData = this.generateSampleData(24, 20, 60);
        
        this.charts.consumption = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Electricity (kWh)',
                        data: electricityData,
                        borderColor: '#f59e0b',
                        backgroundColor: 'rgba(245, 158, 11, 0.1)',
                        fill: true,
                        tension: 0.4,
                        yAxisID: 'y'
                    },
                    {
                        label: 'Water (L)',
                        data: waterData,
                        borderColor: '#06b6d4',
                        backgroundColor: 'rgba(6, 182, 212, 0.1)',
                        fill: true,
                        tension: 0.4,
                        yAxisID: 'y1'
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
                scales: {
                    y: {
                        type: 'linear',
                        position: 'left',
                        title: {
                            display: true,
                            text: 'Electricity (kWh)'
                        }
                    },
                    y1: {
                        type: 'linear',
                        position: 'right',
                        title: {
                            display: true,
                            text: 'Water (L)'
                        },
                        grid: {
                            drawOnChartArea: false
                        }
                    }
                },
                plugins: {
                    legend: {
                        position: 'top'
                    }
                }
            }
        });
    },
    
    /**
     * Update comparison chart
     */
    updateComparisonChart(data) {
        const ctx = document.getElementById('comparison-chart');
        if (!ctx) return;
        
        // Destroy existing chart
        if (this.charts.comparison) {
            this.charts.comparison.destroy();
        }
        
        this.charts.comparison = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                datasets: [
                    {
                        label: 'This Week',
                        data: [15, 18, 14, 20, 16, 25, 22],
                        backgroundColor: '#2563eb'
                    },
                    {
                        label: 'Last Week',
                        data: [17, 19, 16, 18, 17, 23, 20],
                        backgroundColor: '#94a3b8'
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Consumption'
                        }
                    }
                },
                plugins: {
                    legend: {
                        position: 'top'
                    }
                }
            }
        });
    },
    
    /**
     * Update tips section
     */
    updateTips(tips) {
        const tipsContainer = document.getElementById('tips-container');
        if (!tipsContainer || !tips) return;
        
        tipsContainer.innerHTML = tips.map(tip => `
            <div class="tip-card">
                <div class="tip-icon ${tip.category}">
                    <i class="fas fa-${this.getTipIcon(tip.category)}"></i>
                </div>
                <div class="tip-content">
                    <p>${tip.tip}</p>
                    <span class="tip-impact">${tip.impact}</span>
                </div>
            </div>
        `).join('');
    },
    
    // ----- Helper Methods -----
    
    /**
     * Update element text
     */
    updateElement(id, value, useHTML = false) {
        const el = document.getElementById(id);
        if (el) {
            if (useHTML) {
                el.innerHTML = value;
            } else {
                el.textContent = value;
            }
        }
    },
    
    /**
     * Get trend icon HTML
     */
    getTrendIcon(trend) {
        const icons = {
            increasing: '<span class="trend-icon text-error" title="Increasing">&#9650;</span>',
            decreasing: '<span class="trend-icon text-success" title="Decreasing">&#9660;</span>',
            stable: '<span class="trend-icon text-muted" title="Stable">&#8212;</span>'
        };
        return icons[trend] || icons.stable;
    },
    
    /**
     * Get priority class for alerts
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
     * Get priority icon
     */
    getPriorityIcon(priority) {
        const icons = {
            critical: 'exclamation-circle',
            high: 'exclamation-triangle',
            medium: 'info-circle',
            low: 'check-circle'
        };
        return icons[priority] || 'info-circle';
    },
    
    /**
     * Get tip icon by category
     */
    getTipIcon(category) {
        const icons = {
            electricity: 'bolt',
            water: 'tint',
            general: 'lightbulb'
        };
        return icons[category] || 'lightbulb';
    },
    
    /**
     * Generate time labels
     */
    generateTimeLabels(count) {
        const labels = [];
        const now = new Date();
        for (let i = count - 1; i >= 0; i--) {
            const time = new Date(now - i * 3600000);
            labels.push(time.getHours() + ':00');
        }
        return labels;
    },
    
    /**
     * Generate sample data
     */
    generateSampleData(count, min, max) {
        return Array.from({ length: count }, () => 
            Math.random() * (max - min) + min
        );
    },
    
    /**
     * Show loading state
     */
    showLoading() {
        const loader = document.getElementById('dashboard-loader');
        if (loader) loader.classList.remove('hidden');
    },
    
    /**
     * Hide loading state
     */
    hideLoading() {
        const loader = document.getElementById('dashboard-loader');
        if (loader) loader.classList.add('hidden');
    },
    
    /**
     * Show error message
     */
    showError(message) {
        const container = document.getElementById('error-container');
        if (container) {
            container.innerHTML = `
                <div class="alert alert-error">
                    <i class="fas fa-exclamation-circle"></i>
                    <span>${message}</span>
                </div>`;
            container.classList.remove('hidden');
        }
    }
};


// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    Dashboard.init();
});


// Export for module use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = Dashboard;
}

// Make available globally
window.Dashboard = Dashboard;