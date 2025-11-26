/* ============================================
   EchoMind - User Input JavaScript
   ============================================ */

/**
 * User Input handler for EchoMind
 * Handles user data input, validation, and submission
 */

const UserInput = {
    // Current resource type
    currentResource: 'electricity',
    
    // Form data
    formData: {
        electricity: [],
        water: []
    },
    
    // Input mode: 'manual' or 'csv'
    inputMode: 'manual',
    
    /**
     * Initialize user input page
     */
    init() {
        console.log('Initializing user input...');
        
        this.setupEventListeners();
        this.initializeDatePickers();
        this.loadSavedData();
        
        console.log('User input initialized');
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
        
        // Input mode toggle
        document.querySelectorAll('[data-input-mode]').forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.switchInputMode(e.target.dataset.inputMode);
            });
        });
        
        // Add entry button
        const addEntryBtn = document.getElementById('add-entry');
        if (addEntryBtn) {
            addEntryBtn.addEventListener('click', () => this.addEntry());
        }
        
        // CSV file input
        const csvInput = document.getElementById('csv-file');
        if (csvInput) {
            csvInput.addEventListener('change', (e) => this.handleCSVUpload(e));
        }
        
        // Form submission
        const submitBtn = document.getElementById('submit-data');
        if (submitBtn) {
            submitBtn.addEventListener('click', () => this.submitData());
        }
        
        // Clear data button
        const clearBtn = document.getElementById('clear-data');
        if (clearBtn) {
            clearBtn.addEventListener('click', () => this.clearData());
        }
        
        // Quick add presets
        document.querySelectorAll('[data-preset]').forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.applyPreset(e.target.dataset.preset);
            });
        });
    },
    
    /**
     * Initialize date pickers
     */
    initializeDatePickers() {
        // Set default datetime to now
        const datetimeInput = document.getElementById('entry-datetime');
        if (datetimeInput) {
            const now = new Date();
            now.setMinutes(now.getMinutes() - now.getTimezoneOffset());
            datetimeInput.value = now.toISOString().slice(0, 16);
        }
    },
    
    /**
     * Switch resource type
     */
    switchResource(resource) {
        this.currentResource = resource;
        
        // Update tabs
        document.querySelectorAll('[data-resource]').forEach(tab => {
            tab.classList.toggle('active', tab.dataset.resource === resource);
        });
        
        // Update form labels
        this.updateFormLabels();
        
        // Update entries display
        this.renderEntries();
    },
    
    /**
     * Update form labels based on resource
     */
    updateFormLabels() {
        const unitLabel = document.getElementById('unit-label');
        const unitHint = document.getElementById('unit-hint');
        
        if (this.currentResource === 'electricity') {
            if (unitLabel) unitLabel.textContent = 'kWh';
            if (unitHint) unitHint.textContent = 'Enter consumption in kilowatt-hours';
        } else {
            if (unitLabel) unitLabel.textContent = 'Liters';
            if (unitHint) unitHint.textContent = 'Enter consumption in liters';
        }
    },
    
    /**
     * Switch input mode
     */
    switchInputMode(mode) {
        this.inputMode = mode;
        
        // Update buttons
        document.querySelectorAll('[data-input-mode]').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.inputMode === mode);
        });
        
        // Show/hide sections
        const manualSection = document.getElementById('manual-input');
        const csvSection = document.getElementById('csv-input');
        
        if (manualSection) {
            manualSection.classList.toggle('hidden', mode !== 'manual');
        }
        if (csvSection) {
            csvSection.classList.toggle('hidden', mode !== 'csv');
        }
    },
    
    /**
     * Add a single entry
     */
    addEntry() {
        const datetimeInput = document.getElementById('entry-datetime');
        const valueInput = document.getElementById('entry-value');
        const notesInput = document.getElementById('entry-notes');
        
        // Validate inputs
        if (!datetimeInput.value || !valueInput.value) {
            this.showError('Please enter both date/time and consumption value');
            return;
        }
        
        const value = parseFloat(valueInput.value);
        if (isNaN(value) || value < 0) {
            this.showError('Please enter a valid positive number');
            return;
        }
        
        // Create entry
        const entry = {
            id: Date.now(),
            datetime: new Date(datetimeInput.value).toISOString(),
            consumption: value,
            notes: notesInput?.value || ''
        };
        
        // Add to data
        this.formData[this.currentResource].push(entry);
        
        // Clear inputs
        valueInput.value = '';
        if (notesInput) notesInput.value = '';
        
        // Increment datetime by 1 hour
        const nextTime = new Date(datetimeInput.value);
        nextTime.setHours(nextTime.getHours() + 1);
        nextTime.setMinutes(nextTime.getMinutes() - nextTime.getTimezoneOffset());
        datetimeInput.value = nextTime.toISOString().slice(0, 16);
        
        // Update display
        this.renderEntries();
        this.saveData();
        
        // Show success
        this.showSuccess('Entry added successfully');
    },
    
    /**
     * Remove entry by ID
     */
    removeEntry(id) {
        this.formData[this.currentResource] = this.formData[this.currentResource]
            .filter(entry => entry.id !== id);
        
        this.renderEntries();
        this.saveData();
    },
    
    /**
     * Render entries table
     */
    renderEntries() {
        const container = document.getElementById('entries-table');
        if (!container) return;
        
        const entries = this.formData[this.currentResource];
        const unit = this.currentResource === 'electricity' ? 'kWh' : 'L';
        
        if (entries.length === 0) {
            container.innerHTML = `
                <div class="empty-state">
                    <i class="fas fa-database"></i>
                    <p>No entries yet. Add your consumption data above.</p>
                </div>`;
            return;
        }
        
        // Sort by datetime
        const sorted = [...entries].sort((a, b) => 
            new Date(a.datetime) - new Date(b.datetime)
        );
        
        container.innerHTML = `
            <table class="table">
                <thead>
                    <tr>
                        <th>Date/Time</th>
                        <th>Consumption</th>
                        <th>Notes</th>
                        <th>Actions</th>
                    </tr>
                </thead>
                <tbody>
                    ${sorted.map(entry => `
                        <tr>
                            <td>${this.formatDateTime(entry.datetime)}</td>
                            <td>${entry.consumption.toFixed(2)} ${unit}</td>
                            <td>${entry.notes || '-'}</td>
                            <td>
                                <button class="btn btn-sm btn-danger" 
                                        onclick="UserInput.removeEntry(${entry.id})">
                                    <i class="fas fa-trash"></i>
                                </button>
                            </td>
                        </tr>
                    `).join('')}
                </tbody>
            </table>
            <div class="entries-summary">
                <p>Total: ${entries.length} entries</p>
                <p>Sum: ${entries.reduce((sum, e) => sum + e.consumption, 0).toFixed(2)} ${unit}</p>
            </div>`;
    },
    
    /**
     * Handle CSV file upload
     */
    handleCSVUpload(event) {
        const file = event.target.files[0];
        if (!file) return;
        
        const reader = new FileReader();
        reader.onload = (e) => {
            try {
                const csv = e.target.result;
                const entries = this.parseCSV(csv);
                
                // Add entries to current resource
                this.formData[this.currentResource].push(...entries);
                
                this.renderEntries();
                this.saveData();
                this.showSuccess(`Imported ${entries.length} entries from CSV`);
                
            } catch (error) {
                this.showError(`CSV parsing error: ${error.message}`);
            }
        };
        reader.readAsText(file);
    },
    
    /**
     * Parse CSV content
     */
    parseCSV(csv) {
        const lines = csv.trim().split('\n');
        const entries = [];
        
        // Skip header if present
        const startIndex = lines[0].toLowerCase().includes('datetime') ? 1 : 0;
        
        for (let i = startIndex; i < lines.length; i++) {
            const parts = lines[i].split(',');
            
            if (parts.length >= 2) {
                const datetime = new Date(parts[0].trim());
                const consumption = parseFloat(parts[1].trim());
                
                if (!isNaN(datetime.getTime()) && !isNaN(consumption) && consumption >= 0) {
                    entries.push({
                        id: Date.now() + i,
                        datetime: datetime.toISOString(),
                        consumption: consumption,
                        notes: parts[2]?.trim() || ''
                    });
                }
            }
        }
        
        if (entries.length === 0) {
            throw new Error('No valid entries found in CSV');
        }
        
        return entries;
    },
    
    /**
     * Apply preset data
     */
    applyPreset(preset) {
        let entries = [];
        const now = new Date();
        
        switch (preset) {
            case 'sample-day':
                entries = this.generateSampleData(24, 'H');
                break;
            case 'sample-week':
                entries = this.generateSampleData(168, 'H');
                break;
            case 'sample-month':
                entries = this.generateSampleData(30, 'D');
                break;
        }
        
        this.formData[this.currentResource].push(...entries);
        this.renderEntries();
        this.saveData();
        this.showSuccess(`Added ${entries.length} sample entries`);
    },
    
    /**
     * Generate sample data
     */
    generateSampleData(count, frequency) {
        const entries = [];
        const now = new Date();
        const hoursIncrement = frequency === 'H' ? 1 : 24;
        
        for (let i = 0; i < count; i++) {
            const datetime = new Date(now - (count - i) * hoursIncrement * 3600000);
            const hour = datetime.getHours();
            
            let consumption;
            if (this.currentResource === 'electricity') {
                // Electricity pattern
                if (hour >= 6 && hour <= 9) consumption = 2 + Math.random();
                else if (hour >= 17 && hour <= 21) consumption = 2.5 + Math.random();
                else if (hour >= 0 && hour <= 5) consumption = 0.5 + Math.random() * 0.5;
                else consumption = 1 + Math.random();
            } else {
                // Water pattern
                if (hour >= 6 && hour <= 9) consumption = 40 + Math.random() * 20;
                else if (hour >= 18 && hour <= 22) consumption = 35 + Math.random() * 20;
                else if (hour >= 0 && hour <= 5) consumption = 5 + Math.random() * 5;
                else consumption = 15 + Math.random() * 10;
            }
            
            entries.push({
                id: Date.now() + i,
                datetime: datetime.toISOString(),
                consumption: parseFloat(consumption.toFixed(2)),
                notes: 'Sample data'
            });
        }
        
        return entries;
    },
    
    /**
     * Submit data for prediction
     */
    async submitData() {
        const entries = this.formData[this.currentResource];
        
        if (entries.length < 2) {
            this.showError('Please add at least 2 data points for prediction');
            return;
        }
        
        try {
            this.showLoading();
            
            // Format data for API
            const data = entries.map(e => ({
                datetime: e.datetime,
                consumption: e.consumption
            }));
            
            // Get prediction periods from settings
            const periodsInput = document.getElementById('prediction-periods');
            const periods = parseInt(periodsInput?.value) || 24;
            
            // Make prediction request
            let result;
            if (this.currentResource === 'electricity') {
                result = await API.predictions.electricity(data, periods);
            } else {
                result = await API.predictions.water(data, periods);
            }
            
            this.hideLoading();
            
            if (result.success) {
                // Store result and redirect to predictions page
                sessionStorage.setItem('predictionResult', JSON.stringify(result));
                sessionStorage.setItem('predictionResource', this.currentResource);
                window.location.href = '/predictions';
            } else {
                this.showError(result.error || 'Prediction failed');
            }
            
        } catch (error) {
            this.hideLoading();
            this.showError(error.message || 'Failed to generate predictions');
        }
    },
    
    /**
     * Clear all data
     */
    clearData() {
        if (!confirm('Are you sure you want to clear all entries?')) return;
        
        this.formData[this.currentResource] = [];
        this.renderEntries();
        this.saveData();
        this.showSuccess('All entries cleared');
    },
    
    /**
     * Save data to localStorage
     */
    saveData() {
        localStorage.setItem('echomind_formData', JSON.stringify(this.formData));
    },
    
    /**
     * Load saved data from localStorage
     */
    loadSavedData() {
        const saved = localStorage.getItem('echomind_formData');
        if (saved) {
            try {
                this.formData = JSON.parse(saved);
                this.renderEntries();
            } catch (e) {
                console.error('Error loading saved data:', e);
            }
        }
    },
    
    // ----- Helper Methods -----
    
    /**
     * Format datetime for display
     */
    formatDateTime(isoString) {
        const date = new Date(isoString);
        return date.toLocaleString();
    },
    
    /**
     * Show loading state
     */
    showLoading() {
        const loader = document.getElementById('submit-loader');
        const submitBtn = document.getElementById('submit-data');
        if (loader) loader.classList.remove('hidden');
        if (submitBtn) submitBtn.disabled = true;
    },
    
    /**
     * Hide loading state
     */
    hideLoading() {
        const loader = document.getElementById('submit-loader');
        const submitBtn = document.getElementById('submit-data');
        if (loader) loader.classList.add('hidden');
        if (submitBtn) submitBtn.disabled = false;
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
        
        // Auto-hide after 5 seconds
        setTimeout(() => {
            container.classList.add('hidden');
        }, 5000);
    }
};


// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    UserInput.init();
});


// Export for module use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = UserInput;
}

// Make available globally
window.UserInput = UserInput;