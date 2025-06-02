/**
 * Salesforce Voice Query Application - Main JavaScript
 */

// Application state
const App = {
    isListening: false,
    isProcessing: false,
    currentQuery: null,
    recognition: null,
    
    // DOM elements
    elements: {
        voiceBtn: null,
        voiceText: null,
        listeningIndicator: null,
        manualQuery: null,
        manualSubmit: null,
        processingSection: null,
        resultsSection: null,
        errorSection: null,
        exampleCards: null
    },
    
    init() {
        this.bindElements();
        this.bindEvents();
        this.initVoiceRecognition();
        this.showReady();
        console.log('Salesforce Voice Query App initialized');
    },
    
    bindElements() {
        this.elements.voiceBtn = document.getElementById('voiceBtn');
        this.elements.voiceText = document.getElementById('voice-text');
        this.elements.listeningIndicator = document.getElementById('listening-indicator');
        this.elements.manualQuery = document.getElementById('manualQuery');
        this.elements.manualSubmit = document.getElementById('manualSubmit');
        this.elements.processingSection = document.getElementById('processingSection');
        this.elements.resultsSection = document.getElementById('resultsSection');
        this.elements.errorSection = document.getElementById('errorSection');
        this.elements.exampleCards = document.querySelectorAll('.example-card');
    },
    
    bindEvents() {
        // Voice button click
        if (this.elements.voiceBtn) {
            this.elements.voiceBtn.addEventListener('click', () => this.toggleVoiceRecognition());
        }
        
        // Manual query submission
        if (this.elements.manualSubmit) {
            this.elements.manualSubmit.addEventListener('click', () => this.submitManualQuery());
        }
        
        if (this.elements.manualQuery) {
            this.elements.manualQuery.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    this.submitManualQuery();
                }
            });
        }
        
        // Example cards
        this.elements.exampleCards.forEach(card => {
            card.addEventListener('click', () => {
                const query = card.getAttribute('data-query');
                this.executeQuery(query);
            });
        });
        
        // Retry button
        document.addEventListener('click', (e) => {
            if (e.target.id === 'retryBtn') {
                this.hideAllSections();
                this.showReady();
            }
        });
    },
    
    initVoiceRecognition() {
        if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
            console.warn('Speech recognition not supported');
            this.showVoiceUnsupported();
            return;
        }
        
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        this.recognition = new SpeechRecognition();
        
        this.recognition.continuous = false;
        this.recognition.interimResults = true;
        this.recognition.lang = 'en-US';
        
        this.recognition.onstart = () => {
            console.log('Voice recognition started');
            this.showListening();
        };
        
        this.recognition.onresult = (event) => {
            let interimTranscript = '';
            let finalTranscript = '';
            
            for (let i = event.resultIndex; i < event.results.length; i++) {
                const transcript = event.results[i][0].transcript;
                if (event.results[i].isFinal) {
                    finalTranscript += transcript;
                } else {
                    interimTranscript += transcript;
                }
            }
            
            const displayText = finalTranscript || interimTranscript;
            this.updateVoiceText(displayText);
            
            if (finalTranscript) {
                console.log('Final transcript:', finalTranscript);
                this.executeQuery(finalTranscript.trim());
            }
        };
        
        this.recognition.onerror = (event) => {
            console.error('Speech recognition error:', event.error);
            this.showError(`Voice recognition error: ${event.error}`);
            this.showReady();
        };
        
        this.recognition.onend = () => {
            console.log('Voice recognition ended');
            this.isListening = false;
            if (!this.isProcessing) {
                this.showReady();
            }
        };
    },
    
    toggleVoiceRecognition() {
        if (this.isProcessing) {
            return;
        }
        
        if (this.isListening) {
            this.stopListening();
        } else {
            this.startListening();
        }
    },
    
    startListening() {
        if (!this.recognition) {
            this.showError('Voice recognition not available');
            return;
        }
        
        try {
            this.recognition.start();
            this.isListening = true;
        } catch (error) {
            console.error('Error starting recognition:', error);
            this.showError('Failed to start voice recognition');
        }
    },
    
    stopListening() {
        if (this.recognition && this.isListening) {
            this.recognition.stop();
            this.isListening = false;
        }
    },
    
    submitManualQuery() {
        const query = this.elements.manualQuery.value.trim();
        if (query) {
            this.executeQuery(query);
            this.elements.manualQuery.value = '';
        }
    },
    
    async executeQuery(query) {
        if (this.isProcessing || !query) {
            return;
        }
        
        console.log('Executing query:', query);
        this.currentQuery = query;
        this.isProcessing = true;
        this.stopListening();
        
        try {
            this.showProcessing(query);
            
            // Step 1: Convert to SOQL and execute in one call
            this.updateProcessingStep('step-convert', 'active');
            
            const response = await fetch('/api/voice-query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ query: query })
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.updateProcessingStep('step-convert', 'completed');
                this.updateProcessingStep('step-execute', 'completed');
                this.updateProcessingStep('step-format', 'active');
                
                // Brief delay for UX
                await this.delay(500);
                
                this.updateProcessingStep('step-format', 'completed');
                this.showResults(result);
            } else {
                this.showError(result.error, result);
            }
            
        } catch (error) {
            console.error('Query execution error:', error);
            this.showError(`Network error: ${error.message}`);
        } finally {
            this.isProcessing = false;
        }
    },
    
    showReady() {
        this.hideAllSections();
        this.updateVoiceButton('ready', 'Click to Speak');
        this.updateVoiceText('Your voice query will appear here...');
        this.elements.listeningIndicator.classList.remove('active');
    },
    
    showListening() {
        this.updateVoiceButton('listening', 'Listening...');
        this.updateVoiceText('Listening for your query...');
        this.elements.listeningIndicator.classList.add('active');
    },
    
    showProcessing(query) {
        this.hideAllSections();
        this.updateVoiceButton('processing', 'Processing...');
        this.updateVoiceText(`Processing: "${query}"`);
        this.elements.listeningIndicator.classList.remove('active');
        this.elements.processingSection.style.display = 'block';
        
        // Reset processing steps
        document.querySelectorAll('.step').forEach(step => {
            step.classList.remove('active', 'completed');
        });
    },
    
    showResults(data) {
        console.log('Showing results:', data);
        console.log('Records:', data.records);
        console.log('Total Size:', data.totalSize);
        
        this.hideAllSections();
        this.elements.resultsSection.style.display = 'block';
        
        // Update query information
        const naturalQueryEl = document.getElementById('naturalQuery');
        const generatedSOQLEl = document.getElementById('generatedSOQL');
        const recordCountEl = document.getElementById('recordCount');
        
        if (naturalQueryEl) {
            naturalQueryEl.textContent = data.natural_query;
            console.log('Updated natural query');
        } else {
            console.error('naturalQuery element not found');
        }
        
        if (generatedSOQLEl) {
            generatedSOQLEl.textContent = data.soql;
            console.log('Updated SOQL');
        } else {
            console.error('generatedSOQL element not found');
        }
        
        if (recordCountEl) {
            recordCountEl.textContent = data.totalSize;
            console.log('Updated record count');
        } else {
            console.error('recordCount element not found');
        }
        
        // Build results table
        console.log('Building results table...');
        this.buildResultsTable(data.records);
        
        // Update voice button to ready state but DON'T hide results
        this.updateVoiceButton('ready', 'Click to Speak');
        this.updateVoiceText('Results displayed below. Ask another question or click to speak again.');
        this.elements.listeningIndicator.classList.remove('active');
        
        console.log('Results displayed and ready for next query');
    },
    
    showError(message, details = null) {
        this.hideAllSections();
        this.elements.errorSection.style.display = 'block';
        
        document.getElementById('errorMessage').textContent = message;
        
        if (details) {
            const detailsElement = document.getElementById('errorDetails');
            const detailsContent = document.getElementById('errorDetailsContent');
            
            detailsContent.textContent = JSON.stringify(details, null, 2);
            detailsElement.style.display = 'block';
        }
        
        this.showReady();
    },
    
    showVoiceUnsupported() {
        this.updateVoiceButton('disabled', 'Voice Not Supported');
        this.elements.voiceBtn.disabled = true;
        this.elements.voiceBtn.style.opacity = '0.5';
        this.updateVoiceText('Voice recognition not supported in this browser. Please use manual input.');
    },
    
    hideAllSections() {
        this.elements.processingSection.style.display = 'none';
        this.elements.resultsSection.style.display = 'none';
        this.elements.errorSection.style.display = 'none';
    },
    
    updateVoiceButton(state, text) {
        const btn = this.elements.voiceBtn;
        const status = btn.querySelector('.voice-status');
        
        // Remove all state classes
        btn.classList.remove('listening', 'processing');
        
        // Add new state class
        if (state !== 'ready') {
            btn.classList.add(state);
        }
        
        // Update status text
        if (status) {
            status.textContent = text;
        }
    },
    
    updateVoiceText(text) {
        if (this.elements.voiceText) {
            this.elements.voiceText.textContent = text;
            
            // Add visual feedback for content
            if (text && text !== 'Your voice query will appear here...' && text !== 'Listening for your query...') {
                this.elements.voiceText.classList.add('has-content');
            } else {
                this.elements.voiceText.classList.remove('has-content');
            }
        }
    },
    
    updateProcessingStep(stepId, state) {
        const step = document.getElementById(stepId);
        if (step) {
            step.classList.remove('active', 'completed');
            if (state) {
                step.classList.add(state);
            }
        }
    },
    
    buildResultsTable(records) {
        console.log('Building table with records:', records);
        
        const tableHead = document.getElementById('resultsTableHead');
        const tableBody = document.getElementById('resultsTableBody');
        
        if (!tableHead) {
            console.error('resultsTableHead element not found');
            return;
        }
        
        if (!tableBody) {
            console.error('resultsTableBody element not found');
            return;
        }
        
        console.log('Found table elements');
        
        // Clear existing content
        tableHead.innerHTML = '';
        tableBody.innerHTML = '';
        
        if (!records || records.length === 0) {
            console.log('No records to display');
            tableBody.innerHTML = '<tr><td colspan="100%" style="text-align: center; color: #666;">No records found</td></tr>';
            return;
        }
        
        console.log(`Processing ${records.length} records`);
        
        // Get all unique columns from records
        const columns = new Set();
        records.forEach(record => {
            Object.keys(record).forEach(key => {
                if (key !== 'attributes') {
                    columns.add(key);
                }
            });
        });
        
        const columnArray = Array.from(columns).sort();
        console.log('Table columns:', columnArray);
        
        // Build table header
        const headerRow = document.createElement('tr');
        columnArray.forEach(column => {
            const th = document.createElement('th');
            th.textContent = this.formatColumnName(column);
            headerRow.appendChild(th);
        });
        tableHead.appendChild(headerRow);
        console.log('Created table header');
        
        // Build table body
        records.forEach((record, index) => {
            console.log(`Processing record ${index + 1}:`, record);
            const row = document.createElement('tr');
            
            columnArray.forEach(column => {
                const td = document.createElement('td');
                const value = this.getNestedValue(record, column);
                td.textContent = this.formatCellValue(value);
                row.appendChild(td);
            });
            
            tableBody.appendChild(row);
        });
        
        console.log('Table build complete');
    },
    
    formatColumnName(column) {
        // Convert camelCase and snake_case to Title Case
        return column
            .replace(/([A-Z])/g, ' $1')
            .replace(/_/g, ' ')
            .replace(/\b\w/g, l => l.toUpperCase())
            .trim();
    },
    
    getNestedValue(obj, path) {
        return path.split('.').reduce((current, key) => {
            return current && current[key] !== undefined ? current[key] : null;
        }, obj);
    },
    
    formatCellValue(value) {
        if (value === null || value === undefined) {
            return '';
        }
        
        if (typeof value === 'object') {
            return JSON.stringify(value);
        }
        
        // Format dates
        if (typeof value === 'string' && value.match(/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}/)) {
            try {
                const date = new Date(value);
                return date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
            } catch (e) {
                return value;
            }
        }
        
        // Format numbers with commas
        if (typeof value === 'number' && value > 999) {
            return value.toLocaleString();
        }
        
        return String(value);
    },
    
    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
};

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    App.init();
});

// Make App available globally for debugging
window.SalesforceVoiceApp = App; 