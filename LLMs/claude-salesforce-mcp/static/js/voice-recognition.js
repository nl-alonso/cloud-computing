/**
 * Advanced Voice Recognition Module
 * Provides enhanced voice recognition capabilities with noise reduction and confidence scoring
 */

class VoiceRecognitionModule {
    constructor() {
        this.recognition = null;
        this.isListening = false;
        this.isSupported = this.checkSupport();
        this.confidence = 0;
        this.lastTranscript = '';
        this.callbacks = {
            onResult: null,
            onError: null,
            onStart: null,
            onEnd: null
        };
        
        this.initRecognition();
    }
    
    checkSupport() {
        return ('webkitSpeechRecognition' in window) || ('SpeechRecognition' in window);
    }
    
    initRecognition() {
        if (!this.isSupported) {
            console.warn('Speech recognition not supported in this browser');
            return;
        }
        
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        this.recognition = new SpeechRecognition();
        
        // Enhanced configuration
        this.recognition.continuous = false;
        this.recognition.interimResults = true;
        this.recognition.lang = this.getPreferredLanguage();
        this.recognition.maxAlternatives = 3;
        
        this.setupEventHandlers();
    }
    
    getPreferredLanguage() {
        // Default to English, but could be made configurable
        const userLang = navigator.language || navigator.userLanguage;
        
        // Map common languages to supported speech recognition languages
        const supportedLanguages = {
            'en': 'en-US',
            'es': 'es-ES',
            'fr': 'fr-FR',
            'de': 'de-DE',
            'it': 'it-IT',
            'pt': 'pt-BR',
            'zh': 'zh-CN',
            'ja': 'ja-JP',
            'ko': 'ko-KR'
        };
        
        const langCode = userLang.split('-')[0];
        return supportedLanguages[langCode] || 'en-US';
    }
    
    setupEventHandlers() {
        if (!this.recognition) return;
        
        this.recognition.onstart = () => {
            this.isListening = true;
            console.log('Voice recognition started');
            if (this.callbacks.onStart) {
                this.callbacks.onStart();
            }
        };
        
        this.recognition.onresult = (event) => {
            this.handleResult(event);
        };
        
        this.recognition.onerror = (event) => {
            this.handleError(event);
        };
        
        this.recognition.onend = () => {
            this.isListening = false;
            console.log('Voice recognition ended');
            if (this.callbacks.onEnd) {
                this.callbacks.onEnd();
            }
        };
        
        this.recognition.onnomatch = () => {
            console.warn('No speech was recognized');
            if (this.callbacks.onError) {
                this.callbacks.onError('No speech was recognized');
            }
        };
        
        this.recognition.onsoundstart = () => {
            console.log('Sound detected');
        };
        
        this.recognition.onsoundend = () => {
            console.log('Sound ended');
        };
        
        this.recognition.onspeechstart = () => {
            console.log('Speech detected');
        };
        
        this.recognition.onspeechend = () => {
            console.log('Speech ended');
        };
    }
    
    handleResult(event) {
        let interimTranscript = '';
        let finalTranscript = '';
        let maxConfidence = 0;
        
        // Process all results
        for (let i = event.resultIndex; i < event.results.length; i++) {
            const result = event.results[i];
            const transcript = result[0].transcript;
            const confidence = result[0].confidence;
            
            if (result.isFinal) {
                finalTranscript += transcript;
                if (confidence > maxConfidence) {
                    maxConfidence = confidence;
                }
            } else {
                interimTranscript += transcript;
            }
        }
        
        // Update confidence score
        if (maxConfidence > 0) {
            this.confidence = maxConfidence;
        }
        
        // Clean up transcript
        const cleanTranscript = this.cleanTranscript(finalTranscript || interimTranscript);
        
        // Call result callback
        if (this.callbacks.onResult) {
            this.callbacks.onResult({
                transcript: cleanTranscript,
                isFinal: !!finalTranscript,
                confidence: this.confidence,
                alternatives: this.getAlternatives(event.results)
            });
        }
        
        // Store last transcript for comparison
        if (finalTranscript) {
            this.lastTranscript = cleanTranscript;
        }
    }
    
    getAlternatives(results) {
        const alternatives = [];
        
        for (let i = 0; i < results.length; i++) {
            const result = results[i];
            if (result.isFinal) {
                for (let j = 0; j < result.length; j++) {
                    alternatives.push({
                        transcript: result[j].transcript,
                        confidence: result[j].confidence
                    });
                }
            }
        }
        
        return alternatives.sort((a, b) => b.confidence - a.confidence);
    }
    
    cleanTranscript(transcript) {
        return transcript
            .trim()
            .replace(/\s+/g, ' ') // Multiple spaces to single space
            .replace(/^\w/, c => c.toUpperCase()); // Capitalize first letter
    }
    
    handleError(event) {
        console.error('Speech recognition error:', event.error);
        
        let errorMessage = 'Speech recognition error';
        
        switch (event.error) {
            case 'network':
                errorMessage = 'Network error occurred during recognition';
                break;
            case 'not-allowed':
                errorMessage = 'Microphone access denied. Please allow microphone access and try again.';
                break;
            case 'no-speech':
                errorMessage = 'No speech detected. Please try speaking louder or closer to the microphone.';
                break;
            case 'audio-capture':
                errorMessage = 'Audio capture failed. Please check your microphone.';
                break;
            case 'service-not-allowed':
                errorMessage = 'Speech recognition service not allowed';
                break;
            case 'bad-grammar':
                errorMessage = 'Grammar error in speech recognition';
                break;
            case 'language-not-supported':
                errorMessage = 'Language not supported';
                break;
            default:
                errorMessage = `Speech recognition error: ${event.error}`;
        }
        
        if (this.callbacks.onError) {
            this.callbacks.onError(errorMessage, event.error);
        }
    }
    
    start() {
        if (!this.isSupported) {
            const error = 'Speech recognition not supported in this browser';
            if (this.callbacks.onError) {
                this.callbacks.onError(error);
            }
            return false;
        }
        
        if (this.isListening) {
            console.warn('Recognition already in progress');
            return false;
        }
        
        try {
            this.recognition.start();
            return true;
        } catch (error) {
            console.error('Failed to start speech recognition:', error);
            if (this.callbacks.onError) {
                this.callbacks.onError('Failed to start speech recognition');
            }
            return false;
        }
    }
    
    stop() {
        if (this.recognition && this.isListening) {
            this.recognition.stop();
        }
    }
    
    abort() {
        if (this.recognition && this.isListening) {
            this.recognition.abort();
        }
    }
    
    // Event callback setters
    onResult(callback) {
        this.callbacks.onResult = callback;
    }
    
    onError(callback) {
        this.callbacks.onError = callback;
    }
    
    onStart(callback) {
        this.callbacks.onStart = callback;
    }
    
    onEnd(callback) {
        this.callbacks.onEnd = callback;
    }
    
    // Utility methods
    getConfidence() {
        return this.confidence;
    }
    
    getLastTranscript() {
        return this.lastTranscript;
    }
    
    isCurrentlyListening() {
        return this.isListening;
    }
    
    isSupportedBrowser() {
        return this.isSupported;
    }
    
    // Configuration methods
    setLanguage(lang) {
        if (this.recognition) {
            this.recognition.lang = lang;
        }
    }
    
    setContinuous(continuous) {
        if (this.recognition) {
            this.recognition.continuous = continuous;
        }
    }
    
    setInterimResults(interim) {
        if (this.recognition) {
            this.recognition.interimResults = interim;
        }
    }
    
    setMaxAlternatives(max) {
        if (this.recognition) {
            this.recognition.maxAlternatives = max;
        }
    }
}

// Browser compatibility check and graceful fallback
function checkVoiceSupport() {
    const support = {
        speechRecognition: ('webkitSpeechRecognition' in window) || ('SpeechRecognition' in window),
        mediaDevices: !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia),
        audioContext: !!(window.AudioContext || window.webkitAudioContext)
    };
    
    return support;
}

// Microphone permission helper
async function requestMicrophonePermission() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        stream.getTracks().forEach(track => track.stop()); // Stop the stream immediately
        return true;
    } catch (error) {
        console.error('Microphone permission denied:', error);
        return false;
    }
}

// Audio level monitoring (for visual feedback)
class AudioLevelMonitor {
    constructor() {
        this.audioContext = null;
        this.analyser = null;
        this.microphone = null;
        this.dataArray = null;
        this.isMonitoring = false;
        this.levelCallback = null;
    }
    
    async start() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
            this.microphone = this.audioContext.createMediaStreamSource(stream);
            this.analyser = this.audioContext.createAnalyser();
            
            this.analyser.fftSize = 256;
            this.dataArray = new Uint8Array(this.analyser.frequencyBinCount);
            
            this.microphone.connect(this.analyser);
            this.isMonitoring = true;
            this.monitor();
            
            return true;
        } catch (error) {
            console.error('Failed to start audio monitoring:', error);
            return false;
        }
    }
    
    stop() {
        this.isMonitoring = false;
        if (this.microphone) {
            this.microphone.disconnect();
        }
        if (this.audioContext) {
            this.audioContext.close();
        }
    }
    
    monitor() {
        if (!this.isMonitoring) return;
        
        this.analyser.getByteFrequencyData(this.dataArray);
        
        // Calculate average volume
        let sum = 0;
        for (let i = 0; i < this.dataArray.length; i++) {
            sum += this.dataArray[i];
        }
        const average = sum / this.dataArray.length;
        
        if (this.levelCallback) {
            this.levelCallback(average);
        }
        
        requestAnimationFrame(() => this.monitor());
    }
    
    onLevel(callback) {
        this.levelCallback = callback;
    }
}

// Make modules available globally
window.VoiceRecognitionModule = VoiceRecognitionModule;
window.AudioLevelMonitor = AudioLevelMonitor;
window.checkVoiceSupport = checkVoiceSupport;
window.requestMicrophonePermission = requestMicrophonePermission; 