#!/usr/bin/env python3
"""
Salesforce Voice Query Web Application
Server-side solution with proper OAuth and API access
"""

# Load environment variables first
from dotenv import load_dotenv
load_dotenv()

from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import requests
import json
import os
from datetime import datetime, timedelta
import base64
from urllib.parse import quote
import anthropic
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'your-secret-key-change-in-production')

# Salesforce OAuth Configuration
SALESFORCE_CONFIG = {
    'client_id': os.environ.get('SALESFORCE_CLIENT_ID', ''),
    'client_secret': os.environ.get('SALESFORCE_CLIENT_SECRET', ''),
    'redirect_uri': os.environ.get('SALESFORCE_REDIRECT_URI', 'http://localhost:5000/oauth/callback'),
    'login_url': os.environ.get('SALESFORCE_LOGIN_URL', 'https://login.salesforce.com'),
    'sandbox_url': os.environ.get('SALESFORCE_SANDBOX_URL', 'https://test.salesforce.com')
}

# Claude API Configuration  
CLAUDE_API_KEY = os.environ.get('CLAUDE_API_KEY', '')

class SalesforceAPI:
    def __init__(self, access_token, instance_url):
        self.access_token = access_token
        self.instance_url = instance_url
        self.headers = {
            'Authorization': f'Bearer {access_token}',
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
    
    def query(self, soql):
        """Execute SOQL query against Salesforce API"""
        try:
            url = f"{self.instance_url}/services/data/v59.0/query/"
            params = {'q': soql}
            
            logger.info(f"Executing SOQL: {soql}")
            response = requests.get(url, headers=self.headers, params=params)
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"Query successful: {data.get('totalSize', 0)} records")
                return {
                    'success': True,
                    'records': data.get('records', []),
                    'totalSize': data.get('totalSize', 0),
                    'done': data.get('done', True)
                }
            else:
                logger.error(f"Query failed: {response.status_code} - {response.text}")
                return {
                    'success': False,
                    'error': f"Salesforce API error: {response.status_code}",
                    'details': response.text
                }
                
        except Exception as e:
            logger.error(f"Query exception: {str(e)}")
            return {
                'success': False,
                'error': f"Exception: {str(e)}"
            }
    
    def get_org_info(self):
        """Get organization information"""
        try:
            url = f"{self.instance_url}/services/data/v59.0/sobjects/"
            response = requests.get(url, headers=self.headers)
            
            if response.status_code == 200:
                return response.json()
            else:
                return None
        except:
            return None

class ClaudeSOQLConverter:
    def __init__(self, api_key):
        self.client = anthropic.Anthropic(api_key=api_key)
    
    def convert_to_soql(self, natural_language_query):
        """Convert natural language to SOQL using Claude"""
        try:
            prompt = f"""
You are a Salesforce SOQL expert. Convert this natural language query to SOQL:

"{natural_language_query}"

Rules:
1. Use standard Salesforce objects (Account, Contact, Opportunity, Lead, Case, etc.)
2. Include commonly needed fields (Id, Name, etc.)
3. Add relationship fields when appropriate (Account.Name for Opportunities)
4. Use proper SOQL syntax
5. Limit results to 50 records max
6. For Opportunities, include: Id, Name, Amount, StageName, CloseDate, Account.Name, AccountId
7. For Accounts, include: Id, Name, Industry, Type, AnnualRevenue
8. For Contacts, include: Id, Name, Email, Phone, Account.Name, AccountId
9. For Leads, include: Id, Name, Company, Status, LeadSource

Return only the SOQL query, no explanations.
"""

            message = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=500,
                messages=[{
                    "role": "user", 
                    "content": prompt
                }]
            )
            
            soql_query = message.content[0].text.strip()
            
            # Clean up the response
            if soql_query.startswith('```'):
                soql_query = soql_query.split('\n')[1:-1]
                soql_query = '\n'.join(soql_query)
            
            logger.info(f"Generated SOQL: {soql_query}")
            return soql_query
            
        except Exception as e:
            logger.error(f"Claude API error: {str(e)}")
            return None

@app.route('/')
def index():
    """Main application page"""
    if 'salesforce_token' not in session:
        return render_template('login.html', salesforce_config=SALESFORCE_CONFIG)
    
    return render_template('voice_query.html')

@app.route('/oauth/login')
def oauth_login():
    """Redirect to Salesforce OAuth"""
    auth_url = f"{SALESFORCE_CONFIG['login_url']}/services/oauth2/authorize"
    params = {
        'response_type': 'code',
        'client_id': SALESFORCE_CONFIG['client_id'],
        'redirect_uri': SALESFORCE_CONFIG['redirect_uri'],
        'scope': 'full refresh_token api',
        'state': 'security_token_here'
    }
    
    query_string = '&'.join([f"{k}={quote(str(v))}" for k, v in params.items()])
    oauth_url = f"{auth_url}?{query_string}"
    
    return redirect(oauth_url)

@app.route('/oauth/callback')
def oauth_callback():
    """Handle OAuth callback from Salesforce"""
    code = request.args.get('code')
    
    if not code:
        return "OAuth failed: No authorization code received", 400
    
    # Exchange code for access token
    token_url = f"{SALESFORCE_CONFIG['login_url']}/services/oauth2/token"
    token_data = {
        'grant_type': 'authorization_code',
        'code': code,
        'client_id': SALESFORCE_CONFIG['client_id'],
        'client_secret': SALESFORCE_CONFIG['client_secret'],
        'redirect_uri': SALESFORCE_CONFIG['redirect_uri']
    }
    
    try:
        response = requests.post(token_url, data=token_data)
        
        if response.status_code == 200:
            token_info = response.json()
            
            # Store in session
            session['salesforce_token'] = token_info['access_token']
            session['instance_url'] = token_info['instance_url']
            session['refresh_token'] = token_info.get('refresh_token')
            
            logger.info("OAuth successful, tokens stored in session")
            return redirect(url_for('index'))
        else:
            logger.error(f"Token exchange failed: {response.text}")
            return f"OAuth failed: {response.text}", 400
            
    except Exception as e:
        logger.error(f"OAuth callback error: {str(e)}")
        return f"OAuth error: {str(e)}", 500

@app.route('/api/convert-query', methods=['POST'])
def convert_query():
    """Convert natural language to SOQL"""
    try:
        data = request.json
        natural_query = data.get('query', '')
        
        if not natural_query:
            return jsonify({'success': False, 'error': 'No query provided'})
        
        if not CLAUDE_API_KEY:
            return jsonify({'success': False, 'error': 'Claude API key not configured'})
        
        converter = ClaudeSOQLConverter(CLAUDE_API_KEY)
        soql_query = converter.convert_to_soql(natural_query)
        
        if soql_query:
            return jsonify({
                'success': True,
                'soql': soql_query,
                'natural_query': natural_query
            })
        else:
            return jsonify({'success': False, 'error': 'Failed to generate SOQL'})
            
    except Exception as e:
        logger.error(f"Convert query error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/execute-query', methods=['POST'])
def execute_query():
    """Execute SOQL query against Salesforce"""
    try:
        if 'salesforce_token' not in session:
            return jsonify({'success': False, 'error': 'Not authenticated with Salesforce'})
        
        data = request.json
        soql_query = data.get('soql', '')
        
        if not soql_query:
            return jsonify({'success': False, 'error': 'No SOQL query provided'})
        
        # Create Salesforce API client
        sf_api = SalesforceAPI(
            access_token=session['salesforce_token'],
            instance_url=session['instance_url']
        )
        
        # Execute query
        result = sf_api.query(soql_query)
        
        if result['success']:
            return jsonify({
                'success': True,
                'records': result['records'],
                'totalSize': result['totalSize'],
                'soql': soql_query
            })
        else:
            return jsonify({
                'success': False,
                'error': result['error'],
                'details': result.get('details', '')
            })
            
    except Exception as e:
        logger.error(f"Execute query error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/voice-query', methods=['POST'])
def voice_query():
    """Complete voice-to-data pipeline"""
    try:
        if 'salesforce_token' not in session:
            return jsonify({'success': False, 'error': 'Not authenticated with Salesforce'})
        
        data = request.json
        natural_query = data.get('query', '')
        
        if not natural_query:
            return jsonify({'success': False, 'error': 'No query provided'})
        
        # Step 1: Convert to SOQL
        converter = ClaudeSOQLConverter(CLAUDE_API_KEY)
        soql_query = converter.convert_to_soql(natural_query)
        
        if not soql_query:
            return jsonify({'success': False, 'error': 'Failed to generate SOQL'})
        
        # Step 2: Execute against Salesforce
        sf_api = SalesforceAPI(
            access_token=session['salesforce_token'],
            instance_url=session['instance_url']
        )
        
        result = sf_api.query(soql_query)
        
        if result['success']:
            return jsonify({
                'success': True,
                'natural_query': natural_query,
                'soql': soql_query,
                'records': result['records'],
                'totalSize': result['totalSize'],
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({
                'success': False,
                'error': result['error'],
                'soql': soql_query,
                'natural_query': natural_query
            })
            
    except Exception as e:
        logger.error(f"Voice query error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/logout')
def logout():
    """Clear session and logout"""
    session.clear()
    return redirect(url_for('index'))

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'claude_configured': bool(CLAUDE_API_KEY),
        'salesforce_configured': bool(SALESFORCE_CONFIG['client_id'])
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=3000) 