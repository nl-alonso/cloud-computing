# Claude-Salesforce MCP Voice Assistant

A powerful voice-powered web application that integrates **Claude AI** with **Salesforce CRM** using natural language processing to query Salesforce data through voice commands. Built with Flask, the application converts speech to SOQL queries using Claude's advanced language understanding capabilities.

## Demo

_Video demo: Upload 

https://github.com/user-attachments/assets/f92a2ab3-f5db-476e-bc07-351a7b3bb308

- **Salesforce Integration**: Full OAuth 2.0 authentication with Salesforce APIs
- **Real-time Results**: Instant query execution with formatted data display
- **Secure Architecture**: No hardcoded credentials, environment-based configuration
- **Sample Data Tools**: Scripts to populate Salesforce with realistic test data
- **Modern Web UI**: Clean, responsive interface with voice controls

## Quick Start

1. **Install dependencies**: `pip install -r requirements.txt`
2. **Configure environment**: `cp .env.example .env` and edit with your credentials
3. **Setup Salesforce Connected App** with callback URL: `http://localhost:3000/oauth/callback`
4. **Start application**: `python app.py`
5. **Access**: Open `http://localhost:3000`

## Voice Commands Examples

- "Show me all accounts"
- "Find opportunities over 100 thousand dollars"
- "Show me technology companies"
- "What deals are closing this month?"
- "Find contacts who are CEOs"

## Technologies

- **Backend**: Flask (Python)
- **AI**: Claude 4 Sonnet (Anthropic)
- **CRM**: Salesforce REST API v59.0
- **Frontend**: HTML5, JavaScript, Web Speech API
- **Authentication**: OAuth 2.0

## Sample Data

Import included CSV files using Salesforce Data Import Wizard:
1. `sample_data.csv` (50 accounts)
2. `sample_opportunities.csv` (50 opportunities)
3. `sample_contacts.csv` (52 contacts)

Or use automated scripts: `python create_quick_data.py`

## Security Features

- Environment variable configuration
- OAuth 2.0 authentication with Salesforce
- Session-based token management
- No hardcoded credentials

## Troubleshooting

- **Authentication issues**: Verify Salesforce Connected App settings
- **Empty results**: Import sample data using CSV files or scripts
- **Voice not working**: Enable microphone permissions in browser
- **API errors**: Check Claude API key and Salesforce credentials

## License

MIT License - Part of the cloud-computing repository.
