#!/usr/bin/env python3
"""
Simple script to create test data in Salesforce using REST API
Uses the same OAuth credentials as the Flask app
"""

import requests
import json
import random
from datetime import datetime, timedelta

# Same credentials as Flask app
CONSUMER_KEY = "<YOUR_SALESFORCE_CLIENT_ID>"
CONSUMER_SECRET = "<YOUR_SALESFORCE_CLIENT_SECRET>"
REDIRECT_URI = "http://localhost:3000/oauth/callback"

# Salesforce instance
INSTANCE_URL = "https://flow-inspiration-8638.my.salesforce.com"

def get_access_token():
    """Get access token using OAuth 2.0 Web Server Flow"""
    print("Getting access token...")
    
    # For server-to-server, we'll use the client credentials flow
    # But since we need user context, we'll need to get this manually
    print("This script needs an access token from your Flask app session.")
    print("Here's how to get it:")
    print("1. Go to http://localhost:3000 and log in to Salesforce")
    print("2. Open browser dev tools (F12)")
    print("3. Go to Application/Storage > Session Storage > localhost:3000")
    print("4. Copy the 'access_token' value")
    print("5. Paste it below:")
    
    access_token = input("\nPaste your access_token here: ").strip()
    
    if not access_token:
        print("No access token provided")
        return None
        
    print("Access token received")
    return access_token

def create_account(session, account_name):
    """Create a single account"""
    url = f"{INSTANCE_URL}/services/data/v58.0/sobjects/Account"
    
    account_data = {
        "Name": account_name,
        "Type": "Customer",
        "Industry": random.choice([
            "Technology", "Healthcare", "Financial Services", 
            "Manufacturing", "Retail", "Education", "Government"
        ]),
        "AnnualRevenue": random.randint(100000, 10000000),
        "NumberOfEmployees": random.randint(10, 5000),
        "Phone": f"({random.randint(100,999)}) {random.randint(100,999)}-{random.randint(1000,9999)}",
        "Website": f"https://www.{account_name.lower().replace(' ', '').replace(',', '')}.com"
    }
    
    response = session.post(url, json=account_data)
    
    if response.status_code == 201:
        account_id = response.json()['id']
        print(f"Created account: {account_name} (ID: {account_id})")
        return account_id
    else:
        print(f"Failed to create account {account_name}: {response.text}")
        return None

def create_opportunity(session, account_id, account_name):
    """Create an opportunity for an account"""
    url = f"{INSTANCE_URL}/services/data/v58.0/sobjects/Opportunity"
    
    stages = [
        "Prospecting", "Qualification", "Needs Analysis", 
        "Value Proposition", "Proposal/Price Quote", "Negotiation/Review", "Closed Won"
    ]
    
    close_date = datetime.now() + timedelta(days=random.randint(30, 365))
    
    opportunity_data = {
        "Name": f"{account_name} - {random.choice(['Software License', 'Consulting Services', 'Support Contract', 'Implementation', 'Training'])}",
        "AccountId": account_id,
        "StageName": random.choice(stages),
        "Amount": random.randint(10000, 500000),
        "CloseDate": close_date.strftime("%Y-%m-%d"),
        "Probability": random.randint(10, 90),
        "Type": random.choice(["New Customer", "Existing Customer - Upgrade", "Existing Customer - Replacement", "Existing Customer - Downgrade"])
    }
    
    response = session.post(url, json=opportunity_data)
    
    if response.status_code == 201:
        opp_id = response.json()['id']
        print(f"Created opportunity: {opportunity_data['Name']} (ID: {opp_id})")
        return opp_id
    else:
        print(f"Failed to create opportunity for {account_name}: {response.text}")
        return None

def create_contact(session, account_id, account_name):
    """Create a contact for an account"""
    url = f"{INSTANCE_URL}/services/data/v58.0/sobjects/Contact"
    
    first_names = ["John", "Jane", "Michael", "Sarah", "David", "Lisa", "Robert", "Emily", "James", "Maria"]
    last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez"]
    
    first_name = random.choice(first_names)
    last_name = random.choice(last_names)
    
    contact_data = {
        "FirstName": first_name,
        "LastName": last_name,
        "AccountId": account_id,
        "Email": f"{first_name.lower()}.{last_name.lower()}@{account_name.lower().replace(' ', '').replace(',', '')}.com",
        "Phone": f"({random.randint(100,999)}) {random.randint(100,999)}-{random.randint(1000,9999)}",
        "Title": random.choice([
            "CEO", "CTO", "VP Sales", "Director of IT", "Project Manager", 
            "Senior Developer", "Business Analyst", "Account Manager"
        ])
    }
    
    response = session.post(url, json=contact_data)
    
    if response.status_code == 201:
        contact_id = response.json()['id']
        print(f"Created contact: {first_name} {last_name} (ID: {contact_id})")
        return contact_id
    else:
        print(f"Failed to create contact for {account_name}: {response.text}")
        return None

def main():
    print("Salesforce Test Data Creator")
    print("=" * 50)
    
    # Get access token
    access_token = get_access_token()
    if not access_token:
        return
    
    # Setup session
    session = requests.Session()
    session.headers.update({
        'Authorization': f'Bearer {access_token}',
        'Content-Type': 'application/json'
    })
    
    # Test connection
    print("\nTesting Salesforce connection...")
    test_url = f"{INSTANCE_URL}/services/data/v58.0/sobjects"
    response = session.get(test_url)
    
    if response.status_code != 200:
        print(f"Failed to connect to Salesforce: {response.text}")
        return
    
    print("Connected to Salesforce successfully!")
    
    # Ask how many records to create
    try:
        count = int(input("\nHow many accounts (with opportunities and contacts) to create? [10]: ") or "10")
    except ValueError:
        count = 10
    
    print(f"\nCreating {count} accounts with opportunities and contacts...")
    
    # Company names for test data
    company_names = [
        "Acme Corporation", "Tech Solutions Inc", "Global Systems Ltd", "Digital Innovations",
        "Future Technologies", "Smart Solutions", "Advanced Systems", "Premier Services",
        "Elite Consulting", "Dynamic Solutions", "Innovative Tech", "Strategic Systems",
        "NextGen Solutions", "Quantum Technologies", "Alpha Systems", "Beta Corporation",
        "Gamma Enterprises", "Delta Solutions", "Epsilon Technologies", "Zeta Systems",
        "Omega Corporation", "Prime Technologies", "Nova Solutions", "Stellar Systems",
        "Cosmic Enterprises", "Infinite Solutions", "Ultimate Technologies", "Supreme Systems",
        "Pinnacle Corporation", "Apex Solutions", "Vertex Technologies", "Summit Systems"
    ]
    
    created_accounts = 0
    created_opportunities = 0
    created_contacts = 0
    
    for i in range(count):
        # Pick a random company name or generate one
        if i < len(company_names):
            company_name = company_names[i]
        else:
            company_name = f"Company {i + 1} Solutions"
        
        print(f"\nCreating record set {i + 1}/{count}: {company_name}")
        
        # Create account
        account_id = create_account(session, company_name)
        if account_id:
            created_accounts += 1
            
            # Create 1-2 opportunities per account
            num_opps = random.randint(1, 2)
            for j in range(num_opps):
                opp_id = create_opportunity(session, account_id, company_name)
                if opp_id:
                    created_opportunities += 1
            
            # Create 1-3 contacts per account
            num_contacts = random.randint(1, 3)
            for j in range(num_contacts):
                contact_id = create_contact(session, account_id, company_name)
                if contact_id:
                    created_contacts += 1
    
    print("\n" + "=" * 50)
    print("🎉 Test Data Creation Complete!")
    print(f"Created {created_accounts} accounts")
    print(f"Created {created_opportunities} opportunities") 
    print(f"Created {created_contacts} contacts")
    print("\nNow you can test voice queries like:")
    print("   - 'Show me all accounts'")
    print("   - 'Find opportunities over $100,000'")
    print("   - 'Show me contacts at tech companies'")
    print("   - 'What are my biggest deals?'")
    print("\nGo to http://localhost:3000 to try voice queries!")

if __name__ == "__main__":
    main() 