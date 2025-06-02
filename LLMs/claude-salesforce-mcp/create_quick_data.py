#!/usr/bin/env python3
"""
Quick Test Data Creator - Creates 100-200 records automatically
Uses OAuth Username-Password flow for simplicity
"""

import requests
import json
import random
from datetime import datetime, timedelta
import sys

# Salesforce credentials (same as your Flask app)
CONSUMER_KEY = "<YOUR_SALESFORCE_CLIENT_ID>"
CONSUMER_SECRET = "<YOUR_SALESFORCE_CLIENT_SECRET>"

def get_access_token_simple():
    """Get access token using a simpler approach"""
    print("You'll need to provide your Salesforce username and password (or security token)")
    print("This is safe - it only gets an access token for data creation.")
    
    username = input("Salesforce Username: ").strip()
    password = input("Salesforce Password: ").strip()
    
    # For sandbox orgs, you might need a security token
    security_token = input("Security Token (if required, otherwise press Enter): ").strip()
    
    if security_token:
        password = password + security_token
    
    # OAuth Token endpoint
    token_url = "https://login.salesforce.com/services/oauth2/token"
    
    data = {
        'grant_type': 'password',
        'client_id': CONSUMER_KEY,
        'client_secret': CONSUMER_SECRET,
        'username': username,
        'password': password
    }
    
    try:
        response = requests.post(token_url, data=data)
        
        if response.status_code == 200:
            token_data = response.json()
            print("Successfully got access token!")
            return token_data['access_token'], token_data['instance_url']
        else:
            print(f"Failed to get token: {response.text}")
            return None, None
            
    except Exception as e:
        print(f"Error getting token: {str(e)}")
        return None, None

def create_bulk_accounts(session, instance_url, count=50):
    """Create multiple accounts efficiently"""
    print(f"Creating {count} accounts...")
    
    companies = [
        "TechCorp Solutions", "Global Dynamics", "Future Systems", "Digital Edge",
        "Smart Analytics", "Cloud Nine Technologies", "Data Driven Inc", "AI Innovations",
        "Quantum Computing Co", "Cyber Security Pro", "Mobile First Labs", "IoT Solutions",
        "Blockchain Ventures", "Machine Learning Corp", "Virtual Reality Co", "Augmented Systems",
        "Big Data Analytics", "Cloud Computing Inc", "Software Engineering Ltd", "Web Development Co",
        "Mobile App Studio", "Enterprise Solutions", "Business Intelligence", "Data Warehousing",
        "Network Security", "Information Systems", "Technology Partners", "Digital Transformation",
        "Innovation Hub", "Strategic Tech", "Advanced Computing", "Modern Solutions",
        "Tech Innovations", "Digital Solutions", "Smart Technology", "Future Computing",
        "Intelligent Systems", "Automated Solutions", "Connected Devices", "Platform Solutions",
        "Enterprise Tech", "Digital Platform", "Cloud Services", "Data Solutions",
        "Innovation Labs", "Technology Group", "Digital Ventures", "Smart Systems",
        "Future Technologies", "Advanced Analytics", "Intelligent Computing", "Digital Innovation"
    ]
    
    industries = ["Technology", "Healthcare", "Financial Services", "Manufacturing", 
                 "Retail", "Education", "Government", "Media", "Energy", "Transportation"]
    
    created_accounts = []
    
    for i in range(count):
        company_name = companies[i % len(companies)]
        if i >= len(companies):
            company_name = f"{company_name} {i // len(companies) + 1}"
        
        account_data = {
            "Name": company_name,
            "Type": random.choice(["Customer", "Partner", "Prospect"]),
            "Industry": random.choice(industries),
            "AnnualRevenue": random.randint(100000, 50000000),
            "NumberOfEmployees": random.randint(10, 10000),
            "Phone": f"({random.randint(100,999)}) {random.randint(100,999)}-{random.randint(1000,9999)}",
            "BillingCity": random.choice(["New York", "San Francisco", "Chicago", "Boston", "Austin", "Seattle", "Miami", "Denver"])
        }
        
        url = f"{instance_url}/services/data/v59.0/sobjects/Account"
        response = session.post(url, json=account_data)
        
        if response.status_code == 201:
            account_id = response.json()['id']
            created_accounts.append((account_id, company_name))
            print(f"{i+1}/{count}: Created {company_name}")
        else:
            print(f"Failed to create {company_name}: {response.status_code}")
    
    return created_accounts

def create_bulk_opportunities(session, instance_url, accounts, opp_count=100):
    """Create opportunities for the accounts"""
    print(f"💰 Creating {opp_count} opportunities...")
    
    stages = ["Prospecting", "Qualification", "Needs Analysis", "Value Proposition", 
             "Proposal/Price Quote", "Negotiation/Review", "Closed Won", "Closed Lost"]
    
    opp_types = ["New Customer", "Existing Customer - Upgrade", "Existing Customer - Replacement"]
    
    products = ["Software License", "Consulting Services", "Support Contract", "Training",
               "Implementation", "Integration", "Maintenance", "Custom Development"]
    
    created_opps = 0
    
    for i in range(opp_count):
        if not accounts:
            break
            
        account_id, account_name = random.choice(accounts)
        
        # Random close date between now and 1 year
        days_out = random.randint(0, 365)
        close_date = datetime.now() + timedelta(days=days_out)
        
        opportunity_data = {
            "Name": f"{account_name} - {random.choice(products)}",
            "AccountId": account_id,
            "StageName": random.choice(stages),
            "Amount": random.randint(5000, 1000000),
            "CloseDate": close_date.strftime("%Y-%m-%d"),
            "Probability": random.randint(10, 95),
            "Type": random.choice(opp_types)
        }
        
        url = f"{instance_url}/services/data/v59.0/sobjects/Opportunity"
        response = session.post(url, json=opportunity_data)
        
        if response.status_code == 201:
            created_opps += 1
            print(f"{created_opps}/{opp_count}: Created opportunity for {account_name}")
        else:
            print(f"Failed to create opportunity for {account_name}")
    
    return created_opps

def create_bulk_contacts(session, instance_url, accounts, contact_count=80):
    """Create contacts for the accounts"""
    print(f"👥 Creating {contact_count} contacts...")
    
    first_names = ["John", "Jane", "Michael", "Sarah", "David", "Lisa", "Robert", "Emily", 
                  "James", "Maria", "William", "Jennifer", "Richard", "Linda", "Thomas", "Patricia"]
    
    last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
                 "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson", "Taylor"]
    
    titles = ["CEO", "CTO", "VP Sales", "VP Marketing", "Director of IT", "Project Manager",
             "Senior Developer", "Business Analyst", "Account Manager", "Sales Director"]
    
    created_contacts = 0
    
    for i in range(contact_count):
        if not accounts:
            break
            
        account_id, account_name = random.choice(accounts)
        
        first_name = random.choice(first_names)
        last_name = random.choice(last_names)
        
        # Create a simple domain from company name
        domain = account_name.lower().replace(" ", "").replace(",", "")[:15] + ".com"
        
        contact_data = {
            "FirstName": first_name,
            "LastName": last_name,
            "AccountId": account_id,
            "Email": f"{first_name.lower()}.{last_name.lower()}@{domain}",
            "Phone": f"({random.randint(100,999)}) {random.randint(100,999)}-{random.randint(1000,9999)}",
            "Title": random.choice(titles)
        }
        
        url = f"{instance_url}/services/data/v59.0/sobjects/Contact"
        response = session.post(url, json=contact_data)
        
        if response.status_code == 201:
            created_contacts += 1
            print(f"{created_contacts}/{contact_count}: Created {first_name} {last_name}")
        else:
            print(f"Failed to create contact {first_name} {last_name}")
    
    return created_contacts

def main():
    print("Quick Salesforce Data Creator")
    print("=" * 50)
    print("This will create approximately 200+ records total:")
    print("• 50 Accounts")
    print("• 100 Opportunities") 
    print("• 80 Contacts")
    print("=" * 50)
    
    # Get access token
    access_token, instance_url = get_access_token_simple()
    if not access_token:
        print("Could not get access token. Exiting.")
        return
    
    # Setup session
    session = requests.Session()
    session.headers.update({
        'Authorization': f'Bearer {access_token}',
        'Content-Type': 'application/json'
    })
    
    # Test connection
    print("\nTesting Salesforce connection...")
    test_url = f"{instance_url}/services/data/v59.0/sobjects"
    response = session.get(test_url)
    
    if response.status_code != 200:
        print(f"Failed to connect to Salesforce: {response.text}")
        return
    
    print("Connected to Salesforce successfully!")
    
    try:
        # Create accounts first
        accounts = create_bulk_accounts(session, instance_url, 50)
        
        if not accounts:
            print("No accounts created. Cannot proceed.")
            return
        
        # Create opportunities
        opp_count = create_bulk_opportunities(session, instance_url, accounts, 100)
        
        # Create contacts
        contact_count = create_bulk_contacts(session, instance_url, accounts, 80)
        
        print("\n" + "=" * 50)
        print("🎉 Data Creation Complete!")
        print(f"Created {len(accounts)} accounts")
        print(f"Created {opp_count} opportunities")
        print(f"Created {contact_count} contacts")
        print(f"Total records: {len(accounts) + opp_count + contact_count}")
        
        print("\nTry these voice queries now:")
        print("   • 'Show me all accounts'")
        print("   • 'Find big opportunities over 100k'")
        print("   • 'Show me tech companies'")
        print("   • 'What are my open deals?'")
        print("   • 'Show me contacts at TechCorp'")
        
        print("\nGo to http://localhost:3000 to test voice queries!")
        
    except KeyboardInterrupt:
        print("\n⏹️  Data creation interrupted by user")
    except Exception as e:
        print(f"\nError during data creation: {str(e)}")

if __name__ == "__main__":
    main() 