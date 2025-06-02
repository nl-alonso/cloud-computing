#!/usr/bin/env python3
"""
Fix Account-Opportunity Relationships
Diagnose and repair links between opportunities and accounts
"""

import requests
import json

# Use your Flask app to execute queries
FLASK_URL = "http://localhost:3000"

def test_flask():
    """Test if Flask app is running"""
    try:
        response = requests.get(f"{FLASK_URL}/health")
        return response.status_code == 200
    except:
        return False

def run_voice_query(query):
    """Execute a voice query via Flask app"""
    try:
        response = requests.post(f"{FLASK_URL}/api/voice-query", 
                               json={"query": query},
                               timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return {"success": False, "error": f"HTTP {response.status_code}"}
    except Exception as e:
        return {"success": False, "error": str(e)}

def diagnose_data():
    """Run diagnostic queries to understand the current state"""
    
    print("DIAGNOSING YOUR SALESFORCE DATA")
    print("=" * 50)
    
    if not test_flask():
        print("Flask app not running. Start it first:")
        print("   python3 app.py")
        return
    
    print("Flask app is running")
    print("\n🏢 CHECKING ACCOUNTS...")
    
    # Check accounts
    result = run_voice_query("Show me all accounts with their names and IDs")
    if result.get('success'):
        count = result.get('totalSize', 0)
        print(f"Found {count} accounts")
        if count > 0:
            print("   Sample accounts found in your org")
        else:
            print("No accounts found - you need to import accounts first!")
    else:
        print(f"Account query failed: {result.get('error')}")
    
    print("\n💰 CHECKING OPPORTUNITIES...")
    
    # Check opportunities
    result = run_voice_query("Show me all opportunities with account names")
    if result.get('success'):
        count = result.get('totalSize', 0)
        print(f"Found {count} opportunities")
        
        # Check if they have account relationships
        if count > 0:
            result2 = run_voice_query("Show me opportunities where account name is not null")
            if result2.get('success'):
                linked_count = result2.get('totalSize', 0)
                unlinked_count = count - linked_count
                
                if linked_count > 0:
                    print(f"{linked_count} opportunities are properly linked to accounts")
                if unlinked_count > 0:
                    print(f"{unlinked_count} opportunities are NOT linked to accounts")
                    print("   These need to be fixed!")
            else:
                print("Could not check opportunity-account relationships")
        else:
            print("No opportunities found")
    else:
        print(f"Opportunity query failed: {result.get('error')}")
    
    print("\nDIAGNOSTIC SUMMARY")
    print("=" * 50)
    return result

def show_solutions():
    """Show solutions based on the diagnosis"""
    
    print("\nSOLUTIONS:")
    print("=" * 20)
    
    print("\n1️⃣ CLEAN SLATE APPROACH (Recommended)")
    print("   If your data is test data, delete and re-import:")
    print("   a) Delete all opportunities in Salesforce")
    print("   b) Import accounts first: sample_data.csv")
    print("   c) Import opportunities: sample_opportunities.csv")
    print("   d) Import contacts: sample_contacts.csv")
    
    print("\n2️⃣ MANUAL FIX IN SALESFORCE")
    print("   Go to Salesforce and manually edit opportunities:")
    print("   a) Go to Opportunities tab")
    print("   b) Open each opportunity")
    print("   c) Set the Account Name field")
    print("   d) Save")
    
    print("\n3️⃣ BULK UPDATE (Advanced)")
    print("   Use Salesforce Data Import Wizard to update:")
    print("   a) Export current opportunities with IDs")
    print("   b) Add AccountName column")
    print("   c) Re-import as 'Update existing records'")
    
    print("\nQUICK TEST:")
    print("   Try this voice query: 'Show me opportunities with their account names'")
    print("   If you see account names, the links are working!")

def main():
    print("Account-Opportunity Relationship Fixer")
    print("=" * 50)
    
    diagnose_data()
    show_solutions()
    
    print(f"\nNEXT STEPS:")
    print("1. Follow the diagnostic results above")
    print("2. Choose the best solution for your situation")
    print("3. Test with voice queries after fixing")
    print("4. Your voice app will work much better with proper relationships!")

if __name__ == "__main__":
    main() 