# Quick Data Import Instructions

You now have 3 CSV files ready to import:

## Files Ready:
- **`sample_data.csv`** - 50 Accounts
- **`sample_opportunities.csv`** - 50 Opportunities (with AccountName column)  
- **`sample_contacts.csv`** - 52 Contacts (with AccountName column)

## Import Order (Important!):

### 1. Import Accounts First
1. Go to your Salesforce org: https://flow-inspiration-8638.my.salesforce.com
2. Setup → Data Management → Data Import Wizard
3. Choose "Accounts and Contacts" → "Accounts"
4. Select "Add new records"
5. Upload `sample_data.csv`
6. Map fields automatically, click Next
7. Start Import

### 2. Import Opportunities 
1. Data Import Wizard → "Opportunities" 
2. Upload `sample_opportunities.csv`
3. **Key**: Map "AccountName" to "Account Name" (not Account ID)
4. This will link opportunities to accounts by name
5. Start Import

### 3. Import Contacts
1. Data Import Wizard → "Accounts and Contacts" → "Contacts"
2. Upload `sample_contacts.csv` 
3. Map "AccountName" to "Account Name"
4. Start Import

## Quick Test:

After import, test your voice app with:
- **"Show me all accounts"** - should return ~50 accounts
- **"Find opportunities over 100k"** - should return multiple opportunities
- **"Show me technology companies"** - should return tech accounts
- **"Find contacts at TechCorp"** - should return contacts

## Already Working!

I see from your recent logs that you already have data working:
```
INFO:__main__:Query successful: 32 records
```

Your voice query for opportunities over $100k already returned 32 records! 🎉

## Voice Query Examples:
- "Show me my biggest opportunities"
- "Find accounts in the technology industry" 
- "Show me contacts who are CEOs"
- "What deals are closing this month?"
- "Find prospects in healthcare"

Total records after import: **~150+ records** to make your voice queries much more interesting! 