#!/usr/bin/env python3
"""
Stock Symbol Extraction Script
Extract NASDAQ and NYSE stock symbols from PDF files
"""

import pdfplumber
import re
import os
from typing import List, Set
import pandas as pd

def extract_symbols_from_pdf(pdf_path: str) -> List[str]:
    """Extract stock symbols from a PDF file using pdfplumber"""
    symbols = []
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                # Try to extract tables first
                tables = page.extract_tables()
                if tables:
                    for table in tables:
                        for row in table:
                            if row and len(row) >= 2:
                                # Look for stock symbols in the second column (index 1)
                                potential_symbol = str(row[1]).strip() if row[1] else ""
                                if re.match(r'^[A-Z]{2,5}$', potential_symbol):
                                    symbols.append(potential_symbol)
                
                # Also extract text and look for patterns
                text = page.extract_text()
                if text:
                    lines = text.split('\n')
                    
                    for line in lines:
                        # Skip header lines and non-data lines
                        if any(skip_text in line.lower() for skip_text in [
                            'symbol', 'company name', 'market cap', 'stock price',
                            'change', 'revenue', 'total stocks', 'screener', 'find',
                            'indicators', 'all stocks listed', 'nasdaq', 'nyse',
                            'page', 'back to top', 'previous', 'next', 'rows'
                        ]):
                            continue
                        
                        # Look for patterns like "1 NVDA NVIDIA Corporation"
                        # or "27 APP AppLovin Corporation"
                        match = re.search(r'^\s*(\d+)\s+([A-Z]{2,5})\s+(.+)', line.strip())
                        if match:
                            symbol = match.group(2).strip()
                            # Additional validation - should be 2-5 uppercase letters
                            if re.match(r'^[A-Z]{2,5}$', symbol) and symbol not in ['USD', 'INC', 'LLC', 'LTD', 'CORP']:
                                symbols.append(symbol)
    
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
    
    return symbols

def extract_all_symbols(stock_list_dir: str) -> tuple:
    """Extract all NASDAQ and NYSE symbols from PDF files"""
    nasdaq_symbols = set()
    nyse_symbols = set()
    
    if not os.path.exists(stock_list_dir):
        print(f"Directory not found: {stock_list_dir}")
        return [], []
    
    # Process NASDAQ PDF files
    nasdaq_files = [f for f in os.listdir(stock_list_dir) if f.startswith('NASDAQ') and f.endswith('.pdf')]
    nasdaq_files.sort()  # Sort to process in order
    
    print(f"Found {len(nasdaq_files)} NASDAQ PDF files")
    for file in nasdaq_files:
        file_path = os.path.join(stock_list_dir, file)
        print(f"Processing {file}...")
        symbols = extract_symbols_from_pdf(file_path)
        nasdaq_symbols.update(symbols)
        print(f"  Extracted {len(symbols)} symbols from {file}")
    
    # Process NYSE PDF files
    nyse_files = [f for f in os.listdir(stock_list_dir) if f.startswith('NYSE') and f.endswith('.pdf')]
    nyse_files.sort()  # Sort to process in order
    
    print(f"Found {len(nyse_files)} NYSE PDF files")
    for file in nyse_files:
        file_path = os.path.join(stock_list_dir, file)
        print(f"Processing {file}...")
        symbols = extract_symbols_from_pdf(file_path)
        nyse_symbols.update(symbols)
        print(f"  Extracted {len(symbols)} symbols from {file}")
    
    return sorted(list(nasdaq_symbols)), sorted(list(nyse_symbols))

def main():
    """Main function to extract and save stock symbols"""
    stock_list_dir = r"C:\quant_system_v2\quant_system_full\Stock list"
    
    print("Starting stock symbol extraction...")
    nasdaq_symbols, nyse_symbols = extract_all_symbols(stock_list_dir)
    
    print(f"\nExtraction Results:")
    print(f"NASDAQ symbols extracted: {len(nasdaq_symbols)}")
    print(f"NYSE symbols extracted: {len(nyse_symbols)}")
    
    # Save to separate files
    nasdaq_file = r"C:\quant_system_v2\nasdaq_symbols.txt"
    nyse_file = r"C:\quant_system_v2\nyse_symbols.txt"
    all_symbols_file = r"C:\quant_system_v2\all_stock_symbols.txt"
    
    # Save NASDAQ symbols
    with open(nasdaq_file, 'w') as f:
        for symbol in nasdaq_symbols:
            f.write(symbol + '\n')
    print(f"NASDAQ symbols saved to: {nasdaq_file}")
    
    # Save NYSE symbols
    with open(nyse_file, 'w') as f:
        for symbol in nyse_symbols:
            f.write(symbol + '\n')
    print(f"NYSE symbols saved to: {nyse_file}")
    
    # Save all symbols combined
    all_symbols = nasdaq_symbols + nyse_symbols
    with open(all_symbols_file, 'w') as f:
        f.write(f"# NASDAQ Symbols ({len(nasdaq_symbols)} total)\n")
        for symbol in nasdaq_symbols:
            f.write(f"NASDAQ:{symbol}\n")
        f.write(f"\n# NYSE Symbols ({len(nyse_symbols)} total)\n")
        for symbol in nyse_symbols:
            f.write(f"NYSE:{symbol}\n")
    
    print(f"All symbols saved to: {all_symbols_file}")
    
    # Create CSV files for easier use
    nasdaq_df = pd.DataFrame({'Symbol': nasdaq_symbols, 'Exchange': 'NASDAQ'})
    nyse_df = pd.DataFrame({'Symbol': nyse_symbols, 'Exchange': 'NYSE'})
    all_df = pd.concat([nasdaq_df, nyse_df], ignore_index=True)
    
    nasdaq_csv = r"C:\quant_system_v2\nasdaq_symbols.csv"
    nyse_csv = r"C:\quant_system_v2\nyse_symbols.csv"
    all_csv = r"C:\quant_system_v2\all_stock_symbols.csv"
    
    nasdaq_df.to_csv(nasdaq_csv, index=False)
    nyse_df.to_csv(nyse_csv, index=False)
    all_df.to_csv(all_csv, index=False)
    
    print(f"CSV files created:")
    print(f"  {nasdaq_csv}")
    print(f"  {nyse_csv}")
    print(f"  {all_csv}")
    
    # Verification
    print(f"\nVerification:")
    print(f"Expected NASDAQ symbols: 3332, Extracted: {len(nasdaq_symbols)}")
    print(f"Expected NYSE symbols: 1936, Extracted: {len(nyse_symbols)}")
    
    if len(nasdaq_symbols) >= 3300:  # Allow some tolerance
        print("✓ NASDAQ symbol count looks good")
    else:
        print("⚠ NASDAQ symbol count seems low")
    
    if len(nyse_symbols) >= 1900:  # Allow some tolerance
        print("✓ NYSE symbol count looks good")
    else:
        print("⚠ NYSE symbol count seems low")
    
    # Show some sample symbols
    print(f"\nSample NASDAQ symbols: {nasdaq_symbols[:10]}")
    print(f"Sample NYSE symbols: {nyse_symbols[:10]}")

if __name__ == "__main__":
    main()