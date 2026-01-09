#!/usr/bin/env python3
"""
Unicode Fix Tool - Remove Unicode characters and replace with ASCII equivalents
"""

import os
import re
import sys
import glob
from pathlib import Path

# Configure encoding
os.environ['PYTHONIOENCODING'] = 'utf-8'

# Unicode to ASCII replacement mappings
UNICODE_REPLACEMENTS = {
    # Emojis to text
    '[TARGET]': '[TARGET]',
    '[ROCKET]': '[ROCKET]',
    '[OK]': '[OK]',
    '[FAIL]': '[FAIL]',
    '[WARNING]': '[WARNING]',
    '[REFRESH]': '[REFRESH]',
    '[WAITING]': '[WAITING]',
    '[SEARCH]': '[SEARCH]',
    '[CHART]': '[CHART]',
    '[IDEA]': '[IDEA]',
    '[AI]': '[AI]',
    '[SHIELD]': '[SHIELD]',
    '[FAST]': '[FAST]',
    '[TOOL]': '[TOOL]',
    '[UP]': '[UP]',
    '[DOWN]': '[DOWN]',
    '[MONEY]': '[MONEY]',
    '[PHONE]': '[PHONE]',
    '[COMPUTER]': '[COMPUTER]',
    '[GLOBAL]': '[GLOBAL]',
    '[ART]': '[ART]',
    '[BUILD]': '[BUILD]',
    '[PACKAGE]': '[PACKAGE]',
    '[KEY]': '[KEY]',
    '[STAR]': '[STAR]',
    '[FIRE]': '[FIRE]',
    '[DIAMOND]': '[DIAMOND]',

    # Special characters
    '->': '->',
    '<-': '<-',
    '=>': '=>',
    '<->': '<->',
    'therefore': 'therefore',
    'because': 'because',
    '~=': '~=',
    '<=': '<=',
    '>=': '>=',
    '+/-': '+/-',
    'x': 'x',
    '/': '/',
    'inf': 'inf',
    'sqrt': 'sqrt',
    'sum': 'sum',
    'delta': 'delta',
    'grad': 'grad',
    'in': 'in',
    'not_in': 'not_in',
    'for_all': 'for_all',
    'exists': 'exists',

    # Currency symbols
    'EUR': 'EUR',
    'GBP': 'GBP',
    'JPY': 'JPY',
    'BTC': 'BTC',

    # Mathematical symbols
    ' degrees': ' degrees',
    '^1': '^1',
    '^2': '^2',
    '^3': '^3',
    '^4': '^4',
    '1/2': '1/2',
    '1/3': '1/3',
    '2/3': '2/3',
    '1/4': '1/4',
    '3/4': '3/4',

    # Box drawing and special punctuation
    '|': '|',
    '-': '-',
    '+': '+',
    '+': '+',
    '+': '+',
    '+': '+',
    '+': '+',
    '+': '+',
    '+': '+',
    '+': '+',
    '+': '+',
    '"': '"',
    '"': '"',
    '''''''''...': '...',
    '-': '-',
    '--': '--',

    # Common accented characters
    'a': 'a', 'a': 'a', 'a': 'a', 'a': 'a', 'a': 'a', 'a': 'a',
    'e': 'e', 'e': 'e', 'e': 'e', 'e': 'e',
    'i': 'i', 'i': 'i', 'i': 'i', 'i': 'i',
    'o': 'o', 'o': 'o', 'o': 'o', 'o': 'o', 'o': 'o', 'o': 'o',
    'u': 'u', 'u': 'u', 'u': 'u', 'u': 'u',
    'y': 'y', 'y': 'y',
    'n': 'n',
    'c': 'c',

    # Uppercase versions
    'A': 'A', 'A': 'A', 'A': 'A', 'A': 'A', 'A': 'A', 'A': 'A',
    'E': 'E', 'E': 'E', 'E': 'E', 'E': 'E',
    'I': 'I', 'I': 'I', 'I': 'I', 'I': 'I',
    'O': 'O', 'O': 'O', 'O': 'O', 'O': 'O', 'O': 'O', 'O': 'O',
    'U': 'U', 'U': 'U', 'U': 'U', 'U': 'U',
    'Y': 'Y', 'Y': 'Y',
    'N': 'N',
    'C': 'C',
}

def fix_unicode_in_file(file_path: str) -> bool:
    """
    Fix Unicode characters in a single file
    Returns True if changes were made
    """
    try:
        # Read file with proper encoding handling
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    content = f.read()
            except UnicodeDecodeError:
                with open(file_path, 'r', encoding='cp1252', errors='replace') as f:
                    content = f.read()

        original_content = content

        # Apply replacements
        for unicode_char, ascii_replacement in UNICODE_REPLACEMENTS.items():
            content = content.replace(unicode_char, ascii_replacement)

        # Remove any remaining non-ASCII characters with a warning
        non_ascii_pattern = re.compile(r'[^\x00-\x7F]')
        matches = non_ascii_pattern.findall(content)
        if matches:
            unique_chars = list(set(matches))
            print(f"WARNING: Removing {len(unique_chars)} unmapped Unicode characters from {file_path}: {unique_chars[:10]}")
            content = non_ascii_pattern.sub('?', content)

        # Write back if changes were made
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Fixed Unicode characters in: {file_path}")
            return True

        return False

    except Exception as e:
        print(f"ERROR fixing {file_path}: {e}")
        return False

def fix_unicode_in_directory(directory: str = ".", patterns: list = None):
    """
    Fix Unicode characters in all Python files in directory
    """
    if patterns is None:
        patterns = ["*.py"]

    fixed_files = []
    error_files = []

    for pattern in patterns:
        for file_path in glob.glob(os.path.join(directory, "**", pattern), recursive=True):
            try:
                if fix_unicode_in_file(file_path):
                    fixed_files.append(file_path)
            except Exception as e:
                error_files.append((file_path, str(e)))
                print(f"ERROR processing {file_path}: {e}")

    # Summary
    print(f"\n=== Unicode Fix Summary ===")
    print(f"Fixed files: {len(fixed_files)}")
    print(f"Error files: {len(error_files)}")

    if fixed_files:
        print("\nFixed files:")
        for file_path in fixed_files[:20]:  # Show first 20
            print(f"  {file_path}")
        if len(fixed_files) > 20:
            print(f"  ... and {len(fixed_files) - 20} more")

    if error_files:
        print("\nError files:")
        for file_path, error in error_files[:10]:  # Show first 10
            print(f"  {file_path}: {error}")

if __name__ == "__main__":
    print("Starting Unicode fix process...")

    # Fix all Python files in current directory and subdirectories
    fix_unicode_in_directory(".", ["*.py"])

    print("\nUnicode fix process completed!")