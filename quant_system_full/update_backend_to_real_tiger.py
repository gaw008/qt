#!/usr/bin/env python3
"""
Update Backend to Use Real Tiger Data
Update the backend app.py to use real Tiger data instead of mock data
"""

import os
import sys
from pathlib import Path

def update_backend_app():
    """Update the backend app.py to use real Tiger data provider"""

    app_path = Path(__file__).parent / 'dashboard' / 'backend' / 'app.py'

    try:
        # Read the current app.py
        with open(app_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Make the substitution
        old_import = "from tiger_data_provider import tiger_provider"
        new_import = "from tiger_data_provider_real import real_tiger_provider as tiger_provider"

        if old_import in content:
            updated_content = content.replace(old_import, new_import)

            # Write back the updated content
            with open(app_path, 'w', encoding='utf-8') as f:
                f.write(updated_content)

            print("✅ Successfully updated backend app.py to use real Tiger data provider")
            print(f"   Changed: {old_import}")
            print(f"   To:      {new_import}")
            return True
        else:
            print("❌ Could not find the import line to replace")
            print(f"   Looking for: {old_import}")
            return False

    except Exception as e:
        print(f"❌ Error updating backend app.py: {e}")
        return False

if __name__ == "__main__":
    print("Updating Backend to Use Real Tiger Data")
    print("=" * 50)

    success = update_backend_app()

    if success:
        print("\n🎉 Backend successfully updated!")
        print("The API will now return real Tiger account data instead of mock data.")
        print("\nNext steps:")
        print("1. Restart the backend server")
        print("2. Test the API endpoints to verify real data")
    else:
        print("\n❌ Update failed. Please check the error messages above.")