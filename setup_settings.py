#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
from pathlib import Path

def main():
    # Get current working directory for absolute paths
    current_dir = os.path.abspath(os.getcwd())
    print(f"Current directory: {current_dir}")
    
    # Ultralytics settings file location
    settings_file = os.path.expanduser("~/Library/Application Support/Ultralytics/settings.json")
    settings_dir = os.path.dirname(settings_file)
    
    # Create directory if it doesn't exist
    os.makedirs(settings_dir, exist_ok=True)
    
    # Default settings
    settings = {}
    
    # Load existing settings if file exists
    if os.path.exists(settings_file):
        try:
            with open(settings_file, 'r') as f:
                settings = json.load(f)
            print("Loaded existing Ultralytics settings")
        except Exception as e:
            print(f"Error loading settings: {e}")
    
    # Update dataset directory to current project directory
    settings['datasets_dir'] = current_dir
    print(f"Setting datasets_dir to: {current_dir}")
    
    # Save settings
    try:
        with open(settings_file, 'w') as f:
            json.dump(settings, f, indent=2)
        print(f"Updated settings saved to: {settings_file}")
    except Exception as e:
        print(f"Error saving settings: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("Settings updated successfully!")
    else:
        print("Failed to update settings.")