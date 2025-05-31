#!/usr/bin/env python3
"""
This script adds the AGPL-3.0 copyright header to all Python files in the project.

Copyright © 2025 Cadenzai, Inc.

This program is free software: you can redistribute it and/or modify it under the terms of the 
GNU Affero General Public License as published by the Free Software Foundation, either 
version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; 
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 
See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with this program. 
If not, see <https://www.gnu.org/licenses/>.
"""

import os
import re
import argparse
from pathlib import Path

# Copyright header template
COPYRIGHT_HEADER = '''"""
This file is part of SOMA (Self-Organizing Memory Architecture).

Copyright © 2025 Cadenzai, Inc.

SOMA is free software: you can redistribute it and/or modify it under the terms of the 
GNU Affero General Public License as published by the Free Software Foundation, either 
version 3 of the License, or (at your option) any later version.

SOMA is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; 
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 
See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with SOMA. 
If not, see <https://www.gnu.org/licenses/>.
"""
'''

def add_header_to_file(file_path):
    """Add copyright header to a Python file if it doesn't already have one."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if file already has a copyright notice
    if "Copyright © 2025 Cadenzai, Inc." in content:
        print(f"Skipping {file_path} - already has copyright header")
        return False
    
    # Handle shebang line if present
    if content.startswith('#!'):
        shebang_end = content.find('\n') + 1
        new_content = content[:shebang_end] + '\n' + COPYRIGHT_HEADER + content[shebang_end:]
    # Handle existing docstring
    elif content.lstrip().startswith('"""') or content.lstrip().startswith("'''"):
        # Find the end of the existing docstring
        content_stripped = content.lstrip()
        start_quote = content_stripped[:3]
        docstring_start = content.find(start_quote)
        docstring_end = content.find(start_quote, docstring_start + 3) + 3
        
        # Replace the existing docstring with our copyright header + original docstring content
        original_docstring = content[docstring_start:docstring_end]
        docstring_content = original_docstring.strip('"\'\n ')
        
        # Create a new docstring with copyright header and original content
        if docstring_content:
            new_docstring = COPYRIGHT_HEADER.rstrip() + "\n\n" + docstring_content + '\n"""'
        else:
            new_docstring = COPYRIGHT_HEADER
        
        new_content = content[:docstring_start] + new_docstring + content[docstring_end:]
    else:
        # No existing docstring, just add the header at the top
        new_content = COPYRIGHT_HEADER + content
    
    # Write the modified content back to the file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"Added copyright header to {file_path}")
    return True

def process_directory(directory, extensions=None, recursive=True):
    """Process all files in a directory."""
    if extensions is None:
        extensions = ['.py']
    
    modified_count = 0
    skipped_count = 0
    
    for root, dirs, files in os.walk(directory):
        if not recursive and root != directory:
            continue
            
        for file in files:
            file_path = os.path.join(root, file)
            if any(file.endswith(ext) for ext in extensions):
                if add_header_to_file(file_path):
                    modified_count += 1
                else:
                    skipped_count += 1
    
    return modified_count, skipped_count

def main():
    parser = argparse.ArgumentParser(description='Add AGPL-3.0 copyright headers to Python files')
    parser.add_argument('--dir', type=str, default='.', help='Directory to process')
    parser.add_argument('--extensions', type=str, default='.py', help='Comma-separated list of file extensions to process')
    parser.add_argument('--no-recursive', action='store_true', help='Do not process subdirectories')
    
    args = parser.parse_args()
    
    extensions = args.extensions.split(',')
    directory = args.dir
    recursive = not args.no_recursive
    
    print(f"Processing {directory} {'recursively' if recursive else 'non-recursively'}")
    print(f"Looking for files with extensions: {', '.join(extensions)}")
    
    modified, skipped = process_directory(directory, extensions, recursive)
    
    print(f"\nSummary:")
    print(f"  {modified} files modified")
    print(f"  {skipped} files skipped (already had copyright header)")
    print(f"  {modified + skipped} total files processed")

if __name__ == "__main__":
    main()
