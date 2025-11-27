import re

# Read the file
with open('preprocess_datasets.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace Unicode characters with ASCII equivalents
replacements = {
    '✓': 'OK',
    '→': '->',
    '⚠️': 'WARNING:',
    '\r\n': '\n'  # Normalize line endings
}

for old, new in replacements.items():
    content = content.replace(old, new)

# Write back
with open('preprocess_datasets.py', 'w', encoding='utf-8', newline='\r\n') as f:
    f.write(content)

print("Fixed Unicode characters in preprocess_datasets.py")
