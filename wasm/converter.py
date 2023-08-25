#!/usr/bin/env python3
import re

# Read the content of the input file
file_path = 'usearch-wasm.c'

with open(file_path, 'r') as input_file:
    content = input_file.read()

content = "#include <emscripten/emscripten.h>\n" + content

# Define the regular expression pattern and replacement
pattern = r'\b(.*) \_\_wasm\_export\_usearch\_wasm\_'
replacement = r'EMSCRIPTEN_KEEPALIVE \1 '

# Use re.sub to perform the replacement
replaced_content = re.sub(pattern, replacement, content)

# Write the replaced content to the output file
with open(file_path, 'w') as file_path:
    file_path.write(replaced_content)
