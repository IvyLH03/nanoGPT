import pandas as pd
import os

# 1. Load the CSV file
input_path = os.path.join(os.path.dirname(__file__), 'train.csv')
df = pd.read_csv(input_path)

# 2. Define the cleaning function
def clean_and_format_text(text):
    if isinstance(text, str):
        # Remove trailing newline characters
        text = text.rstrip('\n')

        # Replace duplicated quotation marks (e.g., "" with ")
        # This addresses common escaping issues from CSV parsing
        text = text.replace('""', '"')

        # Remove leading/trailing quotes if they were remnants of CSV parsing
        # and don't make sense as part of the content.
        # This handles cases like '"joke text"' -> 'joke text'
        if text.startswith('"') and text.endswith('"'):
             if len(text) > 1:
                text = text.strip('"')

        return text
    return ''

# 3. Apply the cleaning function
df['cleaned_text'] = df['text'].apply(clean_and_format_text)

# 4. Prepare the content for the .txt file: each item followed by an empty line
# The '\n\n' ensures there is one empty line between each item.
txt_content = '\n\n'.join(df['cleaned_text'].astype(str).tolist())

# 5. Save the content to a .txt file
output_filename = 'formatted_jokes.txt'
with open(output_filename, 'w', encoding='utf-8') as f:
    f.write(txt_content)

print(f"Success! The formatted text has been saved to {output_filename}")