import os

def sync_metadata():
    metadata_file = 'metadata.yml'
    readme_file = 'README.md'
    output_file = 'README.hf.md' # This is what you would push to HF as README.md

    if not os.path.exists(metadata_file):
        print(f"Error: {metadata_file} not found.")
        return

    with open(metadata_file, 'r') as f:
        metadata = f.read()

    with open(readme_file, 'r') as f:
        readme_content = f.read()

    # Combine them with YAML delimiters
    full_content = f"---\n{metadata}---\n\n{readme_content}"

    with open(output_file, 'w') as f:
        f.write(full_content)

    print(f"Successfully created {output_file} with metadata prepended.")

if __name__ == "__main__":
    sync_metadata()
