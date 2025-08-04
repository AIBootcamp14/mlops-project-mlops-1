import os
import re
from pathlib import Path
from mermaid.graph import Graph

def extract_mermaid_code(markdown_file):
    with open(markdown_file, 'r', encoding='utf-8') as f:
        content = f.read()
    mermaid_blocks = re.findall(r'```mermaid\n(.*?)\n```', content, re.DOTALL)
    return mermaid_blocks[0] if mermaid_blocks else None

def generate_svg_from_code(mermaid_code, output_file):
    graph = Graph(mermaid_code)
    svg = graph.to_mermaid()
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(svg)

if __name__ == '__main__':
    mermaid_code = extract_mermaid_code('README.md')
    if mermaid_code:
        os.makedirs('docs/images', exist_ok=True)
        generate_svg_from_code(mermaid_code, 'docs/images/architecture.svg')
