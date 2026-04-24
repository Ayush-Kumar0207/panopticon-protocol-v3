import re
import os

with open("FINAL_SUBMISSION_BLUEPRINT.md", "r", encoding="utf-8") as f:
    content = f.read()

mermaid_blocks = re.findall(r'```mermaid\n(.*?)\n```', content, re.DOTALL)

html_template = """
<!DOCTYPE html>
<html>
<head>
    <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
    <script>mermaid.initialize({startOnLoad:true, theme: 'forest'});</script>
    <style>
        .diagram-container {
            margin: 50px;
            padding: 20px;
            border: 1px solid #ddd;
            display: inline-block;
            background: white;
        }
    </style>
</head>
<body>
    %s
</body>
</html>
"""

containers = ""
for i, block in enumerate(mermaid_blocks):
    containers += f'<div id="diagram-{i}" class="diagram-container"><pre class="mermaid">{block}</pre></div>\n'

with open("mermaid_render.html", "w", encoding="utf-8") as f:
    f.write(html_template % containers)

print(f"Extracted {len(mermaid_blocks)} diagrams to mermaid_render.html")
