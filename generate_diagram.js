const fs = require('fs');
const path = require('path');
const mermaid = require('mermaid');

const inputPath = path.join(__dirname, 'README.md');
const outputPath = path.join(__dirname, 'docs', 'images', 'architecture.svg');

const markdownContent = fs.readFileSync(inputPath, 'utf-8');
const mermaidCodeMatch = markdownContent.match(/```mermaid\n([\s\S]+?)\n```/);

if (mermaidCodeMatch && mermaidCodeMatch[1]) {
  const mermaidCode = mermaidCodeMatch[1];
  const { svg } = mermaid.render('diagram', mermaidCode);

  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  fs.writeFileSync(outputPath, svg, 'utf-8');

  console.log('Diagram generated successfully!');
} else {
  console.error('No Mermaid code block found in README.md');
  process.exit(1);
}
