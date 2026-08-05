const fs = require('fs');
const content = fs.readFileSync('user_prompt.txt', 'utf8');

const nodesMatch = content.match(/nodes = new vis\.DataSet\((\[.*?\])\);/s);
const edgesMatch = content.match(/edges = new vis\.DataSet\((\[.*?\])\);/s);
const optionsMatch = content.match(/var options = (\{[\s\S]*?\});/);

if (nodesMatch && edgesMatch && optionsMatch) {
  const nodes = JSON.parse(nodesMatch[1]);
  const edges = JSON.parse(edgesMatch[1]);
  const options = JSON.parse(optionsMatch[1]);
  fs.writeFileSync('src/data/visData.json', JSON.stringify({ nodes, edges, options }, null, 2));
  console.log("Extracted successfully.");
} else {
  console.log("Failed to extract.");
}
