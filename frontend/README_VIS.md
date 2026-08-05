# Vis Network React Integration

To export your Python PyVis network into this React application, follow these steps:

1. **In Python**: Do not export as `.html`. Export the `nodes` and `edges` arrays as JSON.
```python
import json
# Assuming 'net' is your pyvis Network object
data = {
    "nodes": net.nodes,
    "edges": net.edges,
    "options": net.options
}
with open("graph_data.json", "w") as f:
    json.dump(data, f)
```

2. **In React**: Pass this JSON data directly to the `<VisNetworkGraph />` component (as demonstrated in `src/components/BioNetworkGraph.tsx`).

This method avoids embedding a 3MB HTML/JS file into an iframe, improving load times and allowing you to interact with the graph natively within React.
