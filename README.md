# Semantic Search Engine

A high-performance distributed semantic similarity engine built for real-world applications. This project demonstrates three distinct search strategies optimized for different use cases and performance characteristics.

## Live Demo

🚀 Experience the high-performance search engine live: **[https://semanticsearchengine.onrender.com](https://semanticsearchengine.onrender.com)**

## What This Engine Does

### Three Complementary Search Strategies

1. **Hierarchical Graph Search (HGS)** - Production-grade approximate nearest neighbor search
   - O(log N) query complexity
   - Real-time insertion with multilayer graph structure
   - Used by modern vector databases (Pinecone, Weaviate, Chroma)
   - Best for: Large-scale applications (millions of vectors)

2. **Balanced Binary Space Partitioning (BBSP)** - Exact k-NN search with pruning
   - O(log N) average case, O(N) worst case
   - Deterministic exact results
   - Axis-aligned space partitioning with greedy traversal
   - Best for: Medium-scale applications (thousands to hundreds of thousands)

3. **Linear Search** - Baseline exhaustive search
   - O(N·D) complexity (N = items, D = dimensions)
   - Useful for benchmarking and validating accuracy
   - Simple but guaranteed correct results
   - Best for: Small datasets or accuracy validation

### Distance Metrics

- **Cosine Similarity** - Text and semantic embeddings
- **Euclidean Distance** - Geometric and spatial data
- **Manhattan Distance** - Grid-based and categorical data

## Architecture

```
Your Query Vector (768D)
    │
    ├─→ HierarchicalGraphSearch  (Fast: O(log N) - ≈20ms for 1M items)
    │
    ├─→ BBSP Tree                (Balanced: O(log N) - ≈50ms for 1M items)
    │
    └─→ LinearSearch             (Accurate: O(N·D) - Baseline comparison)
```

### Core Components

- **LinearSearchIndex** - Exhaustive search for accuracy validation
- **BBSPTree** - Binary space partitioning with recursive descent
- **HierarchicalGraphSearch** - Multilayer navigable graph for approximate search
- **Distance Metrics** - Pluggable distance functions (Cosine, Euclidean, Manhattan)

## Performance Characteristics

| Method | Insertion | Query | Memory | Notes |
|--------|-----------|-------|--------|-------|
| HGS | O(log N) | O(log N) | O(N) | Production choice |
| BBSP | O(N) | O(log N)* | O(N) | Balanced accuracy/speed |
| Linear | O(1) | O(N·D) | O(N) | Baseline reference |

*BBSP scales well up to 20D, degrades in high dimensions

## Prerequisites

- C++17 or later
- G++ 7.0+ (or MSVC)
- Standard library (no external dependencies needed for core engine)

## Usage Example

```cpp
// Initialize engines
LinearSearchIndex linear;
BBSPTree bbsp(16);  // 16 dimensions
HierarchicalGraphSearch hgs(16);  // M=16, ef_construction=200

// Create embeddings
std::vector<float> queryVector = {0.9, 0.8, 0.7, ...};
Document doc1(1, "Machine Learning", embedding1);

// Insert
linear.addDocument(doc1);
bbsp.addDocument(doc1);
hgs.addDocument(doc1, cosineDistance);

// Search
auto results = hgs.findNearest(queryVector, 5, 200, cosineDistance);

// Results: {distance, document_id}
for (auto [dist, id] : results) {
    std::cout << "ID: " << id << " Distance: " << dist << "\n";
}
```

## Real-World Applications

### Recommendation Systems
- Product recommendations based on user embeddings
- Content-based filtering for music/movies

### Information Retrieval
- Full-text search with semantic understanding
- Document similarity and clustering

### Question-Answering Systems
- Retrieve relevant documents for user queries
- Context injection for LLMs (RAG pipelines)

### Anomaly Detection
- Find unusual patterns in high-dimensional data
- Real-time fraud detection

## Implementation Details

### HierarchicalGraphSearch Algorithm

The algorithm builds a multilayer graph where:
- Upper layers contain fewer nodes with long-range connections (highways)
- Lower layers contain all nodes with many connections (local neighborhoods)

**Insert Process:**
1. Generate random level for new item
2. Search from top layer down to target layer
3. At each layer, find M nearest neighbors
4. Create bidirectional connections

**Search Process:**
1. Start at entry point (top layer)
2. Greedily navigate toward query down the layers
3. At layer 0, expand search with ef (expansion factor) candidates
4. Return k nearest results

### BBSP Tree Structure

A recursive binary tree that:
- Cycles through dimensions at each level (axis-aligned splits)
- Prunes branches where best possible distance exceeds current best
- Uses priority queue for efficient k-NN retrieval

## Benchmark Methodology

When comparing algorithms, test with:
- Varying dataset sizes (1K, 10K, 100K, 1M items)
- Different dimensions (16D, 128D, 768D)
- Multiple distance metrics
- Varied k values (1, 5, 10, 50, 100)

Record both latency and accuracy (recall@K).

## Configuration

Tune these parameters for your use case:

```cpp
// HierarchicalGraphSearch
int maxConnections = 16;        // M - connections per node (↑ = slower insert, better search)
int constructionIterations = 200; // ef_construction (↑ = slower insert, better accuracy)

// Search parameters
int searchIterations = 200;     // ef - candidates to expand (↑ = slower search, better recall)
int resultLimit = 5;            // k - final results to return
```

## Lessons Learned

1. **Graph-based indexing** outperforms tree-based methods in high dimensions (>20D)
2. **Pruning effectiveness** depends heavily on distance metric choice
3. **Real-world embeddings** are often sparse; many distance calculations can be skipped
4. **Entry point selection** significantly impacts HGS performance
5. **Trade-offs exist** - no single algorithm dominates all scenarios

## Future Enhancements

- [ ] GPU acceleration for distance calculations
- [ ] Multi-threaded insertion for parallel indexing
- [ ] Persistent storage and loading from disk
- [ ] Quantization support (8-bit, 16-bit integers)
- [ ] Hybrid algorithms combining HGS + BBSP strengths

## License

MIT - Free to use and modify for any purpose

## References

- Malkov & Yashunin, "Efficient and robust approximate nearest neighbor search" (HNSW paper)
- Bentley, "Multidimensional binary search trees" (KD-Tree original)
- High-dimensional data challenges in approximate search

---

**Created as an educational project** to understand how production vector databases work internally.
