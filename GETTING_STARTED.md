# GETTING STARTED GUIDE

## Quick Start (5 minutes)

### Prerequisites
- C++17 compiler (g++ 7.0+ or MSVC 2017+)
- Any modern OS (Windows, Linux, macOS)

### Step 1: Build the Engine

```bash
# Navigate to project directory
cd SemanticSearchEngine

# Compile with g++
g++ -std=c++17 -O2 engine.cpp -o engine

# Or with MSVC
cl /std:c++latest /O2 engine.cpp
```

### Step 2: Run Basic Example

Create a test file `example.cpp`:

```cpp
#include "engine.cpp"
#include <iostream>

int main() {
    // Initialize searcher
    LinearSearchIndex searcher;
    BBSPTree bbsp(16);
    
    // Create sample embeddings
    std::vector<float> emb1 = {0.9, 0.8, 0.7, /* ... */};
    Document doc1(1, "Machine Learning", emb1);
    
    // Insert and search
    searcher.addDocument(doc1);
    
    std::vector<float> query = {0.85, 0.75, 0.65, /* ... */};
    auto results = searcher.findNearest(query, 5, computeCosineDistance);
    
    // Print results
    for (auto [distance, id] : results) {
        std::cout << "ID: " << id << " Distance: " << distance << "\n";
    }
    
    return 0;
}
```

Compile and run:
```bash
g++ -std=c++17 example.cpp -o example
./example
```

## Common Tasks

### Task 1: Benchmark All Three Algorithms

```cpp
// Create sample data
LinearSearchIndex linear;
BBSPTree bbsp(768);
HierarchicalGraphSearch hgs(16, 200);

// Load your vectors
for (const auto& doc : documents) {
    linear.addDocument(doc);
    bbsp.addDocument(doc);
    hgs.addDocument(doc, computeCosineDistance);
}

// Benchmark
auto start = std::chrono::high_resolution_clock::now();
auto results = hgs.findNearest(query, 10, 200, computeCosineDistance);
auto end = std::chrono::high_resolution_clock::now();

auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
std::cout << "HGS took: " << duration.count() << "ms\n";
```

### Task 2: Switch Distance Metrics

```cpp
// Using cosine similarity
auto cosineDist = getMetricFunction("cosine");
auto results = searcher.findNearest(query, k, cosineDist);

// Using Euclidean distance
auto euclideanDist = getMetricFunction("euclidean");
auto results = searcher.findNearest(query, k, euclideanDist);

// Using Manhattan distance
auto manhattanDist = getMetricFunction("manhattan");
auto results = searcher.findNearest(query, k, manhattanDist);
```

### Task 3: Profile Large-Scale Performance

```cpp
#include <chrono>

// Test with 1M vectors
HierarchicalGraphSearch hgs(16);

// Insertion benchmark
auto insertStart = std::chrono::high_resolution_clock::now();
for (int i = 0; i < 1000000; i++) {
    // Generate random embedding
    std::vector<float> embedding(768);
    // ... fill with data
    
    hgs.addDocument(doc, metric);
}
auto insertEnd = std::chrono::high_resolution_clock::now();

// Query benchmark
auto queryStart = std::chrono::high_resolution_clock::now();
auto results = hgs.findNearest(queryVector, 10, 200, metric);
auto queryEnd = std::chrono::high_resolution_clock::now();

std::cout << "Insertion time: " 
          << std::chrono::duration_cast<std::chrono::seconds>(insertEnd - insertStart).count() 
          << "s\n";
std::cout << "Query time: " 
          << std::chrono::duration_cast<std::chrono::milliseconds>(queryEnd - queryStart).count() 
          << "ms\n";
```

## Architecture Overview

### Class Hierarchy

```
Document
├── id (int)
├── title (std::string)
├── content (std::string)
└── embedding (std::vector<float>)

Searchers:
├── LinearSearchIndex (exhaustive, O(N·D))
├── BBSPTree (binary space partitioning, O(log N))
└── HierarchicalGraphSearch (multilayer graph, O(log N))

Distance Metrics:
├── computeCosineDistance
├── computeEuclideanDistance
├── computeManhattanDistance
└── getMetricFunction(name) -> DistanceMetric
```

### Data Flow

```
Input: Query Vector (768D)
  ↓
Choose Algorithm & Metric
  ↓
Insert/Build Index (offline)
  ↓
Execute Search (online)
  ↓
Retrieve K Nearest Neighbors
  ↓
Output: [(distance, id), ...]
```

## Optimization Tips

### For Insertion Speed
- Use Linear Search if you need to load millions of vectors quickly
- HGS has overhead per insertion, but pays off during queries

### For Query Speed
- Use HGS for high-dimensional data (>20D)
- Use BBSP for medium dimensions (5D-20D)
- Use Linear only for validation

### For Memory Efficiency
- All three use O(N) space
- HGS has higher constant factors due to multilayer structure
- Consider using 16-bit float representations for embeddings (2x memory savings)

## Troubleshooting

### Slow Queries
- Check if k-value is too large
- Verify distance metric matches your data type
- For HGS, increase `searchIterations` parameter

### Memory Issues
- Reduce vector dimension if possible
- Use single-precision (float) instead of double
- Stream large datasets instead of loading all at once

### Incorrect Results
- Verify embeddings are normalized (for cosine similarity)
- Check if query vector has same dimension as stored vectors
- Compare with Linear Search results as ground truth

## Next Steps

1. **Integrate with Real Data**: Replace demo embeddings with actual data
2. **Add Persistence**: Serialize/deserialize indexes to disk
3. **Multi-threading**: Parallelize insertion for faster indexing
4. **GPU Acceleration**: Move distance calculations to GPU
5. **Quantization**: Use 8-bit integers instead of floats for 4x compression

## Resources

- **Algorithm Paper**: "Efficient and robust approximate nearest neighbor search" (Malkov & Yashunin)
- **KD-Tree Reference**: Bentley's original paper on multidimensional binary search
- **Practical Guide**: Designing High-Performance Vector Databases

## Support & Questions

- Check `README.md` for detailed algorithm explanations
- Review `config.json` for tuning parameters
- Open an issue or check documentation for FAQs

---

Good luck with your semantic search engine! Happy building! 🚀
