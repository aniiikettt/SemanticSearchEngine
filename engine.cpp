#include "server.h"
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <random>
#include <chrono>
#include <mutex>
#include <unordered_map>
#include <queue>
#include <set>
#include <sstream>
#include <iomanip>
#include <functional>
#include <fstream>
#include <climits>

/*
 * SEMANTIC SEARCH ENGINE
 * A high-performance semantic similarity engine with multiple search strategies
 * 
 * Features:
 * - Approximate Nearest Neighbor (ANN) search using Hierarchical Graph Search
 * - Balanced Binary Space Partitioning (BBSP) for exact search
 * - Exhaustive Linear Search as baseline
 * - Support for Cosine, Euclidean, and Manhattan distance metrics
 * - Real-time vector insertion and deletion
 */

static const int DEFAULT_DIMS = 16;

// ========================================
// VECTOR REPRESENTATION
// ========================================

struct Document {
    int id;
    std::string title;
    std::string content;
    std::vector<float> embedding;
    
    Document() = default;
    Document(int id, const std::string& title, const std::vector<float>& embedding)
        : id(id), title(title), embedding(embedding) {}
};

using DistanceMetric = std::function<float(const std::vector<float>&, const std::vector<float>&)>;

// ========================================
// DISTANCE METRICS
// ========================================

float computeEuclideanDistance(const std::vector<float>& v1, const std::vector<float>& v2) {
    float distance = 0.0f;
    for (size_t i = 0; i < v1.size(); i++) {
        float diff = v1[i] - v2[i];
        distance += diff * diff;
    }
    return std::sqrt(distance);
}

float computeCosineDistance(const std::vector<float>& v1, const std::vector<float>& v2) {
    float dotProduct = 0.0f, mag1 = 0.0f, mag2 = 0.0f;
    for (size_t i = 0; i < v1.size(); i++) {
        dotProduct += v1[i] * v2[i];
        mag1 += v1[i] * v1[i];
        mag2 += v2[i] * v2[i];
    }
    
    if (mag1 < 1e-9f || mag2 < 1e-9f) return 1.0f;
    return 1.0f - dotProduct / (std::sqrt(mag1) * std::sqrt(mag2));
}

float computeManhattanDistance(const std::vector<float>& v1, const std::vector<float>& v2) {
    float distance = 0.0f;
    for (size_t i = 0; i < v1.size(); i++) {
        distance += std::abs(v1[i] - v2[i]);
    }
    return distance;
}

DistanceMetric getMetricFunction(const std::string& metricName) {
    if (metricName == "cosine") return computeCosineDistance;
    if (metricName == "manhattan") return computeManhattanDistance;
    return computeEuclideanDistance;
}

// ========================================
// EXHAUSTIVE LINEAR SEARCH
// ========================================

class LinearSearchIndex {
public:
    std::vector<Document> documents;

    void addDocument(const Document& doc) {
        documents.push_back(doc);
    }

    std::vector<std::pair<float, int>> findNearest(
        const std::vector<float>& queryVec, int k, DistanceMetric metric)
    {
        std::vector<std::pair<float, int>> results;
        for (const auto& doc : documents) {
            float distance = metric(queryVec, doc.embedding);
            results.push_back({distance, doc.id});
        }
        std::sort(results.begin(), results.end());
        if (results.size() > (size_t)k) {
            results.resize(k);
        }
        return results;
    }

    void deleteDocument(int docId) {
        documents.erase(
            std::remove_if(documents.begin(), documents.end(),
                [docId](const Document& d) { return d.id == docId; }),
            documents.end()
        );
    }
};

// ========================================
// BALANCED BINARY SPACE PARTITIONING TREE
// ========================================

struct BBSPNode {
    Document document;
    BBSPNode* leftChild = nullptr;
    BBSPNode* rightChild = nullptr;
    
    explicit BBSPNode(const Document& doc) : document(doc) {}
};

class BBSPTree {
    BBSPNode* root = nullptr;
    int dimensionCount;

    void deleteNode(BBSPNode* node) {
        if (!node) return;
        deleteNode(node->leftChild);
        deleteNode(node->rightChild);
        delete node;
    }

    BBSPNode* insertNode(BBSPNode* node, const Document& doc, int depth) {
        if (!node) return new BBSPNode(doc);
        
        int splitAxis = depth % dimensionCount;
        if (doc.embedding[splitAxis] < node->document.embedding[splitAxis]) {
            node->leftChild = insertNode(node->leftChild, doc, depth + 1);
        } else {
            node->rightChild = insertNode(node->rightChild, doc, depth + 1);
        }
        return node;
    }

    BBSPNode* buildBalanced(std::vector<Document>& docs, int start, int end, int depth) {
        if (start >= end) return nullptr;
        
        int splitAxis = depth % dimensionCount;
        int mid = start + (end - start) / 2;
        
        std::nth_element(docs.begin() + start, docs.begin() + mid, docs.begin() + end,
            [splitAxis](const Document& a, const Document& b) {
                return a.embedding[splitAxis] < b.embedding[splitAxis];
            });
            
        BBSPNode* node = new BBSPNode(docs[mid]);
        node->leftChild = buildBalanced(docs, start, mid, depth + 1);
        node->rightChild = buildBalanced(docs, mid + 1, end, depth + 1);
        return node;
    }

    void findKNearest(
        BBSPNode* node,
        const std::vector<float>& query,
        int k,
        int depth,
        DistanceMetric metric,
        std::priority_queue<std::pair<float, int>>& candidates)
    {
        if (!node) return;
        
        float distance = metric(query, node->document.embedding);
        
        if ((int)candidates.size() < k || distance < candidates.top().first) {
            candidates.push({distance, node->document.id});
            if ((int)candidates.size() > k) {
                candidates.pop();
            }
        }
        
        int splitAxis = depth % dimensionCount;
        float axisDifference = query[splitAxis] - node->document.embedding[splitAxis];
        
        BBSPNode* primaryBranch = (axisDifference < 0) ? node->leftChild : node->rightChild;
        BBSPNode* secondaryBranch = (axisDifference < 0) ? node->rightChild : node->leftChild;
        
        findKNearest(primaryBranch, query, k, depth + 1, metric, candidates);
        
        if ((int)candidates.size() < k || std::abs(axisDifference) < candidates.top().first) {
            findKNearest(secondaryBranch, query, k, depth + 1, metric, candidates);
        }
    }

public:
    explicit BBSPTree(int dims) : dimensionCount(dims) {}
    
    ~BBSPTree() { deleteNode(root); }

    void addDocument(const Document& doc) {
        root = insertNode(root, doc, 0);
    }

    std::vector<std::pair<float, int>> findNearest(
        const std::vector<float>& query, int k, DistanceMetric metric)
    {
        std::priority_queue<std::pair<float, int>> candidates;
        findKNearest(root, query, k, 0, metric, candidates);
        
        std::vector<std::pair<float, int>> results;
        while (!candidates.empty()) {
            results.push_back(candidates.top());
            candidates.pop();
        }
        std::sort(results.begin(), results.end());
        return results;
    }

    void rebuild(const std::vector<Document>& docs) {
        deleteNode(root);
        root = nullptr;
        std::vector<Document> docsCopy = docs;
        root = buildBalanced(docsCopy, 0, docsCopy.size(), 0);
    }
};

// ========================================
// HIERARCHICAL GRAPH SEARCH (Production-Grade ANN)
// ========================================

class HierarchicalGraphSearch {
    struct GraphNode {
        Document document;
        int maxLayer;
        std::vector<std::vector<int>> neighbors;
    };

    std::unordered_map<int, GraphNode> graph;
    int maxConnections, layerMultiplier, constructionIterations;
    float layerProbability;
    int topLayer = -1;
    int entryPoint = -1;
    std::mt19937 randomGenerator;

    int generateRandomLayer() {
        std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
        return (int)std::floor(-std::log(distribution(randomGenerator)) * layerProbability);
    }

    std::vector<std::pair<float, int>> searchAtLayer(
        const std::vector<float>& query,
        int entryNode,
        int numCandidates,
        int layer,
        DistanceMetric metric)
    {
        std::unordered_map<int, bool> visited;
        std::priority_queue<std::pair<float, int>,
            std::vector<std::pair<float, int>>, std::greater<>> candidates;
        std::priority_queue<std::pair<float, int>> results;

        float initialDistance = metric(query, graph[entryNode].document.embedding);
        visited[entryNode] = true;
        candidates.push({initialDistance, entryNode});
        results.push({initialDistance, entryNode});

        while (!candidates.empty()) {
            auto [candidateDistance, candidateId] = candidates.top();
            candidates.pop();
            
            if ((int)results.size() >= numCandidates && 
                candidateDistance > results.top().first) {
                break;
            }
            
            if (layer >= (int)graph[candidateId].neighbors.size()) {
                continue;
            }

            for (int neighborId : graph[candidateId].neighbors[layer]) {
                if (visited[neighborId] || !graph.count(neighborId)) {
                    continue;
                }
                visited[neighborId] = true;
                
                float neighborDistance = metric(query, graph[neighborId].document.embedding);
                if ((int)results.size() < numCandidates || 
                    neighborDistance < results.top().first) {
                    candidates.push({neighborDistance, neighborId});
                    results.push({neighborDistance, neighborId});
                    if ((int)results.size() > numCandidates) {
                        results.pop();
                    }
                }
            }
        }

        std::vector<std::pair<float, int>> output;
        while (!results.empty()) {
            output.push_back(results.top());
            results.pop();
        }
        std::sort(output.begin(), output.end());
        return output;
    }

    std::vector<int> selectOptimalNeighbors(
        std::vector<std::pair<float, int>>& candidates,
        int maxCount)
    {
        std::vector<int> selected;
        for (int i = 0; i < std::min((int)candidates.size(), maxCount); i++) {
            selected.push_back(candidates[i].second);
        }
        return selected;
    }

public:
    HierarchicalGraphSearch(int connections = 16, int constructionIter = 200)
        : maxConnections(connections),
          layerMultiplier(2 * connections),
          constructionIterations(constructionIter),
          layerProbability(1.0f / std::log((float)connections)),
          randomGenerator(42) {}

    void addDocument(const Document& doc, DistanceMetric metric) {
        int docId = doc.id;
        int layer = generateRandomLayer();
        graph[docId] = {doc, layer, std::vector<std::vector<int>>(layer + 1)};

        if (entryPoint == -1) {
            entryPoint = docId;
            topLayer = layer;
            return;
        }

        int currentEntry = entryPoint;
        
        // Navigate from top layer to target layer
        for (int currentLayer = topLayer; currentLayer > layer; currentLayer--) {
            if (currentLayer < (int)graph[currentEntry].neighbors.size()) {
                auto nearest = searchAtLayer(doc.embedding, currentEntry, 1, currentLayer, metric);
                if (!nearest.empty()) {
                    currentEntry = nearest[0].second;
                }
            }
        }

        // Insert into all layers from min(topLayer, layer) down to 0
        for (int currentLayer = std::min(topLayer, layer); currentLayer >= 0; currentLayer--) {
            auto candidates = searchAtLayer(doc.embedding, currentEntry, constructionIterations, currentLayer, metric);
            int connectionLimit = (currentLayer == 0) ? layerMultiplier : maxConnections;
            auto selected = selectOptimalNeighbors(candidates, connectionLimit);
            
            graph[docId].neighbors[currentLayer] = selected;

            // Update reciprocal connections
            for (int neighborId : selected) {
                if (!graph.count(neighborId)) continue;
                
                if ((int)graph[neighborId].neighbors.size() <= currentLayer) {
                    graph[neighborId].neighbors.resize(currentLayer + 1);
                }
                
                auto& connections = graph[neighborId].neighbors[currentLayer];
                connections.push_back(docId);
                
                if ((int)connections.size() > connectionLimit) {
                    // Prune excess connections
                    std::vector<std::pair<float, int>> distances;
                    for (int connectedId : connections) {
                        if (graph.count(connectedId)) {
                            float dist = metric(graph[neighborId].document.embedding,
                                              graph[connectedId].document.embedding);
                            distances.push_back({dist, connectedId});
                        }
                    }
                    std::sort(distances.begin(), distances.end());
                    
                    connections.clear();
                    for (int i = 0; i < connectionLimit && i < (int)distances.size(); i++) {
                        connections.push_back(distances[i].second);
                    }
                }
            }

            if (!candidates.empty()) {
                currentEntry = candidates[0].second;
            }
        }

        // Update global entry point if necessary
        if (layer > topLayer) {
            topLayer = layer;
            entryPoint = docId;
        }
    }

    std::vector<std::pair<float, int>> findNearest(
        const std::vector<float>& query, int k, int searchIterations, DistanceMetric metric)
    {
        if (entryPoint == -1) return {};
        
        int currentEntry = entryPoint;
        
        // Navigate from top layer down to layer 0
        for (int currentLayer = topLayer; currentLayer > 0; currentLayer--) {
            if (currentLayer < (int)graph[currentEntry].neighbors.size()) {
                auto nearest = searchAtLayer(query, currentEntry, 1, currentLayer, metric);
                if (!nearest.empty()) {
                    currentEntry = nearest[0].second;
                }
            }
        }

        auto results = searchAtLayer(query, currentEntry, std::max(searchIterations, k), 0, metric);
        if ((int)results.size() > k) {
            results.resize(k);
        }
        return results;
    }

    void deleteDocument(int docId) {
        if (!graph.count(docId)) return;
        
        for (auto& [nodeId, node] : graph) {
            for (auto& layer : node.neighbors) {
                layer.erase(std::remove(layer.begin(), layer.end(), docId), layer.end());
            }
        }

        if (entryPoint == docId) {
            entryPoint = -1;
            topLayer = -1;
            for (auto& [nodeId, node] : graph) {
                if (nodeId != docId && node.maxLayer > topLayer) {
                    entryPoint = nodeId;
                    topLayer = node.maxLayer;
                }
            }
            if (entryPoint == -1 && graph.size() > 1) {
                for (auto& [nodeId, node] : graph) {
                    if (nodeId != docId) {
                        entryPoint = nodeId;
                        topLayer = node.maxLayer;
                        break;
                    }
                }
            }
        }

        graph.erase(docId);
    }
};

#ifdef BUILD_CLI
#include <iostream>
#include <fstream>
#include <sstream>

std::vector<float> parseVector(const std::string& str) {
    std::vector<float> vec;
    std::stringstream ss(str);
    std::string token;
    while (std::getline(ss, token, ',')) {
        try {
            vec.push_back(std::stof(token));
        } catch (...) {
        }
    }
    return vec;
}

int main(int argc, char* argv[]) {
    if (argc < 7) {
        std::cerr << "Usage: " << argv[0] << " <command> <metric> <algorithm> <k> <query_vector> <docs_file>\n";
        return 1;
    }

    std::string cmd = argv[1];
    std::string metricName = argv[2];
    std::string algoName = argv[3];
    int k = std::stoi(argv[4]);
    std::vector<float> query = parseVector(argv[5]);
    std::string docsFile = argv[6];

    std::ifstream infile(docsFile);
    if (!infile.is_open()) {
        std::cerr << "Error: Could not open documents file: " << docsFile << "\n";
        return 2;
    }

    std::vector<Document> documents;
    std::string line;
    while (std::getline(infile, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string idStr, title, embStr;
        if (std::getline(ss, idStr, '|') && std::getline(ss, title, '|') && std::getline(ss, embStr, '|')) {
            int id = std::stoi(idStr);
            std::vector<float> embedding = parseVector(embStr);
            documents.push_back(Document(id, title, embedding));
        }
    }

    auto metric = getMetricFunction(metricName);

    if (cmd == "search") {
        std::vector<std::pair<float, int>> results;
        auto startTime = std::chrono::high_resolution_clock::now();

        if (algoName == "linear") {
            LinearSearchIndex index;
            for (const auto& doc : documents) index.addDocument(doc);
            results = index.findNearest(query, k, metric);
        } else if (algoName == "bbsp") {
            int dims = query.empty() ? DEFAULT_DIMS : query.size();
            BBSPTree index(dims);
            index.rebuild(documents);
            results = index.findNearest(query, k, metric);
        } else if (algoName == "hgs") {
            HierarchicalGraphSearch index(16, 200);
            for (const auto& doc : documents) index.addDocument(doc, metric);
            results = index.findNearest(query, k, 200, metric);
        } else {
            std::cerr << "Error: Unknown algorithm: " << algoName << "\n";
            return 3;
        }

        auto endTime = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - startTime).count() / 1e6;

        std::cout << "{\n";
        std::cout << "  \"algorithm\": \"" << algoName << "\",\n";
        std::cout << "  \"metric\": \"" << metricName << "\",\n";
        std::cout << "  \"query_time_ms\": " << duration << ",\n";
        std::cout << "  \"results\": [\n";
        for (size_t i = 0; i < results.size(); ++i) {
            int docId = results[i].second;
            std::string title = "";
            for (const auto& doc : documents) {
                if (doc.id == docId) {
                    title = doc.title;
                    break;
                }
            }
            std::cout << "    {\n";
            std::cout << "      \"id\": " << docId << ",\n";
            std::cout << "      \"title\": \"" << title << "\",\n";
            std::cout << "      \"distance\": " << results[i].first << "\n";
            std::cout << "    }" << (i + 1 < results.size() ? "," : "") << "\n";
        }
        std::cout << "  ]\n";
        std::cout << "}\n";
    } else if (cmd == "benchmark") {
        auto getBuildAndQueryTime = [&](const std::string& name) {
            double buildTimeMs = 0.0;
            double queryTimeMs = 0.0;

            if (name == "linear") {
                auto t1 = std::chrono::high_resolution_clock::now();
                LinearSearchIndex index;
                for (const auto& doc : documents) index.addDocument(doc);
                auto t2 = std::chrono::high_resolution_clock::now();
                auto res = index.findNearest(query, k, metric);
                auto t3 = std::chrono::high_resolution_clock::now();
                buildTimeMs = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1e6;
                queryTimeMs = std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count() / 1e6;
            } else if (name == "bbsp") {
                int dims = query.empty() ? DEFAULT_DIMS : query.size();
                auto t1 = std::chrono::high_resolution_clock::now();
                BBSPTree index(dims);
                index.rebuild(documents);
                auto t2 = std::chrono::high_resolution_clock::now();
                auto res = index.findNearest(query, k, metric);
                auto t3 = std::chrono::high_resolution_clock::now();
                buildTimeMs = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1e6;
                queryTimeMs = std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count() / 1e6;
            } else if (name == "hgs") {
                auto t1 = std::chrono::high_resolution_clock::now();
                HierarchicalGraphSearch index(16, 200);
                for (const auto& doc : documents) index.addDocument(doc, metric);
                auto t2 = std::chrono::high_resolution_clock::now();
                auto res = index.findNearest(query, k, 200, metric);
                auto t3 = std::chrono::high_resolution_clock::now();
                buildTimeMs = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1e6;
                queryTimeMs = std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count() / 1e6;
            }

            return std::make_pair(buildTimeMs, queryTimeMs);
        };

        auto [linearBuild, linearQuery] = getBuildAndQueryTime("linear");
        auto [bbspBuild, bbspQuery] = getBuildAndQueryTime("bbsp");
        auto [hgsBuild, hgsQuery] = getBuildAndQueryTime("hgs");

        std::cout << "{\n";
        std::cout << "  \"linear\": { \"build_time_ms\": " << linearBuild << ", \"query_time_ms\": " << linearQuery << " },\n";
        std::cout << "  \"bbsp\": { \"build_time_ms\": " << bbspBuild << ", \"query_time_ms\": " << bbspQuery << " },\n";
        std::cout << "  \"hgs\": { \"build_time_ms\": " << hgsBuild << ", \"query_time_ms\": " << hgsQuery << " }\n";
        std::cout << "}\n";
    }

    return 0;
}
#endif
