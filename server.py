import http.server
import socketserver
import json
import os
import subprocess
import time
import math
import random
import urllib.parse
import sys
import heapq

PORT = 8080
EXE_NAME = "./engine.exe" if os.name == 'nt' else "./engine"
TEMP_DOCS_FILE = "temp_docs.txt"

# State variables
has_cpp_engine = False
active_dataset = []

# Distance Metrics in Python
def cosine_distance(v1, v2):
    dot_product = 0.0
    mag1 = 0.0
    mag2 = 0.0
    for x, y in zip(v1, v2):
        dot_product += x * y
        mag1 += x * x
        mag2 += y * y
    if mag1 < 1e-9 or mag2 < 1e-9:
        return 1.0
    val = dot_product / (math.sqrt(mag1) * math.sqrt(mag2))
    # Clamp to prevent domain errors
    val = max(-1.0, min(1.0, val))
    return 1.0 - val

def euclidean_distance(v1, v2):
    dist = 0.0
    for x, y in zip(v1, v2):
        dist += (x - y) ** 2
    return math.sqrt(dist)

def manhattan_distance(v1, v2):
    dist = 0.0
    for x, y in zip(v1, v2):
        dist += abs(x - y)
    return dist

def get_metric_function(name):
    if name == "cosine":
        return cosine_distance
    elif name == "manhattan":
        return manhattan_distance
    else:
        return euclidean_distance

# ----------------------------------------------------
# 1. LINEAR SEARCH INDEX (Python implementation)
# ----------------------------------------------------
class PythonLinearSearch:
    def __init__(self, documents):
        self.documents = documents

    def find_nearest(self, query, k, metric_func):
        results = []
        for doc in self.documents:
            dist = metric_func(query, doc['embedding'])
            results.append((dist, doc['id'], doc['title']))
        results.sort(key=lambda x: x[0])
        return results[:k]

# ----------------------------------------------------
# 2. BALANCED BINARY SPACE PARTITIONING TREE (Python implementation)
# ----------------------------------------------------
class PythonKDNode:
    def __init__(self, doc, left=None, right=None):
        self.doc = doc
        self.left = left
        self.right = right

class PythonKDTree:
    def __init__(self, documents, dimensions):
        self.dimensions = dimensions
        # Copy to avoid side effects during sorting
        docs_copy = list(documents)
        self.root = self._build(docs_copy, 0)

    def _build(self, docs, depth):
        if not docs:
            return None
        axis = depth % self.dimensions
        docs.sort(key=lambda d: d['embedding'][axis])
        mid = len(docs) // 2
        node = PythonKDNode(docs[mid])
        node.left = self._build(docs[:mid], depth + 1)
        node.right = self._build(docs[mid+1:], depth + 1)
        return node

    def find_nearest(self, query, k, metric_func):
        candidates = [] # max heap of (-dist, id, title)

        def search(node, depth):
            if not node:
                return
            
            dist = metric_func(query, node.doc['embedding'])
            if len(candidates) < k:
                heapq.heappush(candidates, (-dist, node.doc['id'], node.doc['title']))
            elif dist < -candidates[0][0]:
                heapq.heapreplace(candidates, (-dist, node.doc['id'], node.doc['title']))

            axis = depth % self.dimensions
            diff = query[axis] - node.doc['embedding'][axis]

            primary = node.left if diff < 0 else node.right
            secondary = node.right if diff < 0 else node.left

            search(primary, depth + 1)

            # We can only prune coordinate differences for Euclidean/Manhattan.
            # For Cosine distance, we must traverse both branches as the split axis value doesn't bound cosine distance.
            is_cosine = (metric_func == cosine_distance)
            if len(candidates) < k or is_cosine or abs(diff) < -candidates[0][0]:
                search(secondary, depth + 1)

        search(self.root, 0)
        res = []
        while candidates:
            neg_dist, doc_id, title = heapq.heappop(candidates)
            res.append(( -neg_dist, doc_id, title ))
        res.sort(key=lambda x: x[0])
        return res

# ----------------------------------------------------
# 3. HIERARCHICAL GRAPH SEARCH / HNSW (Python implementation)
# ----------------------------------------------------
class PythonHGSNode:
    def __init__(self, doc, max_layer):
        self.doc = doc
        self.max_layer = max_layer
        self.neighbors = [[] for _ in range(max_layer + 1)] # neighbors[layer] = list of doc_ids

class PythonHGSIndex:
    def __init__(self, dimensions, max_connections=16, construction_iterations=200):
        self.dimensions = dimensions
        self.max_connections = max_connections
        self.construction_iterations = construction_iterations
        self.layer_probability = 1.0 / math.log(max_connections)
        self.graph = {}
        self.entry_point = None
        self.top_layer = -1
        self.random = random.Random(42)

    def _generate_random_layer(self):
        r = self.random.random()
        if r == 0:
            r = 1e-9
        return int(math.floor(-math.log(r) * self.layer_probability))

    def add_document(self, doc, metric_func):
        doc_id = doc['id']
        layer = self._generate_random_layer()
        node = PythonHGSNode(doc, layer)
        self.graph[doc_id] = node

        if self.entry_point is None:
            self.entry_point = doc_id
            self.top_layer = layer
            return

        current_entry = self.entry_point
        # Navigate from top_layer down to layer
        for curr_layer in range(self.top_layer, layer, -1):
            nearest = self._search_at_layer(doc['embedding'], current_entry, 1, curr_layer, metric_func)
            if nearest:
                current_entry = nearest[0][1]

        # Insert into layers from min(top_layer, layer) down to 0
        for curr_layer in range(min(self.top_layer, layer), -1, -1):
            candidates = self._search_at_layer(doc['embedding'], current_entry, self.construction_iterations, curr_layer, metric_func)
            conn_limit = self.max_connections * 2 if curr_layer == 0 else self.max_connections

            selected = [c[1] for c in candidates[:conn_limit]]
            node.neighbors[curr_layer] = selected

            # Update reciprocal connections
            for neighbor_id in selected:
                neighbor = self.graph[neighbor_id]
                if len(neighbor.neighbors) <= curr_layer:
                    neighbor.neighbors.extend([[] for _ in range(curr_layer + 1 - len(neighbor.neighbors))])

                neighbor.neighbors[curr_layer].append(doc_id)

                # Prune if necessary
                if len(neighbor.neighbors[curr_layer]) > conn_limit:
                    dists = []
                    for conn_id in neighbor.neighbors[curr_layer]:
                        dists.append((metric_func(neighbor.doc['embedding'], self.graph[conn_id].doc['embedding']), conn_id))
                    dists.sort(key=lambda x: x[0])
                    neighbor.neighbors[curr_layer] = [d[1] for d in dists[:conn_limit]]

            if candidates:
                current_entry = candidates[0][1]

        if layer > self.top_layer:
            self.top_layer = layer
            self.entry_point = doc_id

    def _search_at_layer(self, query, entry_node, num_candidates, layer, metric_func):
        visited = {entry_node}
        candidates_heap = [(metric_func(query, self.graph[entry_node].doc['embedding']), entry_node)]
        results_heap = [(-candidates_heap[0][0], entry_node)]

        while candidates_heap:
            cand_dist, cand_id = heapq.heappop(candidates_heap)

            if len(results_heap) >= num_candidates and cand_dist > -results_heap[0][0]:
                break

            node = self.graph[cand_id]
            if layer >= len(node.neighbors):
                continue

            for neighbor_id in node.neighbors[layer]:
                if neighbor_id in visited or neighbor_id not in self.graph:
                    continue
                visited.add(neighbor_id)

                dist = metric_func(query, self.graph[neighbor_id].doc['embedding'])

                if len(results_heap) < num_candidates or dist < -results_heap[0][0]:
                    heapq.heappush(candidates_heap, (dist, neighbor_id))
                    heapq.heappush(results_heap, (-dist, neighbor_id))
                    if len(results_heap) > num_candidates:
                        heapq.heappop(results_heap)

        out = []
        for neg_dist, node_id in results_heap:
            out.append((-neg_dist, node_id))
        out.sort(key=lambda x: x[0])
        return out

    def find_nearest(self, query, k, search_iterations, metric_func):
        if self.entry_point is None:
            return []
        current_entry = self.entry_point
        for curr_layer in range(self.top_layer, 0, -1):
            nearest = self._search_at_layer(query, current_entry, 1, curr_layer, metric_func)
            if nearest:
                current_entry = nearest[0][1]

        results = self._search_at_layer(query, current_entry, max(search_iterations, k), 0, metric_func)
        return results[:k]


# ----------------------------------------------------
# 4. DATASET INITIALIZATION (20 semantically clustered docs)
# ----------------------------------------------------
def normalize_vector(v):
    mag = math.sqrt(sum(x*x for x in v))
    if mag < 1e-9:
        return [0.0] * len(v)
    return [x / mag for x in v]

def generate_default_dataset():
    # 4 clusters in 16D:
    # Dim 0-3: AI/ML
    # Dim 4-7: Space Exploration
    # Dim 8-11: Culinary Arts
    # Dim 12-15: Finance / Business
    topics = [
        # AI/ML
        {"id": 1, "title": "Introduction to Neural Networks", "text": "Basic structures of artificial neurons and weights.", "cluster": 0},
        {"id": 2, "title": "Deep Learning Frameworks compared", "text": "Analyzing PyTorch vs TensorFlow vs JAX for deep modeling.", "cluster": 0},
        {"id": 3, "title": "Natural Language Processing Basics", "text": "Tokenization, embeddings, and Transformer architectures.", "cluster": 0},
        {"id": 4, "title": "Computer Vision and Image Recognition", "text": "CNNs, ResNets, and visual object detection methods.", "cluster": 0},
        {"id": 5, "title": "Reinforcement Learning in Robotics", "text": "Q-learning and policy gradients for robot controllers.", "cluster": 0},
        # Space Exploration
        {"id": 6, "title": "The Hubble Space Telescope Legacy", "text": "Three decades of deep cosmic images and discoveries.", "cluster": 1},
        {"id": 7, "title": "Mars Rover Exploration Missions", "text": "Seeking biosignatures and geological history on Mars.", "cluster": 1},
        {"id": 8, "title": "Search for Exoplanets in Galaxy", "text": "Kepler and TESS finding habitable transit systems.", "cluster": 1},
        {"id": 9, "title": "James Webb Space Telescope Discoveries", "text": "Infrared spectroscopy of the early universe and atmospheres.", "cluster": 1},
        {"id": 10, "title": "The Physics of Black Holes", "text": "Event horizons, singularity, and Hawking radiation theory.", "cluster": 1},
        # Cooking
        {"id": 11, "title": "French Pastry Baking Techniques", "text": "Perfecting croissants, puff pastry, and choux dough.", "cluster": 2},
        {"id": 12, "title": "Mastering the Art of Italian Pasta", "text": "Fresh semolina dough, rolling techniques, and sauces.", "cluster": 2},
        {"id": 13, "title": "Spices of India and Culinary Heritage", "text": "Roasting and blending complex aromatics and masalas.", "cluster": 2},
        {"id": 14, "title": "Modern Gastronomy and Food Science", "text": "Spherification, gels, and temperature-controlled cooking.", "cluster": 2},
        {"id": 15, "title": "Perfect Sourdough Bread Recipe", "text": "Wild yeast fermentation, hydration rates, and crust control.", "cluster": 2},
        # Finance
        {"id": 16, "title": "Global Stock Markets Analysis", "text": "Analyzing equities, bond yields, and economic indicators.", "cluster": 3},
        {"id": 17, "title": "Understanding Cryptocurrency and Blockchain", "text": "Decentralized ledgers, smart contracts, and Web3 assets.", "cluster": 3},
        {"id": 18, "title": "Personal Finance and Retirement Planning", "text": "Compounding interest, index funds, and tax-sheltered accounts.", "cluster": 3},
        {"id": 19, "title": "The Role of Central Banks in Economy", "text": "Monetary policy, interest rates, and inflation targeting.", "cluster": 3},
        {"id": 20, "title": "Venture Capital and Startup Funding", "text": "Pre-seed valuation, term sheets, and dilution math.", "cluster": 3}
    ]

    random.seed(42)
    dataset = []
    for t in topics:
        v = [0.05 * random.uniform(-1, 1) for _ in range(16)]
        
        # Boost cluster dims
        c = t["cluster"]
        for d in range(c * 4, (c + 1) * 4):
            v[d] += random.uniform(0.6, 1.0)
            
        normalized_emb = normalize_vector(v)
        dataset.append({
            "id": t["id"],
            "title": t["title"],
            "text": t["text"],
            "embedding": normalized_emb
        })
    return dataset

# ----------------------------------------------------
# 5. SUBPROCESS RUNNER FOR C++ ENGINE
# ----------------------------------------------------
def write_dataset_to_temp_file(docs, filename=TEMP_DOCS_FILE):
    with open(filename, "w", encoding="utf-8") as f:
        for doc in docs:
            emb_str = ",".join(f"{x:.6f}" for x in doc["embedding"])
            f.write(f"{doc['id']}|{doc['title']}|{emb_str}\n")

def run_cpp_engine(command, metric, algo, k, query_vec, docs_file=TEMP_DOCS_FILE):
    query_str = ",".join(f"{x:.6f}" for x in query_vec)
    cmd = [EXE_NAME, command, metric, algo, str(k), query_str, docs_file]
    
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=10.0)
        if res.returncode == 0:
            return json.loads(res.stdout)
        else:
            print(f"C++ Engine Error (exit {res.returncode}):", res.stderr)
            return None
    except Exception as e:
        print("Exception executing C++ Engine subprocess:", e)
        return None

# ----------------------------------------------------
# 6. HTTP REQUEST HANDLER
# ----------------------------------------------------
class RequestHandler(http.server.BaseHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path

        if path == "/" or path == "/index.html" or path == "/dashboard.html":
            self.serve_file("dashboard.html", "text/html")
        elif path == "/config.json":
            self.serve_file("config.json", "application/json")
        elif path == "/api/stats":
            self.handle_api_stats()
        else:
            self.send_error(404, "File Not Found")

    def do_POST(self):
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path

        # Read JSON body
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length)
        
        try:
            body = json.loads(post_data) if post_data else {}
        except json.JSONDecodeError:
            self.send_error_json(400, "Invalid JSON payload")
            return

        if path == "/api/search":
            self.handle_api_search(body)
        elif path == "/api/benchmark":
            self.handle_api_benchmark(body)
        elif path == "/api/add_document":
            self.handle_api_add_document(body)
        elif path == "/api/delete_document":
            self.handle_api_delete_document(body)
        elif path == "/api/reset_dataset":
            self.handle_api_reset_dataset()
        else:
            self.send_error_json(404, "API endpoint not found")

    def serve_file(self, filepath, content_type):
        if not os.path.exists(filepath):
            self.send_error(404, f"File {filepath} not found")
            return
        
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content.encode('utf-8'))))
            self.end_headers()
            self.wfile.write(content.encode('utf-8'))
        except Exception as e:
            self.send_error(500, f"Error reading file: {e}")

    def send_error_json(self, status_code, message):
        response = json.dumps({"error": message}).encode('utf-8')
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def send_success_json(self, data):
        response = json.dumps(data).encode('utf-8')
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    # API Handlers
    def handle_api_stats(self):
        global has_cpp_engine, active_dataset
        dim = len(active_dataset[0]["embedding"]) if active_dataset else 16
        self.send_success_json({
            "vector_count": len(active_dataset),
            "dimensions": dim,
            "has_cpp_engine": has_cpp_engine,
            "engine_mode": "C++ Compiled Executable" if has_cpp_engine else "Pure Python Fallback"
        })

    def handle_api_search(self, body):
        global has_cpp_engine, active_dataset
        
        # 1. Parse & Validate input parameters
        query_vector_raw = body.get("query_vector")
        k_raw = body.get("k", 5)
        metric = body.get("metric", "cosine")

        if query_vector_raw is None:
            self.send_error_json(400, "Query vector is required")
            return
        
        # Validate query_vector string / list
        if isinstance(query_vector_raw, str):
            try:
                query_vector = [float(x.strip()) for x in query_vector_raw.split(",") if x.strip()]
            except ValueError:
                self.send_error_json(400, "Invalid query vector: elements must be floats")
                return
        elif isinstance(query_vector_raw, list):
            try:
                query_vector = [float(x) for x in query_vector_raw]
            except ValueError:
                self.send_error_json(400, "Invalid query vector: elements must be floats")
                return
        else:
            self.send_error_json(400, "Query vector must be a comma-separated string or an array of numbers")
            return

        if not active_dataset:
            self.send_error_json(400, "Database is empty")
            return

        # Check dimension match
        expected_dim = len(active_dataset[0]["embedding"])
        if len(query_vector) != expected_dim:
            self.send_error_json(400, f"Dimension mismatch: query has {len(query_vector)} dims, but database requires {expected_dim}")
            return

        # Check for NaN / Infinity
        if any(math.isnan(x) or math.isinf(x) for x in query_vector):
            self.send_error_json(400, "Query vector contains invalid numbers (NaN or Infinity)")
            return

        # Validate k bounds
        try:
            k = int(k_raw)
        except (ValueError, TypeError):
            self.send_error_json(400, "k must be an integer")
            return

        if k <= 0:
            self.send_error_json(400, "k must be a positive integer greater than zero")
            return
        
        if k > len(active_dataset):
            self.send_error_json(400, f"k ({k}) cannot be greater than dataset size ({len(active_dataset)})")
            return

        # Validate metric
        if metric not in ["cosine", "euclidean", "manhattan"]:
            self.send_error_json(400, f"Unknown distance metric: {metric}")
            return

        # 2. Run execution (C++ vs Python Fallback)
        results_payload = {}
        used_cpp = False

        if has_cpp_engine:
            write_dataset_to_temp_file(active_dataset)
            
            # Executing all three search methods in C++
            cpp_linear = run_cpp_engine("search", metric, "linear", k, query_vector)
            cpp_bbsp = run_cpp_engine("search", metric, "bbsp", k, query_vector)
            cpp_hgs = run_cpp_engine("search", metric, "hgs", k, query_vector)
            
            if cpp_linear and cpp_bbsp and cpp_hgs:
                results_payload = {
                    "linear": cpp_linear,
                    "bbsp": cpp_bbsp,
                    "hgs": cpp_hgs
                }
                used_cpp = True

        if not used_cpp:
            # Fall back to Pure Python implementation
            metric_func = get_metric_function(metric)
            
            # --- Linear search
            t_start = time.perf_counter_ns()
            py_linear_idx = PythonLinearSearch(active_dataset)
            linear_res = py_linear_idx.find_nearest(query_vector, k, metric_func)
            t_linear = (time.perf_counter_ns() - t_start) / 1e6 # ms
            
            # --- BBSP Tree (KD Tree)
            t_start = time.perf_counter_ns()
            py_kd_idx = PythonKDTree(active_dataset, expected_dim)
            kd_res = py_kd_idx.find_nearest(query_vector, k, metric_func)
            t_kd = (time.perf_counter_ns() - t_start) / 1e6 # ms
            
            # --- HGS (HNSW)
            t_start = time.perf_counter_ns()
            py_hgs_idx = PythonHGSIndex(expected_dim, max_connections=16, construction_iterations=200)
            for doc in active_dataset:
                py_hgs_idx.add_document(doc, metric_func)
            hgs_res = py_hgs_idx.find_nearest(query_vector, k, 200, metric_func)
            t_hgs = (time.perf_counter_ns() - t_start) / 1e6 # ms

            # Format the output matching C++ JSON outputs
            doc_map = {d["id"]: d["title"] for d in active_dataset}
            results_payload = {
                "linear": {
                    "algorithm": "linear",
                    "metric": metric,
                    "query_time_ms": t_linear,
                    "results": [{"id": r[1], "title": r[2], "distance": r[0]} for r in linear_res]
                },
                "bbsp": {
                    "algorithm": "bbsp",
                    "metric": metric,
                    "query_time_ms": t_kd,
                    "results": [{"id": r[1], "title": r[2], "distance": r[0]} for r in kd_res]
                },
                "hgs": {
                    "algorithm": "hgs",
                    "metric": metric,
                    "query_time_ms": t_hgs,
                    "results": [{"id": r[1], "title": doc_map.get(r[1], "Unknown Document"), "distance": r[0]} for r in hgs_res]
                }
            }

        response_data = {
            "engine": "C++ Compiled" if used_cpp else "Python Fallback",
            "search_data": results_payload
        }
        self.send_success_json(response_data)

    def handle_api_benchmark(self, body):
        global has_cpp_engine
        
        size = body.get("size", "medium")
        metric = body.get("metric", "cosine")
        
        # Map dataset sizing
        if size == "small":
            vector_count = 100
            dim = 16
        elif size == "large":
            vector_count = 5000  # Cap at 5000 to keep responses fast on standard hardware
            dim = 16
        else:
            vector_count = 1000
            dim = 16

        # Generate random benchmark dataset
        benchmark_docs = []
        random.seed(1337)
        for i in range(vector_count):
            emb = normalize_vector([random.gauss(0, 1) for _ in range(dim)])
            benchmark_docs.append({
                "id": i + 1,
                "title": f"Bench Document {i + 1}",
                "embedding": emb
            })

        query_vector = normalize_vector([random.gauss(0, 1) for _ in range(dim)])
        k = 10
        
        used_cpp = False
        benchmark_payload = {}

        if has_cpp_engine:
            # Write benchmark dataset to temp file
            write_dataset_to_temp_file(benchmark_docs, "temp_bench.txt")
            cpp_res = run_cpp_engine("benchmark", metric, "linear", k, query_vector, "temp_bench.txt")
            
            # Clean up temp file
            try:
                os.remove("temp_bench.txt")
            except:
                pass
                
            if cpp_res:
                benchmark_payload = cpp_res
                # In C++ benchmark, let's also compute mock recall based on HGS returning same doc IDs as linear
                # Real recall is calculated below
                used_cpp = True

        if not used_cpp:
            # Fall back to pure Python benchmarking
            metric_func = get_metric_function(metric)
            
            # 1. Linear Benchmark
            t0 = time.perf_counter_ns()
            py_linear = PythonLinearSearch(benchmark_docs)
            t1 = time.perf_counter_ns()
            linear_results = py_linear.find_nearest(query_vector, k, metric_func)
            t2 = time.perf_counter_ns()
            
            linear_build = (t1 - t0) / 1e6
            linear_query = (t2 - t1) / 1e6

            # 2. BBSP / KD Tree Benchmark
            t0 = time.perf_counter_ns()
            py_kd = PythonKDTree(benchmark_docs, dim)
            t1 = time.perf_counter_ns()
            kd_results = py_kd.find_nearest(query_vector, k, metric_func)
            t2 = time.perf_counter_ns()
            
            kd_build = (t1 - t0) / 1e6
            kd_query = (t2 - t1) / 1e6

            # 3. HGS / HNSW Benchmark
            t0 = time.perf_counter_ns()
            py_hgs = PythonHGSIndex(dim, max_connections=16, construction_iterations=200)
            for doc in benchmark_docs:
                py_hgs.add_document(doc, metric_func)
            t1 = time.perf_counter_ns()
            hgs_results = py_hgs.find_nearest(query_vector, k, 200, metric_func)
            t2 = time.perf_counter_ns()
            
            hgs_build = (t1 - t0) / 1e6
            hgs_query = (t2 - t1) / 1e6

            benchmark_payload = {
                "linear": { "build_time_ms": linear_build, "query_time_ms": linear_query, "results": linear_results },
                "bbsp": { "build_time_ms": kd_build, "query_time_ms": kd_query, "results": kd_results },
                "hgs": { "build_time_ms": hgs_build, "query_time_ms": hgs_query, "results": hgs_results }
            }

        # Calculate Recall Accuracy against Linear Search
        # To do this, we need the result IDs for all algorithms.
        # If we used C++ execution, we can query both from C++ search output or calculate here.
        # Since the search command in C++ outputs the results, let's look at the document IDs returned.
        # If we got C++ results, we'll run a single query through C++ to verify IDs and calculate recall.
        
        linear_ids = set()
        bbsp_ids = set()
        hgs_ids = set()

        if used_cpp:
            # Let's perform a fast C++ search call to get the exact result IDs
            write_dataset_to_temp_file(benchmark_docs, "temp_bench.txt")
            l_search = run_cpp_engine("search", metric, "linear", k, query_vector, "temp_bench.txt")
            b_search = run_cpp_engine("search", metric, "bbsp", k, query_vector, "temp_bench.txt")
            h_search = run_cpp_engine("search", metric, "hgs", k, query_vector, "temp_bench.txt")
            try:
                os.remove("temp_bench.txt")
            except:
                pass

            if l_search and b_search and h_search:
                linear_ids = {r["id"] for r in l_search["results"]}
                bbsp_ids = {r["id"] for r in b_search["results"]}
                hgs_ids = {r["id"] for r in h_search["results"]}
        else:
            # Python results
            linear_ids = {r[1] for r in benchmark_payload["linear"]["results"]}
            bbsp_ids = {r[1] for r in benchmark_payload["bbsp"]["results"]}
            hgs_ids = {r[1] for r in benchmark_payload["hgs"]["results"]}

        # Compute recall percentage
        bbsp_recall = (len(linear_ids.intersection(bbsp_ids)) / k) * 100 if linear_ids else 100.0
        hgs_recall = (len(linear_ids.intersection(hgs_ids)) / k) * 100 if linear_ids else 100.0

        # Build response object
        response_data = {
            "engine": "C++ Compiled" if used_cpp else "Python Fallback",
            "size": size,
            "vector_count": vector_count,
            "dimensions": dim,
            "metrics": {
                "linear": {
                    "build_time_ms": benchmark_payload["linear"]["build_time_ms"],
                    "query_time_ms": benchmark_payload["linear"]["query_time_ms"],
                    "qps": int(1000.0 / benchmark_payload["linear"]["query_time_ms"]) if benchmark_payload["linear"]["query_time_ms"] > 0 else 9999,
                    "recall": 100.0
                },
                "bbsp": {
                    "build_time_ms": benchmark_payload["bbsp"]["build_time_ms"],
                    "query_time_ms": benchmark_payload["bbsp"]["query_time_ms"],
                    "qps": int(1000.0 / benchmark_payload["bbsp"]["query_time_ms"]) if benchmark_payload["bbsp"]["query_time_ms"] > 0 else 9999,
                    "recall": bbsp_recall
                },
                "hgs": {
                    "build_time_ms": benchmark_payload["hgs"]["build_time_ms"],
                    "query_time_ms": benchmark_payload["hgs"]["query_time_ms"],
                    "qps": int(1000.0 / benchmark_payload["hgs"]["query_time_ms"]) if benchmark_payload["hgs"]["query_time_ms"] > 0 else 9999,
                    "recall": hgs_recall
                }
            }
        }
        self.send_success_json(response_data)

    def handle_api_add_document(self, body):
        global active_dataset
        doc_id = body.get("id")
        title = body.get("title")
        text = body.get("text", "")
        embedding_raw = body.get("embedding")

        if doc_id is None or not title or embedding_raw is None:
            self.send_error_json(400, "Missing required fields (id, title, embedding)")
            return

        try:
            doc_id = int(doc_id)
        except ValueError:
            self.send_error_json(400, "id must be an integer")
            return

        # Check duplicates
        if any(d["id"] == doc_id for d in active_dataset):
            self.send_error_json(400, f"Document ID {doc_id} already exists")
            return

        # Parse vector
        if isinstance(embedding_raw, str):
            try:
                embedding = [float(x.strip()) for x in embedding_raw.split(",") if x.strip()]
            except ValueError:
                self.send_error_json(400, "Invalid embedding vector format")
                return
        elif isinstance(embedding_raw, list):
            try:
                embedding = [float(x) for x in embedding_raw]
            except ValueError:
                self.send_error_json(400, "Invalid embedding vector: elements must be floats")
                return
        else:
            self.send_error_json(400, "Embedding must be array or comma-separated string")
            return

        # Check dimension matches active dataset
        if active_dataset:
            expected_dim = len(active_dataset[0]["embedding"])
            if len(embedding) != expected_dim:
                self.send_error_json(400, f"Dimension mismatch: expected {expected_dim} features, received {len(embedding)}")
                return
        
        # Verify Nan/Inf
        if any(math.isnan(x) or math.isinf(x) for x in embedding):
            self.send_error_json(400, "Embedding contains invalid values (NaN or Infinity)")
            return

        # Add to active dataset
        active_dataset.append({
            "id": doc_id,
            "title": title,
            "text": text,
            "embedding": normalize_vector(embedding)
        })
        self.send_success_json({"message": f"Document {doc_id} successfully added", "total_count": len(active_dataset)})

    def handle_api_delete_document(self, body):
        global active_dataset
        doc_id_raw = body.get("id")

        if doc_id_raw is None:
            self.send_error_json(400, "Document id is required")
            return

        try:
            doc_id = int(doc_id_raw)
        except ValueError:
            self.send_error_json(400, "id must be an integer")
            return

        # Search and remove
        initial_len = len(active_dataset)
        active_dataset = [d for d in active_dataset if d["id"] != doc_id]

        if len(active_dataset) == initial_len:
            self.send_error_json(404, f"Document with ID {doc_id} not found")
        else:
            self.send_success_json({"message": f"Document {doc_id} successfully deleted", "total_count": len(active_dataset)})

    def handle_api_reset_dataset(self):
        global active_dataset
        active_dataset = generate_default_dataset()
        self.send_success_json({"message": "Database reset to 20 default clustered vectors", "total_count": len(active_dataset)})


# ----------------------------------------------------
# 7. SERVER LAUNCH & INITIALIZATION
# ----------------------------------------------------
def try_compile_cpp_engine():
    global has_cpp_engine
    print("Checking for C++ compiler and attempting to compile engine.cpp...")
    
    # Check WinGet MinGW location and add it to PATH if present
    winget_gcc_path = r"C:\Users\ANIKET\AppData\Local\Microsoft\WinGet\Packages\BrechtSanders.WinLibs.MCF.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe\mingw64\bin"
    if os.path.exists(winget_gcc_path):
        if winget_gcc_path not in os.environ["PATH"]:
            os.environ["PATH"] = winget_gcc_path + os.pathsep + os.environ["PATH"]
            print(f"[INFO] Found WinGet MinGW compiler path. Appended to PATH.")
            
    try:
        # Check if engine.cpp exists
        if not os.path.exists("engine.cpp"):
            print("Error: engine.cpp not found in current directory.")
            return

        compile_cmd = ["g++", "-std=c++17", "-O2", "-DBUILD_CLI", "engine.cpp", "-o", EXE_NAME]
        
        # Try compiling
        result = subprocess.run(compile_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            has_cpp_engine = True
            print(f"[OK] C++ Engine successfully compiled as '{EXE_NAME}' with CLI enabled.")
        else:
            print("[ERROR] Compilation of C++ engine failed. Error details:")
            print(result.stderr)
            print("Continuing in pure Python fallback mode.")
    except FileNotFoundError:
        print("[INFO] 'g++' compiler was not found on system PATH. Web app will run using the pure Python fallback engine.")
    except Exception as e:
        print(f"[ERROR] Unexpected compiler error: {e}. Running in pure Python fallback mode.")

def main():
    global active_dataset
    # Load dataset
    active_dataset = generate_default_dataset()
    
    # Try compiling C++ engine
    try_compile_cpp_engine()
    
    # Start web server
    handler = RequestHandler
    # Set thread/socket reuse option
    socketserver.TCPServer.allow_reuse_address = True
    
    try:
        with socketserver.TCPServer(("", PORT), handler) as httpd:
            print(f"\n=======================================================")
            print(f"[STARTING] Semantic Search Engine comparator is live at: http://localhost:{PORT}")
            print(f"[MODE] Engine execution mode: {'C++ HIGH-PERFORMANCE' if has_cpp_engine else 'PURE PYTHON FALLBACK'}")
            print(f"[DATA] Local database initialized with {len(active_dataset)} clustered vectors.")
            print(f"=======================================================\n")
            print("Press Ctrl+C to terminate.")
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        # Cleanup temp file
        if os.path.exists(TEMP_DOCS_FILE):
            try:
                os.remove(TEMP_DOCS_FILE)
            except:
                pass
        sys.exit(0)
    except Exception as e:
        print(f"Server startup failed: {e}")

if __name__ == '__main__':
    main()
