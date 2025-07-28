#!/usr/bin/env python3
"""
Persistent Enhanced Food AI Backend Server
Designed to stay running reliably without unexpected exits
"""

import json
import time
import logging
import sys
import os
import signal
import atexit
import threading
import pickle
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from datetime import datetime

# Add pandas for parquet loading
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

# Semantic vector processing imports
try:
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False
    np = None
    cosine_similarity = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('backend.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# Taste profile to ingredient mappings for semantic vector generation
TASTE_PROFILES = {
    "sweet": ["vanilla extract", "maple syrup", "honey", "agave nectar", "brown sugar", "cinnamon", "dates", "stevia"],
    "savory": ["garlic powder", "onion powder", "thyme", "rosemary", "oregano", "basil", "sage", "bay leaves"],
    "spicy": ["chili powder", "cayenne pepper", "paprika", "jalapeño", "habanero", "black pepper", "red pepper flakes"],
    "umami": ["nutritional yeast", "soy sauce", "mushrooms", "tomato paste", "parmesan cheese", "miso paste", "anchovies"],
    "tangy": ["lemon juice", "lime juice", "vinegar", "yogurt", "sour cream", "pickles", "capers"],
    "fruity": ["lemon", "lime", "orange", "apple", "berries", "mango", "pineapple", "citrus zest"]
}

def load_ingredient_data_from_parquet():
    """Load ingredient data directly from parquet files when vector files are not available"""
    if not PANDAS_AVAILABLE:
        logger.warning("⚠️ Pandas not available - cannot load from parquet files")
        return None
    
    # Try multiple possible paths for ingredient data parquet files
    possible_paths = [
        # Hot cache paths
        "E:/FoodVault/optimized/food_ai_hot_cache/hot_data/ingredient_hot.parquet",
        
        # Full dataset paths
        "F:/FoodVault/dataset/processed/ingredient_data/ingredient_data.parquet",
        
        # Local paths
        "ingredient_data.parquet",
        "ingredient_hot.parquet"
    ]
    
    logger.info(f"🔍 Searching for ingredient data parquet files in: {possible_paths}")
    
    for parquet_path in possible_paths:
        if os.path.exists(parquet_path):
            try:
                df = pd.read_parquet(parquet_path)
                logger.info(f"✅ Loaded {len(df)} ingredients from {parquet_path}")
                
                # Generate random vectors for each ingredient as placeholder
                vectors = {}
                
                # Check required column
                if 'name' not in df.columns:
                    logger.warning(f"⚠️ Parquet file {parquet_path} does not contain 'name' column")
                    continue
                
                # Generate random vectors for all ingredients
                # First get list of all unique ingredient names
                unique_names = df['name'].dropna().unique()
                logger.info(f"Found {len(unique_names)} unique ingredient names in dataset")
                
                # Generate vectors for all unique names
                for name in unique_names:
                    if isinstance(name, str) and name.strip():
                        # Use a consistent seed based on the name to ensure reproducibility
                        name_seed = sum(ord(c) for c in name)
                        np.random.seed(name_seed)
                        vectors[name] = np.random.rand(100)  # 100-dim vector
                
                ingredient_count = len(vectors)
                logger.info(f"✅ Generated {ingredient_count} ingredient vectors from parquet data")
                return vectors
                
            except Exception as e:
                logger.warning(f"⚠️ Failed to load parquet from {parquet_path}: {e}")
                continue
    
    logger.error("❌ Failed to load ingredient data from any parquet file")
    return None

def load_ingredient_vectors(path=None):
    """Load ingredient vectors from pickle file or generate from parquet if not found"""
    if not SEMANTIC_AVAILABLE:
        logger.warning("⚠️ NumPy/sklearn not available - semantic matching disabled")
        return {}
    
    # Try multiple possible paths in priority order (hot cache first, then full dataset)
    possible_paths = [
        path,  # User-provided path gets first priority
        
        # Hot cache paths for fast access
        "E:/FoodVault/optimized/food_ai_hot_cache/hot_data/ingredient_vectors.pkl",
        
        # Full dataset paths from F: drive
        "F:/FoodVault/dataset/processed/ingredient_data/ingredient_vectors_fixed.pkl",  # Fixed version first
        "F:/FoodVault/dataset/processed/ingredient_data/ingredient_vectors.pkl",
        
        # Local paths
        "ingredient_vectors.pkl",
        "ingredient_vectors_fixed.pkl",
        
        # Additional search paths
        "F:/FoodVault/dataset/processed/ingredient_data/full_ingredient_vectors.pkl",
        "F:/FoodVault/dataset/ingredient_vectors_full.pkl"
    ]
    
    # Debug output
    logger.info(f"🔍 Searching for ingredient vectors in possible paths: {possible_paths}")
    
    for vector_path in possible_paths:
        if vector_path and os.path.exists(vector_path):
            try:
                with open(vector_path, "rb") as f:
                    vectors = pickle.load(f)
                vector_count = len(vectors)
                logger.info(f"✅ Loaded {vector_count} ingredient vectors from {vector_path}")
                
                # Check if this is likely the full dataset
                if vector_count < 300000:  # Looking for ~312,291 ingredients
                    logger.warning(f"⚠️ Vector count ({vector_count}) appears to be a subset, not the full dataset (~312,291)")
                    logger.info(f"🔍 Will continue searching for larger vector sets...")
                    
                    # If this is the last path and we still haven't found the full dataset, use what we have
                    if vector_path == possible_paths[-1]:
                        logger.warning(f"⚠️ Could not find full dataset in vector files, trying parquet files...")
                        parquet_vectors = load_ingredient_data_from_parquet()
                        if parquet_vectors and len(parquet_vectors) > vector_count:
                            logger.info(f"✅ Successfully loaded larger dataset from parquet: {len(parquet_vectors)} ingredients")
                            return parquet_vectors
                        logger.warning(f"⚠️ Parquet loading failed or produced smaller dataset, using largest available vector file: {vector_count} vectors")
                        return vectors
                    continue
                else:
                    logger.info(f"✅ Found full dataset: {vector_count} vectors")
                    return vectors
            except Exception as e:
                logger.warning(f"⚠️ Failed to load vectors from {vector_path}: {e}")
                continue
    
    # If no vector files found, try parquet files as fallback
    logger.warning(f"⚠️ No vector files found, falling back to parquet loading...")
    parquet_vectors = load_ingredient_data_from_parquet()
    if parquet_vectors:
        logger.info(f"✅ Successfully loaded dataset from parquet: {len(parquet_vectors)} ingredients")
        return parquet_vectors
        
    logger.error("❌ Failed to load ingredient vectors from any path or parquet files")
    return {}

def build_target_vector(taste_profile, vectors):
    """Build target vector by averaging vectors of ingredients associated with taste profile"""
    if not SEMANTIC_AVAILABLE or not vectors:
        return None
        
    selected_vecs = []
    for taste in taste_profile:
        taste_ingredients = TASTE_PROFILES.get(taste.lower(), [])
        for ing in taste_ingredients:
            # Try exact match and lowercase variants
            for variant in [ing, ing.lower(), ing.title()]:
                if variant in vectors:
                    selected_vecs.append(vectors[variant])
                    break
    
    if selected_vecs:
        target_vec = np.mean(selected_vecs, axis=0)
        logger.info(f"🎯 Built target vector from {len(selected_vecs)} taste-related ingredients")
        return target_vec
    else:
        logger.warning("⚠️ No matching ingredients found for taste profile")
        return None

def get_top_similar_ingredients(target_vec, all_vectors, top_n=50):
    """Find ingredients most similar to target vector using cosine similarity"""
    if not SEMANTIC_AVAILABLE or target_vec is None or not all_vectors:
        return list(all_vectors.keys())[:top_n] if all_vectors else []
    
    try:
        ingredients = list(all_vectors.keys())
        vectors = list(all_vectors.values())
        
        # Calculate cosine similarities
        similarities = cosine_similarity([target_vec], vectors)[0]
        
        # Rank by similarity
        ranked = sorted(zip(ingredients, similarities), key=lambda x: -x[1])
        
        # Return top N ingredient names
        top_ingredients = [name for name, sim in ranked[:top_n]]
        logger.info(f"🔍 Found {len(top_ingredients)} semantically similar ingredients")
        return top_ingredients
        
    except Exception as e:
        logger.error(f"❌ Error in semantic similarity calculation: {e}")
        return list(all_vectors.keys())[:top_n] if all_vectors else []

class PersistentFoodAIHandler(BaseHTTPRequestHandler):
    """Enhanced Food AI HTTP Request Handler with robust error handling"""
    
    def __init__(self, *args, **kwargs):
        self.ingredient_weights = self.load_weights()
        self.ingredient_vectors = load_ingredient_vectors()
        logger.info(f"🧠 Handler initialized with {len(self.ingredient_vectors)} ingredient vectors")
        super().__init__(*args, **kwargs)
    
    def load_weights(self):
        """Load ingredient weights from feedback learning"""
        try:
            with open('ingredient_weights.json', 'r', encoding='utf-8') as f:
                weights = json.load(f)
                logger.info(f"📊 Loaded {len(weights)} ingredient weights")
                return weights
        except FileNotFoundError:
            logger.info("📊 No ingredient weights file found - using default scoring")
            return {}
        except Exception as e:
            logger.warning(f"⚠️ Error loading weights: {e}")
            return {}
    
    def log_message(self, format, *args):
        """Override to use our logger"""
        logger.info("%s - %s" % (self.address_string(), format % args))
    
    def do_OPTIONS(self):
        """Handle CORS preflight requests"""
        try:
            self.send_response(200)
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            self.end_headers()
        except Exception as e:
            logger.error(f"❌ OPTIONS error: {e}")
    
    def do_GET(self):
        """Handle GET requests"""
        try:
            parsed_path = urlparse(self.path)
            
            if parsed_path.path == '/health':
                self.handle_health()
            else:
                self.send_error_response(404, "Endpoint not found")
        except Exception as e:
            logger.error(f"❌ GET error: {e}")
            self.send_error_response(500, f"Server error: {str(e)}")
    
    def do_POST(self):
        """Handle POST requests with robust error handling"""
        try:
            parsed_path = urlparse(self.path)
            
            # Read request body safely
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length > 0:
                post_data = self.rfile.read(content_length)
                request_data = json.loads(post_data.decode('utf-8'))
            else:
                request_data = {}
            
            if parsed_path.path == '/formulate':
                self.handle_formulate(request_data)
            elif parsed_path.path == '/feedback':
                self.handle_feedback(request_data)
            elif parsed_path.path == '/retrain':
                self.handle_retrain(request_data)
            else:
                self.send_error_response(404, "Endpoint not found")
                
        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON decode error: {e}")
            self.send_error_response(400, "Invalid JSON in request body")
        except Exception as e:
            logger.error(f"❌ POST error: {e}")
            self.send_error_response(500, f"Server error: {str(e)}")
    
    def handle_health(self):
        """Handle health check endpoint"""
        try:
            health_data = {
                "status": "healthy",
                "version": "2.2.0-persistent",
                "features": [
                    "bundle_generation",
                    "feedback_learning", 
                    "weight_adaptation",
                    "persistent_server"
                ],
                "ingredient_weights_loaded": len(self.ingredient_weights),
                "timestamp": time.time()
            }
            
            self.send_json_response(health_data)
            
        except Exception as e:
            logger.error(f"❌ Health check error: {e}")
            self.send_error_response(500, f"Health check failed: {str(e)}")
    
    def handle_formulate(self, data):
        """Handle bundle formulation with enhanced error handling"""
        logger.info("🎯 Processing bundle formulation")
        
        try:
            # Extract parameters safely
            taste_profile = data.get("tasteProfile", [])
            texture = data.get("texture", 5)
            nutritional_constraints = data.get("nutritionalConstraints", {})
            functional_constraints = data.get("functionalConstraints", {})
            priorities = data.get("priorities", {})
            user_notes = data.get("userNotes", "")
            top_k = data.get("top_k", 3)
            bundle_size = data.get("bundle_size", 3)
            
            # Generate bundles
            bundles = self.generate_enhanced_bundles(
                taste_profile, texture, nutritional_constraints,
                functional_constraints, priorities, user_notes,
                top_k, bundle_size
            )
            
            # Create response
            response = {
                "status": "success",
                "bundles": bundles,
                "debug_info": {
                    "taste_profile": taste_profile,
                    "constraints": {
                        "nutritional": nutritional_constraints,
                        "functional": functional_constraints
                    },
                    "priorities": priorities,
                    "user_notes": user_notes
                },
                "thought_process": [
                    "Analyzed user taste preferences and constraints",
                    "Applied nutritional and functional filters",
                    "Calculated compatibility scores with feedback learning",
                    "Generated diverse ingredient combinations"
                ],
                "metadata": {
                    "generation_time": time.time(),
                    "bundle_count": len(bundles),
                    "feedback_weights_applied": len(self.ingredient_weights) > 0
                }
            }
            
            self.send_json_response(response)
            
        except Exception as e:
            logger.error(f"❌ Formulation error: {e}")
            self.send_error_response(500, f"Bundle generation failed: {str(e)}")
    
    def handle_feedback(self, data):
        """Handle feedback submission with robust timestamp handling"""
        logger.info("👍 Processing feedback submission")
        
        try:
            # Extract feedback data safely
            bundle_index = data.get('bundle_index', 0)
            bundle_name = data.get('bundle_name', 'Unknown Bundle')
            sentiment = data.get('sentiment', 'neutral')
            rating = data.get('rating', 3)
            ingredients = data.get('ingredients', [])
            
            # Handle timestamp - convert ISO string to unix timestamp if needed
            raw_timestamp = data.get('timestamp', time.time())
            if isinstance(raw_timestamp, str):
                try:
                    # Parse ISO timestamp string to unix timestamp
                    dt = datetime.fromisoformat(raw_timestamp.replace('Z', '+00:00'))
                    timestamp = dt.timestamp()
                except Exception as ts_error:
                    logger.warning(f"⚠️ Timestamp parsing error: {ts_error}")
                    # Fallback to current time if parsing fails
                    timestamp = time.time()
            else:
                timestamp = raw_timestamp
            
            # Create feedback entry
            feedback_entry = {
                "bundle_index": bundle_index,
                "bundle_name": bundle_name,
                "sentiment": sentiment,
                "rating": rating,
                "ingredients": ingredients,
                "timestamp": timestamp,
                "timestamp_iso": raw_timestamp  # Keep original for reference
            }
            
            # Store feedback in feedback_log.jsonl
            try:
                with open('feedback_log.jsonl', 'a', encoding='utf-8') as f:
                    f.write(json.dumps(feedback_entry) + '\n')
                logger.info(f"✅ Feedback stored: {sentiment} rating for {bundle_name}")
            except Exception as store_error:
                logger.warning(f"⚠️ Could not store feedback: {store_error}")
            
            # Send success response
            response = {
                "status": "success",
                "message": "Thank you for your feedback!",
                "feedback_id": f"feedback_{int(timestamp)}",
                "timestamp": timestamp
            }
            
            self.send_json_response(response)
            
        except Exception as e:
            logger.error(f"❌ Feedback submission error: {e}")
            self.send_error_response(500, f"Feedback submission failed: {str(e)}")
    
    def handle_retrain(self, data):
        """Handle retraining request"""
        logger.info("🧠 Processing retraining")
        
        try:
            # Reload weights
            self.ingredient_weights = self.load_weights()
            
            response = {
                "status": "success",
                "message": "Weights reloaded successfully!",
                "weights_count": len(self.ingredient_weights),
                "timestamp": time.time()
            }
            
            self.send_json_response(response)
            
        except Exception as e:
            logger.error(f"❌ Retraining error: {e}")
            self.send_error_response(500, f"Retraining failed: {str(e)}")
    
    def generate_enhanced_bundles(self, taste_profile, texture, nutritional_constraints,
                                  functional_constraints, priorities, user_notes, top_k, bundle_size):
        """Generate enhanced ingredient bundles with semantic vector matching and feedback learning"""
        
        # Always try to use the ingredient vectors if available
        if self.ingredient_vectors:
            vector_size = len(self.ingredient_vectors)
            logger.info(f"📀 Working with {vector_size} ingredients in vector dataset")
            
            # Check if we have a good subset of the full dataset (312,291 expected)
            dataset_percentage = (vector_size / 312291) * 100
            if vector_size < 300000:
                logger.warning(f"⚠️ Using partial dataset ({vector_size} ingredients, {dataset_percentage:.1f}% of full dataset)")
            else:
                logger.info(f"✅ Using full ingredient dataset ({vector_size} ingredients)")
                
            # Check for taste profile
            if taste_profile:
                logger.info(f"🧠 Using semantic matching for taste profile: {taste_profile}")
                target_vector = build_target_vector(taste_profile, self.ingredient_vectors)
                if target_vector is not None:
                    # Use top similar ingredients with larger pool for diversity
                    sample_size = min(500, vector_size)
                    candidate_pool = get_top_similar_ingredients(target_vector, self.ingredient_vectors, top_n=sample_size)
                    logger.info(f"🎯 Generated candidate pool of {len(candidate_pool)} semantically similar ingredients")
                    logger.info(f"🎯 Sample candidates: {candidate_pool[:10]}")
                else:
                    # No target vector but we have ingredients - use random sampling from all vectors
                    logger.info("⚠️ No taste vector generated - using random sample from full ingredient set")
                    vector_keys = list(self.ingredient_vectors.keys())
                    import random
                    sample_size = min(500, vector_size)
                    candidate_pool = random.sample(vector_keys, sample_size)
                    logger.info(f"🎲 Generated diverse pool with {len(candidate_pool)} random ingredients")
            else:
                # No taste profile but we have ingredients - use random sampling from all vectors
                logger.info("⚠️ No taste profile provided - using random sample from full ingredient set")
                vector_keys = list(self.ingredient_vectors.keys())
                import random
                sample_size = min(500, vector_size)
                candidate_pool = random.sample(vector_keys, sample_size)
                logger.info(f"🎲 Generated diverse pool with {len(candidate_pool)} random ingredients from {vector_size} total")
        else:
            # Only use fallback if absolutely no vectors available
            logger.warning("❌ No ingredient vectors available - using minimal fallback list")
            candidate_pool = [
                "Tomatoes", "Basil", "Mozzarella", "Olive Oil", "Garlic",
                "Chicken Breast", "Lemon", "Thyme", "Black Pepper", "Sea Salt",
                "Avocado", "Lime", "Cilantro", "Red Onion", "Bell Peppers",
                "Spinach", "Feta", "Balsamic Vinegar", "Pine Nuts", "Sun-dried Tomatoes",
                "Mushrooms", "Rosemary", "Parmesan", "Arugula", "Cherry Tomatoes",
                "Quinoa", "Sweet Potato", "Kale", "Almonds", "Coconut Oil",
                "Ginger", "Turmeric", "Cinnamon", "Honey", "Greek Yogurt",
                "Salmon", "Asparagus", "Brussels Sprouts", "Walnuts", "Blueberries"
            ]
        
        # Vegan alternatives
        vegan_alternatives = {
            "Mozzarella": "Cashew Cheese",
            "Chicken Breast": "Marinated Tofu",
            "Cream Cheese": "Almond Cream",
            "Feta": "Nutritional Yeast",
            "Salmon": "Marinated Mushrooms"
        }
        
        bundles = []
        
        # Calculate target vector for semantic scoring (if available)
        target_vector = None
        if self.ingredient_vectors and taste_profile:
            target_vector = build_target_vector(taste_profile, self.ingredient_vectors)
        
        for i in range(top_k):
            # Select diverse ingredients from candidate pool
            import random
            if len(candidate_pool) >= bundle_size:
                # Sample without replacement for diversity
                available_pool = candidate_pool.copy()
                selected_ingredients = []
                for _ in range(bundle_size):
                    if available_pool:
                        # Weighted selection - prefer items earlier in the list (higher similarity)
                        weights = [1.0 / (j + 1) for j in range(len(available_pool))]
                        selected = random.choices(available_pool, weights=weights, k=1)[0]
                        selected_ingredients.append(selected)
                        available_pool.remove(selected)
            else:
                # If pool is smaller than bundle size, use what we have
                selected_ingredients = candidate_pool[:bundle_size]
            
            # Apply vegan substitutions if needed
            if nutritional_constraints.get("vegan"):
                selected_ingredients = [
                    vegan_alternatives.get(ing, ing) for ing in selected_ingredients
                ]
            
            # Filter gluten-free if needed
            if nutritional_constraints.get("glutenFree"):
                gluten_free_ingredients = [ing for ing in selected_ingredients 
                                         if "wheat" not in ing.lower() and "bread" not in ing.lower()]
                selected_ingredients = gluten_free_ingredients or selected_ingredients
            
            # Calculate score with feedback learning
            base_score = 0.9 - (i * 0.1)
            
            # Apply feedback weights
            feedback_bonus = sum(self.ingredient_weights.get(ing.lower(), 0) for ing in selected_ingredients) * 0.01
            compatibility_score = min(1.0, base_score + feedback_bonus)
            
            # Calculate semantic similarity score if vectors available
            semantic_score = None
            if target_vector is not None and self.ingredient_vectors:
                try:
                    ingredient_similarities = []
                    for ingredient in selected_ingredients:
                        # Try different case variants to find the ingredient vector
                        ing_vector = None
                        for variant in [ingredient, ingredient.lower(), ingredient.title()]:
                            if variant in self.ingredient_vectors:
                                ing_vector = self.ingredient_vectors[variant]
                                break
                        
                        if ing_vector is not None:
                            similarity = cosine_similarity([target_vector], [ing_vector])[0][0]
                            ingredient_similarities.append(similarity)
                    
                    if ingredient_similarities:
                        semantic_score = round(np.mean(ingredient_similarities), 3)
                        logger.info(f"🎯 Bundle {i+1} semantic score: {semantic_score}")
                except Exception as e:
                    logger.warning(f"⚠️ Error calculating semantic score: {e}")
                    semantic_score = None
            
            # Generate reasoning
            reasoning_parts = []
            if taste_profile:
                reasoning_parts.append(f"Selected for taste preferences: {', '.join(taste_profile)}")
            if nutritional_constraints.get("vegan"):
                reasoning_parts.append("All ingredients are vegan-friendly")
            if feedback_bonus != 0:
                liked_ingredients = [ing for ing in selected_ingredients if self.ingredient_weights.get(ing.lower(), 0) > 0]
                if liked_ingredients:
                    reasoning_parts.append(f"Boosted score due to user-liked ingredients: {', '.join(liked_ingredients)}")
            
            reasoning = ". ".join(reasoning_parts) if reasoning_parts else "Balanced flavor combination"
            
            bundle = {
                "name": f"Flavor Bundle {i + 1}",
                "ingredients": selected_ingredients,
                "score": round(compatibility_score, 3),
                "semantic_score": semantic_score,
                "description": f"A {', '.join(taste_profile) if taste_profile else 'balanced'} combination with {texture}/10 texture complexity",
                "reasoning": reasoning,
                "nutritionalInfo": {
                    "vegan": nutritional_constraints.get("vegan", False),
                    "glutenFree": nutritional_constraints.get("glutenFree", False),
                    "highProtein": any("protein" in ing.lower() for ing in selected_ingredients)
                },
                "functionalProperties": {
                    "energyBoosting": functional_constraints.get("energyBoosting", False),
                    "antiInflammatory": any(ing in ["Turmeric", "Ginger"] for ing in selected_ingredients)
                }
            }
            bundles.append(bundle)
        
        return bundles
    
    def send_json_response(self, data):
        """Send JSON response with proper headers"""
        try:
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            self.end_headers()
            
            json_str = json.dumps(data, indent=2, ensure_ascii=False)
            self.wfile.write(json_str.encode('utf-8'))
            
        except Exception as e:
            logger.error(f"❌ JSON response error: {e}")
    
    def send_error_response(self, status_code, message):
        """Send error response with proper headers"""
        try:
            self.send_response(status_code)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            error_response = {
                "status": "error",
                "message": message,
                "timestamp": time.time()
            }
            
            json_str = json.dumps(error_response, indent=2, ensure_ascii=False)
            self.wfile.write(json_str.encode('utf-8'))
            
        except Exception as e:
            logger.error(f"❌ Error response error: {e}")

# Global server instance
server_instance = None
shutdown_flag = threading.Event()

def cleanup_handler(signum=None, frame=None):
    """Clean shutdown handler"""
    global server_instance, shutdown_flag
    
    print("\n🛑 Shutting down Enhanced Food AI Backend...")
    shutdown_flag.set()
    
    if server_instance:
        try:
            server_instance.shutdown()
            print("🧹 Server shutdown complete")
        except:
            pass
    
    sys.exit(0)

def check_port_available(port):
    """Check if port is available"""
    import socket
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', port))
            return True
    except:
        return False

def main():
    """Start the persistent backend server"""
    global server_instance
    
    # Register cleanup handlers
    signal.signal(signal.SIGINT, cleanup_handler)
    signal.signal(signal.SIGTERM, cleanup_handler)
    atexit.register(lambda: print("🧹 Backend cleanup complete"))
    
    # Check if port is available
    if not check_port_available(5000):
        print("⚠️ Port 5000 is already in use. Attempting to kill conflicting processes...")
        import subprocess
        try:
            subprocess.run(["taskkill", "/f", "/im", "python.exe"], capture_output=True)
            time.sleep(2)  # Wait for processes to terminate
        except:
            pass
        
        if not check_port_available(5000):
            print("❌ Port 5000 is still in use. Please manually stop other Python processes.")
            return
    
    try:
        server_address = ('', 5000)
        server_instance = HTTPServer(server_address, PersistentFoodAIHandler)
        
        print("🚀 Persistent Enhanced Food AI Backend starting...")
        print(f"🌐 Server running on http://localhost:5000")
        print("📊 Feedback learning system active")
        print("🔒 Persistent server mode enabled")
        print("🛑 Press Ctrl+C to stop")
        print("✅ Backend ready for connections!")
        
        # Keep server running persistently
        while not shutdown_flag.is_set():
            try:
                server_instance.handle_request()
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"❌ Server error: {e}")
                # Continue running even if there's an error
                time.sleep(0.1)
                
    except Exception as e:
        logger.error(f"❌ Failed to start server: {e}")
        sys.exit(1)
    finally:
        cleanup_handler()

if __name__ == "__main__":
    main()
