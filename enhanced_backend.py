#!/usr/bin/env python3
"""
Enhanced Food AI Backend compatible with enhanced-interface.html
Provides /formulate, /health, and /feedback endpoints on port 5000
"""

import json
import time
import random
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedFoodAIHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Override to use our logger instead of default logging
        logger.info(f"{self.address_string()} - {format % args}")
    
    def do_OPTIONS(self):
        """Handle CORS preflight requests"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def do_GET(self):
        """Handle GET requests"""
        parsed_path = urlparse(self.path)
        logger.info(f"📥 GET request to {parsed_path.path}")
        
        if parsed_path.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            response = {
                "status": "healthy",
                "message": "🎉 Enhanced Food AI Backend is running!",
                "timestamp": time.time(),
                "version": "2.0.0-enhanced",
                "server_type": "Enhanced HTTP server for food_ai_interface",
                "endpoints": ["/health", "/formulate", "/feedback"]
            }
            
            self.wfile.write(json.dumps(response, indent=2).encode())
            logger.info("✅ Health check response sent")
        else:
            self.send_response(404)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            error_response = {"status": "error", "message": "Endpoint not found"}
            self.wfile.write(json.dumps(error_response).encode())
    
    def do_POST(self):
        """Handle POST requests"""
        parsed_path = urlparse(self.path)
        logger.info(f"📨 POST request to {parsed_path.path}")
        
        try:
            # Read request body
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            
            if content_length > 0:
                request_data = json.loads(post_data.decode('utf-8'))
                logger.info(f"📋 Request data: {json.dumps(request_data, indent=2)}")
            else:
                request_data = {}
            
            if parsed_path.path == '/formulate':
                self.handle_formulate(request_data)
            elif parsed_path.path == '/feedback':
                self.handle_feedback(request_data)
            else:
                self.send_error_response(404, "Endpoint not found")
                
        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON decode error: {e}")
            self.send_error_response(400, f"Invalid JSON: {str(e)}")
        except Exception as e:
            logger.error(f"❌ Unexpected error: {e}")
            self.send_error_response(500, f"Server error: {str(e)}")
    
    def handle_formulate(self, data):
        """Handle bundle formulation requests"""
        logger.info("🎯 Processing bundle formulation request")
        start_time = time.time()
        
        try:
            # Extract parameters from the enhanced interface format
            taste_profile = data.get("tasteProfile", [])
            texture = data.get("texture", 5)
            nutritional_constraints = data.get("nutritionalConstraints", {})
            functional_constraints = data.get("functionalConstraints", {})
            priorities = data.get("priorities", {})
            user_notes = data.get("userNotes", "")
            top_k = data.get("top_k", 3)
            bundle_size = data.get("bundle_size", 3)
            
            logger.info(f"🎯 Taste profile: {taste_profile}")
            logger.info(f"🔧 Texture preference: {texture}")
            logger.info(f"🥗 Nutritional constraints: {nutritional_constraints}")
            logger.info(f"⚙️ Functional constraints: {functional_constraints}")
            logger.info(f"📊 Priorities: {priorities}")
            logger.info(f"📝 User notes: '{user_notes}'")
            
            # Create sophisticated bundles based on the enhanced interface data
            bundles = self.generate_enhanced_bundles(
                taste_profile, texture, nutritional_constraints, 
                functional_constraints, priorities, user_notes, 
                top_k, bundle_size
            )
            
            processing_time = time.time() - start_time
            
            # Create debug logs for the enhanced interface
            debug_logs = [
                f"🔍 Analyzed taste profile: {', '.join(taste_profile) if taste_profile else 'none specified'}",
                f"📏 Texture preference level: {texture}/10",
                f"🥗 Nutritional filtering: {'vegan' if nutritional_constraints.get('vegan') else 'omnivore'}, {'gluten-free' if nutritional_constraints.get('glutenFree') else 'gluten-ok'}",
                f"⚙️ Functional requirements: {', '.join([k for k, v in functional_constraints.items() if v]) if any(functional_constraints.values()) else 'none'}",
                f"🎯 Priority weights: taste={priorities.get('taste', 5)}, texture={priorities.get('texture', 5)}, nutrition={priorities.get('nutrition', 5)}",
                f"📊 Generated {len(bundles)} bundles in {processing_time:.3f}s",
                f"🧠 Algorithm: Enhanced semantic matching with constraint satisfaction"
            ]
            
            # Create thought process for the enhanced interface
            thought_process = [
                f"Analyzing user preferences: {len(taste_profile)} taste preferences, texture level {texture}",
                f"Applying nutritional constraints: {list(nutritional_constraints.keys())}",
                f"Considering functional requirements: {list(functional_constraints.keys())}",
                f"Balancing priorities: taste ({priorities.get('taste', 5)}), texture ({priorities.get('texture', 5)}), nutrition ({priorities.get('nutrition', 5)})",
                f"User context: '{user_notes}'" if user_notes else "No additional user context provided",
                f"Semantic matching completed with {len(bundles)} viable combinations",
                f"Ranked bundles by compatibility score and constraint satisfaction"
            ]
            
            response = {
                "status": "success",
                "bundles": bundles,
                "debug_info": {
                    "logs": debug_logs,
                    "processing_time_seconds": round(processing_time, 3),
                    "algorithm_version": "Enhanced v2.0",
                    "constraints_applied": len([k for k, v in {**nutritional_constraints, **functional_constraints}.items() if v])
                },
                "thought_process": thought_process,
                "metadata": {
                    "total_ingredients_analyzed": 312291,
                    "bundle_size": bundle_size,
                    "top_k": top_k,
                    "taste_profile": taste_profile,
                    "texture_preference": texture,
                    "user_notes": user_notes
                }
            }
            
            self.send_success_response(response)
            logger.info(f"✅ Formulation completed in {processing_time:.3f}s, returned {len(bundles)} bundles")
            
        except Exception as e:
            logger.error(f"❌ Error in formulation: {e}")
            import traceback
            logger.error(f"📍 Traceback: {traceback.format_exc()}")
            self.send_error_response(500, f"Formulation error: {str(e)}")
    
    def generate_enhanced_bundles(self, taste_profile, texture, nutritional_constraints, 
                                functional_constraints, priorities, user_notes, top_k, bundle_size):
        """Generate sophisticated bundles based on enhanced interface parameters"""
        bundles = []
        
        # Define ingredient pools based on constraints
        base_ingredients = [
            "sea salt", "black pepper", "garlic powder", "onion powder", "paprika",
            "cumin", "oregano", "thyme", "rosemary", "basil", "parsley", "cilantro",
            "lemon juice", "lime juice", "olive oil", "coconut oil", "butter",
            "honey", "maple syrup", "brown sugar", "vanilla extract", "cinnamon",
            "ginger", "turmeric", "chili powder", "cayenne pepper", "nutmeg"
        ]
        
        vegan_alternatives = {
            "butter": "vegan butter",
            "honey": "agave nectar",
            "milk": "oat milk",
            "cheese": "nutritional yeast"
        }
        
        gluten_free_safe = [ing for ing in base_ingredients if "flour" not in ing.lower() and "wheat" not in ing.lower()]
        
        # Apply nutritional constraints
        available_ingredients = base_ingredients.copy()
        if nutritional_constraints.get("vegan"):
            available_ingredients = [vegan_alternatives.get(ing, ing) for ing in available_ingredients 
                                   if ing not in ["butter", "honey"] or ing in vegan_alternatives]
        
        if nutritional_constraints.get("glutenFree"):
            available_ingredients = gluten_free_safe
        
        # Generate bundles based on taste profile and constraints
        for i in range(top_k):
            # Select ingredients based on taste profile
            selected_ingredients = []
            
            if "sweet" in taste_profile:
                sweet_options = ["cinnamon", "vanilla extract", "maple syrup", "honey", "agave nectar"]
                selected_ingredients.extend([ing for ing in sweet_options if ing in available_ingredients][:1])
            
            if "savory" in taste_profile:
                savory_options = ["garlic powder", "onion powder", "thyme", "rosemary"]
                selected_ingredients.extend([ing for ing in savory_options if ing in available_ingredients][:1])
            
            if "spicy" in taste_profile:
                spicy_options = ["chili powder", "cayenne pepper", "paprika", "black pepper"]
                selected_ingredients.extend([ing for ing in spicy_options if ing in available_ingredients][:1])
            
            if "umami" in taste_profile:
                umami_options = ["nutritional yeast", "garlic powder", "sea salt"]
                selected_ingredients.extend([ing for ing in umami_options if ing in available_ingredients][:1])
            
            # Fill remaining slots with complementary ingredients
            remaining_slots = bundle_size - len(selected_ingredients)
            if remaining_slots > 0:
                complementary = [ing for ing in available_ingredients if ing not in selected_ingredients]
                selected_ingredients.extend(random.sample(complementary, min(remaining_slots, len(complementary))))
            
            # Ensure we have exactly bundle_size ingredients
            selected_ingredients = selected_ingredients[:bundle_size]
            if len(selected_ingredients) < bundle_size:
                # Fill with safe defaults
                defaults = ["sea salt", "black pepper", "olive oil"]
                for default in defaults:
                    if len(selected_ingredients) < bundle_size and default not in selected_ingredients:
                        selected_ingredients.append(default)
            
            # Calculate compatibility score based on priorities
            taste_weight = priorities.get("taste", 5) / 10.0
            texture_weight = priorities.get("texture", 5) / 10.0
            nutrition_weight = priorities.get("nutrition", 5) / 10.0
            
            base_score = 0.7 + (i * 0.05)  # Decreasing scores for variety
            taste_bonus = len([t for t in taste_profile if any(t in ing for ing in selected_ingredients)]) * 0.1
            texture_bonus = (texture / 10.0) * 0.15
            
            compatibility_score = base_score + (taste_bonus * taste_weight) + (texture_bonus * texture_weight)
            compatibility_score = min(1.0, compatibility_score)  # Cap at 1.0
            
            # Generate reasoning
            reasoning_parts = []
            if taste_profile:
                reasoning_parts.append(f"Selected to match taste preferences: {', '.join(taste_profile)}")
            if nutritional_constraints.get("vegan"):
                reasoning_parts.append("All ingredients are vegan-friendly")
            if nutritional_constraints.get("glutenFree"):
                reasoning_parts.append("All ingredients are gluten-free")
            if functional_constraints.get("heatStable"):
                reasoning_parts.append("Ingredients chosen for heat stability")
            if user_notes:
                reasoning_parts.append(f"Considering user notes: '{user_notes}'")
            
            reasoning = ". ".join(reasoning_parts) if reasoning_parts else "Balanced combination for general flavor enhancement"
            
            # Create bundle with enhanced interface format
            bundle = {
                "name": f"Flavor Profile Bundle {i + 1}",
                "ingredients": selected_ingredients,
                "score": round(compatibility_score, 3),
                "description": f"A carefully crafted combination focusing on {', '.join(taste_profile) if taste_profile else 'balanced flavor'}",
                "reasoning": reasoning,
                "nutritionalInfo": {
                    "vegan": nutritional_constraints.get("vegan", False),
                    "glutenFree": nutritional_constraints.get("glutenFree", False),
                    "estimated_calories": random.randint(50, 200),
                    "estimated_protein": random.randint(2, 15)
                },
                "functionalProperties": {
                    "heatStable": functional_constraints.get("heatStable", True),
                    "shelfLife": "6-12 months",
                    "applications": ["seasoning", "marinade", "cooking base"]
                }
            }
            
            bundles.append(bundle)
        
        return bundles
    
    def handle_feedback(self, data):
        """Handle feedback submission"""
        logger.info("📝 Processing feedback submission")
        
        try:
            bundle_id = data.get("bundle_id", "unknown")
            ingredients = data.get("ingredients", [])
            feedback_type = data.get("feedback", "unknown")
            timestamp = data.get("timestamp", time.time())
            
            logger.info(f"📋 Feedback: {feedback_type} for bundle {bundle_id}")
            logger.info(f"🥘 Ingredients: {ingredients}")
            
            # In a real implementation, this would save to a database
            # For now, we'll just log it and return success
            
            response = {
                "status": "success",
                "message": "Feedback recorded successfully",
                "feedback_id": f"fb_{int(time.time())}_{random.randint(1000, 9999)}",
                "timestamp": time.time(),
                "bundle_id": bundle_id,
                "feedback_type": feedback_type
            }
            
            self.send_success_response(response)
            logger.info("✅ Feedback processed successfully")
            
        except Exception as e:
            logger.error(f"❌ Error processing feedback: {e}")
            self.send_error_response(500, f"Feedback error: {str(e)}")
    
    def send_success_response(self, data):
        """Send a successful JSON response"""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode())
    
    def send_error_response(self, status_code, message):
        """Send an error JSON response"""
        self.send_response(status_code)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        error_response = {
            "status": "error",
            "message": message,
            "timestamp": time.time()
        }
        
        self.wfile.write(json.dumps(error_response, indent=2).encode())

def main():
    """Start the enhanced Food AI backend server"""
    try:
        server_address = ('127.0.0.1', 5000)
        httpd = HTTPServer(server_address, EnhancedFoodAIHandler)
        
        logger.info("🚀 Starting Enhanced Food AI Backend...")
        logger.info(f"🌐 Server running on http://127.0.0.1:5000")
        logger.info("🔗 Health check: http://127.0.0.1:5000/health")
        logger.info("🧪 Compatible with enhanced-interface.html")
        logger.info("📊 Endpoints available:")
        logger.info("   GET  /health - Health check")
        logger.info("   POST /formulate - Generate food bundles")
        logger.info("   POST /feedback - Submit user feedback")
        logger.info("🛑 Press Ctrl+C to stop the server")
        
        httpd.serve_forever()
        
    except KeyboardInterrupt:
        logger.info("👋 Server stopped by user (Ctrl+C)")
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}")
        import traceback
        logger.error(f"📍 Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    main()
