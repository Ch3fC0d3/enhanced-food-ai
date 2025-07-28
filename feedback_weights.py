#!/usr/bin/env python3
"""
Feedback-Driven Weight Tracker for Enhanced Food AI

This script processes user feedback from feedback_log.jsonl and calculates
ingredient-level weights to improve future bundle generation.

Usage:
    python feedback_weights.py
"""

import json
import time
from collections import defaultdict
from pathlib import Path

def calculate_ingredient_weights(feedback_file="feedback_log.jsonl"):
    """
    Calculate ingredient weights based on user feedback
    
    Args:
        feedback_file (str): Path to JSONL feedback log file
        
    Returns:
        dict: Ingredient weights {ingredient_name: weight_score}
    """
    weights = defaultdict(float)
    feedback_count = 0
    
    feedback_path = Path(feedback_file)
    if not feedback_path.exists():
        print(f"⚠️  No feedback file found at {feedback_file}")
        return {}
    
    print(f"📊 Processing feedback from {feedback_file}...")
    
    with open(feedback_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                
                # Extract feedback type and convert to score
                feedback_type = data.get("feedback", "unknown")
                rating = data.get("rating", 0)
                ingredients = data.get("ingredients", [])
                
                # Calculate score based on feedback type and rating
                if feedback_type == "up" or feedback_type == "thumbs_up":
                    score = 1.0
                elif feedback_type == "down" or feedback_type == "thumbs_down":
                    score = -1.0
                elif rating > 0:
                    # Use star rating (1-5) converted to -1 to 1 scale
                    score = (rating - 3) / 2  # 1->-1, 3->0, 5->1
                else:
                    score = 0
                
                # Apply score to all ingredients in the bundle
                for ingredient in ingredients:
                    if isinstance(ingredient, str) and ingredient.strip():
                        weights[ingredient.strip().lower()] += score
                
                feedback_count += 1
                
            except json.JSONDecodeError as e:
                print(f"⚠️  Skipped malformed JSON on line {line_num}: {e}")
            except Exception as e:
                print(f"⚠️  Skipped line {line_num} due to error: {e}")
    
    print(f"✅ Processed {feedback_count} feedback entries")
    print(f"📈 Generated weights for {len(weights)} ingredients")
    
    return dict(weights)

def save_weights(weights, path="ingredient_weights.json"):
    """
    Save ingredient weights to JSON file
    
    Args:
        weights (dict): Ingredient weights
        path (str): Output file path
    """
    # Add metadata
    weight_data = {
        "generated_at": time.time(),
        "generated_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_ingredients": len(weights),
        "weights": weights
    }
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(weight_data, f, indent=2, sort_keys=True)
    
    print(f"💾 Weights saved to {path}")

def analyze_weights(weights):
    """
    Analyze and display weight statistics
    
    Args:
        weights (dict): Ingredient weights
    """
    if not weights:
        print("📊 No weights to analyze")
        return
    
    sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n📊 Weight Analysis:")
    print(f"   Total ingredients: {len(weights)}")
    print(f"   Positive weights: {sum(1 for w in weights.values() if w > 0)}")
    print(f"   Negative weights: {sum(1 for w in weights.values() if w < 0)}")
    print(f"   Neutral weights: {sum(1 for w in weights.values() if w == 0)}")
    
    print(f"\n🔝 Top 10 Most Liked Ingredients:")
    for ingredient, weight in sorted_weights[:10]:
        if weight > 0:
            print(f"   {ingredient}: +{weight:.2f}")
    
    print(f"\n🔻 Top 10 Most Disliked Ingredients:")
    for ingredient, weight in sorted_weights[-10:]:
        if weight < 0:
            print(f"   {ingredient}: {weight:.2f}")

def main():
    """Main execution function"""
    print("🧠 Enhanced Food AI - Feedback Weight Tracker")
    print("=" * 50)
    
    # Calculate weights from feedback
    weights = calculate_ingredient_weights()
    
    if not weights:
        print("❌ No feedback data found. Generate some bundles and provide feedback first!")
        return
    
    # Analyze weights
    analyze_weights(weights)
    
    # Save weights for backend to use
    save_weights(weights)
    
    print(f"\n✅ Weight calculation complete!")
    print(f"💡 Restart your backend to load the new weights.")

if __name__ == "__main__":
    main()
