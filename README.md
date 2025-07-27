# Enhanced Food AI System

A sophisticated food ingredient bundle generator with advanced taste profiling, nutritional constraints, and feedback system.

## Quick Start

### 1. Start the Backend
```bash
python enhanced_backend.py
```
The backend will run on `http://localhost:5000`

### 2. Start the Frontend
```bash
python -m http.server 3000
```
Then open `http://localhost:3000/enhanced-interface.html` in your browser

## Features

### Advanced Interface
- **Taste Profile Selection** (sweet, savory, spicy, umami, etc.)
- **Texture Preferences** with slider controls
- **Nutritional Constraints** (vegan, gluten-free, calorie/protein ranges)
- **Functional Constraints** (heat stability, cost preferences)
- **Priority Weighting** (balance taste, texture, nutrition)
- **Custom User Notes** for specific requirements

### Enhanced Results
- **Detailed Bundle Explanations** with reasoning
- **Backend Debug Logs** showing processing steps
- **AI Thought Process** displaying decision logic
- **Nutritional Information** for each bundle
- **Compatibility Scores** for ranking

### Interactive Feedback
- **Quick Feedback** (thumbs up/down)
- **Detailed Ratings** with star system
- **Comment System** for suggestions

## API Endpoints

- `GET /health` - System status
- `POST /formulate` - Generate food bundles
- `POST /feedback` - Submit user feedback

## Files

- `enhanced-interface.html` - Main frontend interface
- `enhanced_backend.py` - Backend API server
- `README.md` - This documentation

## Development

This is a clean, minimal setup focused on the essential Enhanced Food AI functionality.
