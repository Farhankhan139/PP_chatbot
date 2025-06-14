from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np
from sentence_transformers import util
import torch
import re
import os
import gdown

app = Flask(__name__)

# Download model.pkl from Google Drive if not already downloaded
if not os.path.exists('model.pkl'):
    print("Downloading model.pkl from Google Drive...")
    gdown.download(id='YOUR_FILE_ID_HERE', output='model.pkl', quiet=False)

# Load the trained model and artifacts
with open('model.pkl', 'rb') as f:
    artifacts = pickle.load(f)

model = artifacts['model']
property_embeddings = artifacts['property_embeddings']
df = artifacts['df']
feature_matrix = artifacts['feature_matrix']
scaler = artifacts['scaler']

PREDEFINED_RESPONSES = {
    "hello": "Hello! How can I assist you with your real estate needs today?",
    "hi": "Hi there! Looking for a property? Tell me what you're interested in.",
    "help": "I can help you find properties based on location, price, bedrooms, bathrooms, and more. Try asking something like 'Show me 3 bedroom houses in Karachi under 50 million'.",
    "thank you": "You're welcome! Let me know if you need anything else."
}

PROPERTY_SYNONYMS = {
    'house': ['home', 'villa', 'bungalow', 'residence'],
    'apartment': ['flat', 'condo', 'unit'],
    'plot': ['land', 'piece of land', 'property'],
    'commercial': ['office', 'shop', 'business space']
}

def extract_numerical_filters(query):
    filters = {}
    price_matches = re.findall(r'(?:under|below|less than)\s*([\d,.]+)\s*(?:lakh|million|crore)?', query, re.IGNORECASE)
    if price_matches:
        price = float(price_matches[0].replace(',', ''))
        if 'million' in query.lower() or 'lakh' not in query.lower() and price < 100:
            price *= 1_000_000
        elif 'lakh' in query.lower():
            price *= 100_000
        filters['max_price'] = price
    price_matches = re.findall(r'(?:over|above|more than)\s*([\d,.]+)\s*(?:lakh|million|crore)?', query, re.IGNORECASE)
    if price_matches:
        price = float(price_matches[0].replace(',', ''))
        if 'million' in query.lower() or 'lakh' not in query.lower() and price < 100:
            price *= 1_000_000
        elif 'lakh' in query.lower():
            price *= 100_000
        filters['min_price'] = price
    bed_matches = re.findall(r'(\d+)\s*bed|bedroom', query, re.IGNORECASE)
    if bed_matches:
        filters['beds'] = int(bed_matches[0])
    bath_matches = re.findall(r'(\d+)\s*bath|bathroom', query, re.IGNORECASE)
    if bath_matches:
        filters['baths'] = int(bath_matches[0])
    area_matches = re.findall(r'(\d+)\s*(?:sq|square)\s*(?:ft|feet|m|meter)', query, re.IGNORECASE)
    if area_matches:
        filters['area'] = int(area_matches[0])
    return filters

def extract_property_type(query):
    query_lower = query.lower()
    for prop_type, synonyms in PROPERTY_SYNONYMS.items():
        if prop_type in query_lower:
            return prop_type
        for synonym in synonyms:
            if synonym in query_lower:
                return prop_type
    return None

def extract_location(query):
    possible_locations = set(df['city'].str.lower().unique()) | set(df['province_name'].str.lower().unique())
    words = query.lower().split()
    for word in words:
        if word in possible_locations:
            return word
    return None

def apply_filters(df, filters):
    filtered_df = df.copy()
    if 'max_price' in filters:
        filtered_df = filtered_df[filtered_df['price'] <= filters['max_price']]
    if 'min_price' in filters:
        filtered_df = filtered_df[filtered_df['price'] >= filters['min_price']]
    if 'beds' in filters:
        filtered_df = filtered_df[filtered_df['beds'] == filters['beds']]
    if 'baths' in filters:
        filtered_df = filtered_df[filtered_df['baths'] == filters['baths']]
    if 'area' in filters:
        filtered_df = filtered_df[filtered_df['area'] >= filters['area'] * 0.9]
    property_type = filters.get('property_type')
    if property_type:
        filtered_df = filtered_df[filtered_df['property_type'] == property_type]
    location = filters.get('location')
    if location:
        filtered_df = filtered_df[
            (filtered_df['city'].str.lower() == location) |
            (filtered_df['province_name'].str.lower() == location) |
            (filtered_df['location'].str.lower().str.contains(location))
        ]
    return filtered_df

def generate_response(matching_properties, query):
    if len(matching_properties) == 0:
        return "I couldn't find any properties matching your criteria. Would you like to try different search parameters?"
    response = f"I found {len(matching_properties)} properties that match your query about {query}:\n\n"
    for i, prop in matching_properties.head(3).iterrows():
        response += (
            f"🏠 {prop['property_type'].title()} in {prop['location'].title()}, {prop['city'].title()}\n"
            f"💰 Price: {prop['price']:,} PKR\n"
            f"🛏️ {prop['beds']} beds | 🛁 {prop['baths']} baths | 📏 {prop['area']} sq ft\n"
            f"🔗 More details: {prop['page_url']}\n\n"
        )
    if len(matching_properties) > 3:
        response += f"Plus {len(matching_properties) - 3} more properties. Would you like me to refine the search further?"
    return response

@app.route('/query', methods=['POST'])
def handle_query():
    data = request.json
    query = data.get('query', '').lower().strip()
    for keyword, response in PREDEFINED_RESPONSES.items():
        if keyword in query:
            return jsonify({'response': response, 'properties': []})
    filters = extract_numerical_filters(query)
    prop_type = extract_property_type(query)
    if prop_type:
        filters['property_type'] = prop_type
    location = extract_location(query)
    if location:
        filters['location'] = location
    filtered_df = apply_filters(df, filters)
    if len(filtered_df) == 0:
        return jsonify({
            'response': "I couldn't find any properties matching your criteria. Would you like to try different search parameters?",
            'properties': []
        })
    query_embedding = model.encode(query, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(query_embedding, property_embeddings)[0]
    top_results = torch.topk(cos_scores, k=min(10, len(filtered_df)))
    matching_indices = top_results.indices.cpu().numpy()
    matching_scores = top_results.values.cpu().numpy()
    result_df = filtered_df.iloc[matching_indices].copy()
    result_df['match_score'] = matching_scores
    result_df = result_df.sort_values('match_score', ascending=False)
    response_text = generate_response(result_df, query)
    properties = result_df.head(5).to_dict('records')
    return jsonify({'response': response_text, 'properties': properties})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
