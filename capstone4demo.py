import os
import streamlit as st
import json
import sqlite3
from PIL import Image
from typing import List, Optional
import torch

# Read secrets from Streamlit Cloud's secrets management.
SERPER_API_KEY = st.secrets["SERPER_API_KEY"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
FIRECRAWL_API_KEY = st.secrets["FIRECRAWL_API_KEY"]

os.environ["SERPER_API_KEY"] = SERPER_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# -----------------------------------------------------------------------------
# Initialize Firecrawl
# -----------------------------------------------------------------------------
from firecrawl import FirecrawlApp
firecrawl_app = FirecrawlApp(api_key=FIRECRAWL_API_KEY)

# -----------------------------------------------------------------------------
# CLIP-based Image Analysis
# -----------------------------------------------------------------------------
try:
    from transformers import CLIPProcessor, CLIPModel
except ImportError:
    from transformers import CLIPFeatureExtractor, CLIPTokenizer, CLIPModel
    class CLIPProcessor:
        def __init__(self, feature_extractor, tokenizer):
            self.feature_extractor = feature_extractor
            self.tokenizer = tokenizer

        @classmethod
        def from_pretrained(cls, pretrained_model_name_or_path):
            feature_extractor = CLIPFeatureExtractor.from_pretrained(pretrained_model_name_or_path)
            tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path)
            return cls(feature_extractor, tokenizer)

        def __call__(self, images=None, text=None, return_tensors="pt", padding=True):
            if images is not None and text is not None:
                return {
                    "pixel_values": self.feature_extractor(images, return_tensors="pt", padding=padding)["pixel_values"],
                    "input_ids": self.tokenizer(text, return_tensors="pt", padding=padding)["input_ids"]
                }
            elif images is not None:
                return self.feature_extractor(images, return_tensors="pt", padding=padding)
            elif text is not None:
                return self.tokenizer(text, return_tensors="pt", padding=padding)
            else:
                raise ValueError("At least one of images or text must be provided.")

def analyze_uploaded_images(images):
    """
    Analyze uploaded images using a CLIP model and return the best matching style descriptor.
    """
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    style_descriptions = [
        "Street Style",
        "Classic",
        "Formal Elegance",
        "Casual Chic",
        "Athleisure",
        "Bohemian",
        "Avant-garde"
    ]
    
    text_inputs = processor(text=style_descriptions, return_tensors="pt", padding=True)
    with torch.no_grad():
        text_embeddings = model.get_text_features(**text_inputs)
    
    image_embeddings = []
    for image in images:
        image_input = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            emb = model.get_image_features(**image_input)
        image_embeddings.append(emb)
    
    avg_embedding = torch.mean(torch.cat(image_embeddings), dim=0, keepdim=True)
    similarities = torch.nn.functional.cosine_similarity(avg_embedding, text_embeddings)
    best_idx = torch.argmax(similarities).item()
    return style_descriptions[best_idx]

# -----------------------------------------------------------------------------
# Preference Collection
# -----------------------------------------------------------------------------
def collect_preferences():
    st.write("Please provide additional details for your event.")
    specific_pref = st.text_input("Enter additional style preferences (e.g., colors, fabrics, patterns):")
    budget = st.number_input("Enter your budget (USD):", min_value=50, value=500)
    return specific_pref, budget

# -----------------------------------------------------------------------------
# Firecrawl-based Product Extraction using Firecrawl Scrape Endpoint
# -----------------------------------------------------------------------------
def get_products(category: str, style: str) -> List[dict]:
    """
    Use Firecrawl's scrape endpoint to extract product information for a given category.
    Constructs a Google Shopping URL using the category and style, and sends a scrape request.
    Returns up to 3 product dictionaries with only name, price, and URL.
    Also merges the raw JSON response into 'document.json' in an aggregated structure.
    """
    base_url = "https://www.google.com/search"
    query = f"{category} {style}"
    search_url = f"{base_url}?q={query.replace(' ', '+')}&tbm=shop"
    
    from pydantic import BaseModel
    class Product(BaseModel):
        name: str
        price: Optional[float] = None
        link: str
    class ExtractSchema(BaseModel):
        products: List[Product]
    
    prompt = (
        f"Extract product information about {category} for women with {style} style "
        "from the provided Google Shopping results page. Return the product name, price, and URL. "
        "Ensure that product name and price are always included."
    )
    
    try:
        data = firecrawl_app.extract([search_url], {
            'prompt': prompt,
            'schema': ExtractSchema.model_json_schema()
        })
        # Merge new data into aggregated JSON file
        aggregated_data = {}
        if os.path.exists("document.json"):
            try:
                with open("document.json", "r") as rf:
                    aggregated_data = json.load(rf)
            except Exception:
                aggregated_data = {}
        aggregated_data[category] = data
        with open("document.json", "w") as wf:
            json.dump(aggregated_data, wf, indent=2)
            
        products = data.get("products", [])
        st.write(f"Fetched {len(products)} products for '{category}'.")
        return products[:3]
    except Exception as e:
        st.error(f"Error extracting products for {category}: {e}")
        return []

def search_products_simple(scanned_style, preference, budget):
    """
    For each product category, use the Firecrawl scrape endpoint (configured for Google Shopping)
    to extract up to 3 product recommendations. Then, construct complete wardrobe sets by combining
    the i-th product from each category. Returns a list of complete wardrobe sets.
    """
    product_categories = {
        "Apparel": "apparel",
        "Coat": "women coats",
        "Shoes": "women shoes",
        "Tie": "women tie"
    }
    
    category_products = {}
    for cat, query in product_categories.items():
        prods = get_products(query, scanned_style)
        if len(prods) > 3:
            prods = prods[:3]
        category_products[cat] = prods

    min_count = min(len(v) for v in category_products.values())
    if min_count == 0:
        return []
    
    sets = []
    for i in range(min_count):
        items = []
        for cat in product_categories.keys():
            prod = category_products[cat][i]
            items.append({
                "item": cat,
                "name": prod.get("name", "No title"),
                "price": prod.get("price", "N/A"),
                "link": prod.get("link", "#")
            })
        sets.append({
            "set_name": f"Outfit Set {i+1}",
            "items": items,
            "summary": f"A complete {scanned_style} wardrobe set with a touch of '{preference}'."
        })
    return sets

# -----------------------------------------------------------------------------
# Save / Retrieve Selections (Persistence)
# -----------------------------------------------------------------------------
def save_selection(user_id, selection):
    conn = sqlite3.connect("wardrobe.db")
    c = conn.cursor()
    c.execute("CREATE TABLE IF NOT EXISTS selections (user_id TEXT, selection TEXT)")
    c.execute("INSERT INTO selections (user_id, selection) VALUES (?, ?)", (user_id, json.dumps(selection)))
    conn.commit()
    conn.close()

def retrieve_selection(user_id):
    conn = sqlite3.connect("wardrobe.db")
    c = conn.cursor()
    c.execute("CREATE TABLE IF NOT EXISTS selections (user_id TEXT, selection TEXT)")
    c.execute("SELECT selection FROM selections WHERE user_id=?", (user_id,))
    rows = c.fetchall()
    conn.close()
    return [json.loads(r[0]) for r in rows]

# -----------------------------------------------------------------------------
# Cart Functions
# -----------------------------------------------------------------------------
def init_cart():
    """Initialize a cart in session state if not already present."""
    if "cart" not in st.session_state:
        st.session_state["cart"] = []

def add_to_cart(item: dict):
    """Add a product item (name, price, link) to the session-based cart."""
    st.session_state["cart"].append(item)

def clear_cart():
    """Clear the cart."""
    st.session_state["cart"] = []

# -----------------------------------------------------------------------------
# Streamlit Interface
# -----------------------------------------------------------------------------
st.title("Fashion Designer Agent")
st.write("Curate your wardrobe for any occasion using our multi-agent system.")

init_cart()  # Initialize the cart

menu = st.sidebar.radio("Navigation", ["Upload Photos", "Event Preferences", "Recommendations", "Saved Selections", "Show document.json", "Cart"])

if menu == "Upload Photos":
    st.header("Step 1: Upload Your Photos")
    uploaded_files = st.file_uploader("Upload images (png, jpg, jpeg)", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    if st.button("Scan My Style"):
        if uploaded_files:
            images = [Image.open(f) for f in uploaded_files]
            style = analyze_uploaded_images(images)
            st.success(f"Scanned Style: **{style}**")
            st.session_state["scanned_style"] = style
        else:
            st.error("Please upload at least one image.")

elif menu == "Event Preferences":
    st.header("Step 2: Provide Event Preferences")
    pref, bud = collect_preferences()
    if st.button("Save Preferences"):
        st.session_state["preference"] = pref
        st.session_state["budget"] = bud
        st.success("Preferences saved!")
    st.write("Current Preference:", st.session_state.get("preference", "Not set"))
    st.write("Budget:", st.session_state.get("budget", "Not set"))

elif menu == "Recommendations":
    st.header("Step 3: Wardrobe Recommendations")
    if "scanned_style" not in st.session_state:
        st.error("Please complete 'Upload Photos' first.")
    elif "preference" not in st.session_state or "budget" not in st.session_state:
        st.error("Please complete 'Event Preferences' first.")
    else:
        style = st.session_state["scanned_style"]
        pref = st.session_state["preference"]
        bud = st.session_state["budget"]
        sets = search_products_simple(style, pref, bud)
        if not sets:
            st.warning("No complete wardrobe sets could be formed. Please try refining your style or preferences.")
        else:
            st.session_state["recommendations"] = sets
            for outfit in sets:
                st.subheader(outfit["set_name"])
                st.write(outfit["summary"])
                st.write("**Items Included:**")
                for it in outfit["items"]:
                    st.write(f"- **{it['item']}**: {it['name']} | Price: {it['price']}")
                    st.markdown(f"[Buy Now]({it['link']})")
                    if st.button(f"Add {it['name']} to Cart", key=f"{outfit['set_name']}_{it['item']}"):
                        add_to_cart({
                            "name": it["name"],
                            "price": it["price"],
                            "link": it["link"]
                        })
                st.markdown("---")
            if st.button("Save All Selections"):
                user_id = "user_1"
                save_selection(user_id, sets)
                st.success("Selections saved!")

elif menu == "Saved Selections":
    st.header("Your Saved Wardrobe Selections")
    user_id = "user_1"
    selections = retrieve_selection(user_id)
    if selections:
        for sel in selections:
            st.subheader("Saved Selection")
            st.write(sel)
    else:
        st.info("No selections saved yet.")

elif menu == "Show document.json":
    st.header("Raw JSON Document with Top 3 Recommendations per Category")
    try:
        with open("document.json", "r") as f:
            data = json.load(f)
        # Use exact category keys that were used during extraction
        categories_to_show = ["Apparel", "Coat", "Shoes", "Tie"]
        for cat in categories_to_show:
            st.subheader(f"Category: {cat}")
            if cat in data:
                content = data[cat]
                # Check for expected format: either directly a products list or nested under "data"
                if isinstance(content, dict):
                    if "data" in content and "products" in content["data"]:
                        products = content["data"]["products"]
                    elif "products" in content:
                        products = content["products"]
                    else:
                        products = []
                    for idx, prod in enumerate(products[:3]):
                        st.write(f"**Recommendation {idx+1}:**")
                        st.write(f"- Name: {prod.get('name', 'N/A')}")
                        st.write(f"- Price: {prod.get('price', 'N/A')}")
                        st.write(f"- Link: {prod.get('link', 'N/A')}")
                        st.markdown("---")
                else:
                    st.write(f"Data for {cat} is not in expected format: {content}")
            else:
                st.write(f"No data found for category: {cat}")
    except Exception as e:
        st.error(f"Error reading document.json: {e}")

elif menu == "Cart":
    st.header("Your Cart")
    if st.session_state["cart"]:
        for idx, item in enumerate(st.session_state["cart"]):
            st.write(f"**Item {idx+1}:** {item['name']} | Price: {item['price']}")
            st.markdown(f"[Buy Now Link]({item['link']})")
        if st.button("Clear Cart"):
            clear_cart()
            st.success("Cart cleared.")
    else:
        st.info("Your cart is empty.")
