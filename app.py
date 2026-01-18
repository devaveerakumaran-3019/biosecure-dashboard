from flask import Flask, request, jsonify, render_template, session, redirect, url_for
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from bson.objectid import ObjectId
from processor import ForensicEngine
from datetime import datetime
import os
import cv2
import numpy as np

app = Flask(__name__)
app.secret_key = "super_secure_forensic_key_2026"

# ---------------- CONFIG & SETUP ---------------- #
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, 'static')
ENROLLED_DIR = os.path.join(STATIC_DIR, 'enrolled')
TEMP_DIR = os.path.join(STATIC_DIR, 'temp')

# Create directories if they don't exist
os.makedirs(ENROLLED_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# Database Connection
client = MongoClient("mongodb+srv://devaveerakumaransciet_db_user:DmPjjb81RbAFb79u@engineered-db.nriat39.mongodb.net/?appName=engineered-db")
db = client["biosecure_db"]
users_col = db["users"]
prints_col = db["fingerprints"]

engine = ForensicEngine()

# ---------------- HELPER: VISUALIZATION ---------------- #
def generate_match_image(query_path, train_path, output_filename):
    """ Generates side-by-side comparison with green matching lines """
    try:
        img1 = cv2.imread(query_path, cv2.IMREAD_GRAYSCALE) # Query
        img2 = cv2.imread(train_path, cv2.IMREAD_GRAYSCALE) # Database Record

        if img1 is None or img2 is None: return None

        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)

        if des1 is None or des2 is None: return None

        # FLANN Matcher
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)

        # Ratio Test
        good_matches = []
        matches_mask = [[0, 0] for i in range(len(matches))]
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.75 * n.distance:
                matches_mask[i] = [1, 0]
                good_matches.append(m)

        draw_params = dict(matchColor=(0, 255, 0), singlePointColor=(255, 0, 0),
                           matchesMask=matches_mask, flags=cv2.DrawMatchesFlags_DEFAULT)

        result_img = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)

        output_path = os.path.join(TEMP_DIR, output_filename)
        cv2.imwrite(output_path, result_img)
        
        return f"/static/temp/{output_filename}"
    except Exception as e:
        print(f"Visualization Error: {e}")
        return None

# ---------------- ROUTES ---------------- #
@app.route("/")
def home(): return render_template("index.html")

@app.route("/login")
def login_page(): return render_template("login.html")

@app.route("/signup")
def signup_page(): return render_template("signup.html")

@app.route("/admin")
def admin():
    if session.get("role") != "admin": return redirect("/login")
    return render_template("admin_dashboard.html", 
                           user=session.get("user"), 
                           admin_id=f"ADM-{os.urandom(2).hex().upper()}")

@app.route("/user")
def user():
    if session.get("role") != "user": return redirect("/login")
    return render_template("user_dashboard.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")

# ---------------- AUTH API ---------------- #
@app.route("/api/signup", methods=["POST"])
def api_signup():
    data = request.json
    if users_col.find_one({"username": data["username"]}):
        return jsonify({"success": False, "message": "Username taken"})
    
    users_col.insert_one({
        "fullname": data["fullname"],
        "username": data["username"],
        "password": generate_password_hash(data["password"]),
        "role": data["role"],
        "created_at": datetime.utcnow()
    })
    return jsonify({"success": True})

@app.route("/api/login", methods=["POST"])
def api_login():
    data = request.json
    user = users_col.find_one({"username": data["username"]})
    if user and check_password_hash(user["password"], data["password"]):
        session["user"] = data["username"]
        session["role"] = user["role"]
        return jsonify({"success": True, "redirect": "/admin" if user["role"] == "admin" else "/user"})
    return jsonify({"success": False, "message": "Invalid credentials"})

# ---------------- FINGERPRINT CRUD API (THIS WAS MISSING) ---------------- #

@app.route("/api/enroll", methods=["POST"])
def enroll():
    if session.get("role") != "admin": return jsonify({"success": False, "message": "Unauthorized"})

    img_file = request.files.get("image")
    name = request.form.get("name")
    hand = request.form.get("hand")
    finger = request.form.get("finger")
    notes = request.form.get("notes")
    
    # 1. Process Features
    features = engine.process_image(img_file)
    if not features or len(features) < 5:
        return jsonify({"success": False, "message": "Image quality too low or no fingerprint detected."})

    # 2. Save Physical Image (Critical for Visualization)
    img_file.seek(0) 
    filename = f"{ObjectId()}.png"
    save_path = os.path.join(ENROLLED_DIR, filename)
    img_file.save(save_path)

    # 3. Save to DB
    prints_col.insert_one({
        "name": name,
        "hand": hand,
        "finger": finger,
        "notes": notes,
        "features": features,
        "image_path": filename, # Store filename reference
        "enrolled_by": session.get("user"),
        "created_at": datetime.utcnow()
    })
    return jsonify({"success": True, "message": f"Successfully enrolled {name}"})

@app.route("/api/prints", methods=["GET"])
def get_prints():
    if session.get("role") != "admin": return jsonify({"success": False, "message": "Unauthorized"})
    
    # Exclude 'features' to keep response light
    cursor = prints_col.find({}, {"features": 0}).sort("created_at", -1)
    
    results = []
    for doc in cursor:
        doc["_id"] = str(doc["_id"])
        results.append(doc)
    
    return jsonify({"success": True, "data": results})

@app.route("/api/prints/<id>", methods=["DELETE"])
def delete_print(id):
    if session.get("role") != "admin": return jsonify({"success": False, "message": "Unauthorized"})
    
    result = prints_col.delete_one({"_id": ObjectId(id)})
    if result.deleted_count > 0:
        return jsonify({"success": True, "message": "Record deleted."})
    return jsonify({"success": False, "message": "Record not found."})

@app.route("/api/prints/<id>", methods=["PUT"])
def update_print(id):
    if session.get("role") != "admin": return jsonify({"success": False, "message": "Unauthorized"})
    
    data = request.json
    prints_col.update_one({"_id": ObjectId(id)}, {"$set": {
        "name": data.get("name"),
        "hand": data.get("hand"),
        "finger": data.get("finger"),
        "notes": data.get("notes")
    }})
    return jsonify({"success": True, "message": "Record updated."})

# ---------------- MATCH API ---------------- #

@app.route("/api/match", methods=["POST"])
def match():
    if 'image' not in request.files:
        return jsonify({"success": False, "message": "No image uploaded"})
    
    file = request.files['image']
    
    # 1. Save Query Image Temporarily
    query_filename = f"query_{datetime.now().timestamp()}.png"
    query_path = os.path.join(TEMP_DIR, query_filename)
    file.save(query_path)

    # 2. Process Features
    with open(query_path, 'rb') as f:
        query_features = engine.process_image(f)

    if not query_features:
        return jsonify({"success": True, "match": False, "message": "No features detected."})

    # 3. Search Database
    best_score = 0
    best_match = None
    all_records = prints_col.find({})
    
    for record in all_records:
        if 'features' in record and record['features']:
            score, is_match = engine.match(query_features, record['features'])
            if is_match and score > best_score:
                best_score = score
                best_match = record

    # 4. Generate Visualization
    visual_url = None
    if best_match and 'image_path' in best_match:
        db_image_path = os.path.join(ENROLLED_DIR, best_match['image_path'])
        if os.path.exists(db_image_path):
            result_filename = f"match_{best_match['_id']}_{datetime.now().timestamp()}.jpg"
            visual_url = generate_match_image(query_path, db_image_path, result_filename)

    if best_match:
        return jsonify({
            "success": True, 
            "match": True, 
            "identity": best_match.get('name', 'Unknown'), 
            "score": best_score,
            "visual_url": visual_url
        })
    else:
        return jsonify({"success": True, "match": False, "score": best_score})

if __name__ == "__main__":
    app.run(debug=True, port=5000)