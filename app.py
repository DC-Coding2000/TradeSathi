# app.py
import os
import sqlite3
from flask import Flask, render_template, request, redirect, url_for, flash, send_file
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from datetime import datetime
import base64

# --- CONFIG ---
APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(APP_DIR, 'model')
UPLOADS_DIR = os.path.join(MODEL_DIR, 'uploads')
MODEL_FILE = os.path.join(MODEL_DIR, 'scm_model.joblib')
DATA_FILE = os.path.join(MODEL_DIR, 'training_data.csv')
DB_FILE = os.path.join(APP_DIR, 'data.db')
ADMIN_PASSWORD = 'adminpass'  # change in production

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)

app = Flask(__name__)
app.secret_key = 'replace-with-a-secure-random-key'  # change in production

# --- Database helpers ---
def init_db():
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        email TEXT UNIQUE,
        phone TEXT,
        role TEXT,
        doc_path TEXT,
        verified INTEGER DEFAULT 0,
        created_at TEXT
    )''')
    cur.execute('''CREATE TABLE IF NOT EXISTS escrow (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        amount REAL,
        status TEXT,
        created_at TEXT
    )''')
    conn.commit()
    conn.close()

init_db()

# --- MODEL TRAIN / LOAD ---
from sklearn.exceptions import NotFittedError

def build_and_save_model(train_df=None):
    # combine default CSV and any uploaded CSVs (train_df overrides)
    if train_df is None:
        df = pd.read_csv(DATA_FILE)
    else:
        df = train_df
    X = df[["distance_km", "weight_kg", "mode", "urgency", "volume_m3"]]
    cat_cols = ["mode"]
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ], remainder="passthrough")
    model_cost = make_pipeline(pre, RandomForestRegressor(n_estimators=80, random_state=42))
    model_time = make_pipeline(pre, RandomForestRegressor(n_estimators=50, random_state=42))
    model_carbon = make_pipeline(pre, RandomForestRegressor(n_estimators=50, random_state=42))
    model_cost.fit(X, df["base_cost"])
    model_time.fit(X, df["delivery_time_days"])
    model_carbon.fit(X, df["carbon_kg"])
    joblib.dump({"cost": model_cost, "time": model_time, "carbon": model_carbon}, MODEL_FILE)
    return {"cost": model_cost, "time": model_time, "carbon": model_carbon}

def load_model():
    if not os.path.exists(MODEL_FILE):
        return build_and_save_model()
    else:
        try:
            return joblib.load(MODEL_FILE)
        except Exception:
            return build_and_save_model()

MODELS = load_model()

# --- Utilities ---
def suggest_mode(distance_km, urgency, weight_kg):
    if urgency == 1 and distance_km > 400 and weight_kg < 200:
        return "air"
    if distance_km > 1200 and weight_kg > 500:
        return "sea"
    if distance_km <= 500:
        return "road"
    return "sea"

def estimate_route_hint(source, destination, distance_km, mode):
    hint = f"Approx distance {distance_km} km between {source} → {destination}."
    if mode == "road":
        hint += " Use consolidated truck / LTL for small shipments; prefer express trucking firms for urgent delivery."
    elif mode == "air":
        hint += " Choose direct cargo or express cargo via nearest international airport; pack for air-handling."
    else:
        hint += " Use full-container-load (FCL) for large shipments; consider coastal feeder routes for lower cost."
    return hint

def recommended_docs():
    return [
        "UDYAM Registration proof",
        "GSTIN copy",
        "PAN / Aadhaar of proprietor / directors",
        "Bank account & AD Code",
        "Import Export Code (IEC)",
        "Commercial invoice",
        "Packing list",
        "Bill of Lading / Airway Bill",
        "Export promotion council certificate (if applicable)",
        "Trade license / Trademark (if required)"
    ]

def suggest_logistics_partners(mode):
    partners = {
        "road": ["FastRoad Logistics", "LocalHaul Express", "CityFreight Co."],
        "air": ["SkyCargo Intl", "AirXpress Logistics"],
        "sea": ["BlueWave Shipping", "OceanGate Lines"]
    }
    return partners.get(mode, partners["road"])

# --- Routes ---
@app.route('/')
def index():
    return render_template('index.html')

# Registration
@app.route('/register', methods=['GET','POST'])
def register():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        phone = request.form.get('phone')
        role = request.form.get('role', 'seller')
        doc = request.files.get('doc')
        doc_path = ''
        if doc:
            filename = f"doc_{int(datetime.utcnow().timestamp())}_{doc.filename}"
            doc_path = os.path.join(UPLOADS_DIR, filename)
            doc.save(doc_path)
        conn = sqlite3.connect(DB_FILE)
        cur = conn.cursor()
        cur.execute('INSERT OR IGNORE INTO users (name,email,phone,role,doc_path,created_at) VALUES (?,?,?,?,?,?)',
                    (name,email,phone,role,doc_path,datetime.utcnow().isoformat()))
        conn.commit(); conn.close()
        flash('Registration submitted. Admin will verify you soon.', 'success')
        return redirect(url_for('index'))
    return render_template('register.html')

# Simple admin login
@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        pwd = request.form.get('password')
        if pwd == ADMIN_PASSWORD:
            resp = redirect(url_for('admin'))
            resp.set_cookie('is_admin','1')
            return resp
        flash('Wrong admin password', 'danger')
        return redirect(url_for('login'))
    return render_template('login.html')

def is_admin():
    return request.cookies.get('is_admin') == '1'

# Admin panel - list users and approve
@app.route('/admin')
def admin():
    if not is_admin():
        return redirect(url_for('login'))
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute('SELECT id,name,email,phone,role,verified,created_at FROM users ORDER BY created_at DESC')
    users = cur.fetchall()
    conn.close()
    return render_template('admin.html', users=users)

@app.route('/approve/<int:user_id>')
def approve(user_id):
    if not is_admin():
        return redirect(url_for('login'))
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute('UPDATE users SET verified=1 WHERE id=?', (user_id,))
    conn.commit(); conn.close()
    flash('User approved', 'success')
    return redirect(url_for('admin'))

# CSV upload & retrain
@app.route('/upload', methods=['GET','POST'])
def upload():
    if request.method == 'POST':
        f = request.files.get('csv')
        if not f:
            flash('No file', 'danger'); return redirect(url_for('upload'))
        filename = f"upload_{int(datetime.utcnow().timestamp())}_{f.filename}"
        path = os.path.join(UPLOADS_DIR, filename)
        f.save(path)
        # read and validate
        try:
            df_new = pd.read_csv(path)
            required = set(['distance_km','weight_kg','mode','urgency','volume_m3','base_cost','delivery_time_days','carbon_kg'])
            if not required.issubset(set(df_new.columns)):
                flash('CSV missing required columns', 'danger'); return redirect(url_for('upload'))
            # retrain model using combined dataset (default + all uploads)
            df_base = pd.read_csv(DATA_FILE)
            uploads = []
            for fname in os.listdir(UPLOADS_DIR):
                try:
                    uploads.append(pd.read_csv(os.path.join(UPLOADS_DIR,fname)))
                except Exception:
                    pass
            df_all = pd.concat([df_base] + uploads, ignore_index=True)
            build_and_save_model(df_all)
            flash('Upload saved and model retrained', 'success')
            return redirect(url_for('dashboard'))
        except Exception as e:
            flash('Failed to process CSV: ' + str(e), 'danger')
            return redirect(url_for('upload'))
    return render_template('upload.html')

# Prediction (same as earlier)
@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    source = data.get('source', 'Origin')
    destination = data.get('destination', 'Destination')
    try:
        distance_km = float(data.get('distance_km', '100'))
    except:
        distance_km = 100.0
    try:
        weight_kg = float(data.get('weight_kg', '10'))
    except:
        weight_kg = 10.0
    mode = data.get('mode', 'road')
    urgency = int(data.get('urgency', '0'))
    try:
        volume_m3 = float(data.get('volume_m3', '0.1'))
    except:
        volume_m3 = 0.1

    X = pd.DataFrame([{
        'distance_km': distance_km,
        'weight_kg': weight_kg,
        'mode': mode,
        'urgency': urgency,
        'volume_m3': volume_m3
    }])
    models = load_model()
    pred_cost = models['cost'].predict(X)[0]
    pred_time = models['time'].predict(X)[0]
    pred_carbon = models['carbon'].predict(X)[0]
    pred_cost = float(np.round(pred_cost,2))
    pred_time = float(np.round(pred_time,2))
    pred_carbon = float(np.round(pred_carbon,2))
    mode_suggest = suggest_mode(distance_km, urgency, weight_kg)
    route_hint = estimate_route_hint(source, destination, distance_km, mode_suggest)
    docs = recommended_docs()
    partners = suggest_logistics_partners(mode_suggest)
    breakdown = {
        'freight': round(pred_cost * 0.7, 2),
        'insurance': round(pred_cost * 0.05, 2),
        'customs_docs_and_handling': round(pred_cost * 0.1, 2),
        'other_fees': round(pred_cost * 0.15, 2)
    }
    return render_template('result.html', source=source, destination=destination, distance_km=distance_km,
                           weight_kg=weight_kg, pred_cost=pred_cost, pred_time=pred_time,
                           pred_carbon=pred_carbon, mode_suggest=mode_suggest, route_hint=route_hint,
                           docs=docs, partners=partners, breakdown=breakdown)

# Dashboard (uses uploads + base data)
@app.route('/dashboard')
def dashboard():
    # combine datasets
    df_base = pd.read_csv(DATA_FILE)
    uploads = []
    for fname in os.listdir(UPLOADS_DIR):
        try:
            uploads.append(pd.read_csv(os.path.join(UPLOADS_DIR,fname)))
        except Exception:
            pass
    if uploads:
        df = pd.concat([df_base] + uploads, ignore_index=True)
    else:
        df = df_base
    # KPIs
    total_shipments = len(df)
    avg_cost = df['base_cost'].mean()
    total_carbon = df['carbon_kg'].sum()
    # chart: avg cost by distance bin
    fig = plt.figure(figsize=(6,3))
    df.groupby(pd.cut(df['distance_km'], bins=[0,100,500,1000,5000]))['base_cost'].mean().plot(kind='bar')
    plt.title('Avg Cost by Distance Bin')
    plt.tight_layout()
    buf = BytesIO(); fig.savefig(buf, format='png'); buf.seek(0)
    img_b64 = 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode('ascii')
    return render_template('dashboard.html', total_shipments=total_shipments, avg_cost=round(avg_cost,2),
                           total_carbon=round(total_carbon,2), chart=img_b64)

# Invoice PDF generator (simple)
@app.route('/invoice', methods=['GET','POST'])
def invoice():
    if request.method == 'POST':
        form = request.form
        seller = form.get('seller','Seller')
        buyer = form.get('buyer','Buyer')
        items_desc = form.get('items','1 unit')
        try:
            amount = float(form.get('amount','0'))
        except:
            amount = 0.0
        buffer = BytesIO()
        p = canvas.Canvas(buffer, pagesize=A4)
        p.setFont('Helvetica-Bold', 14)
        p.drawString(50,800,'TradeSathi — Invoice')
        p.setFont('Helvetica', 11)
        p.drawString(50,780,f'Date: {datetime.utcnow().strftime("%Y-%m-%d")}')
        p.drawString(50,760,f'Seller: {seller}')
        p.drawString(50,740,f'Buyer: {buyer}')
        p.drawString(50,720,f'Items: {items_desc}')
        p.drawString(50,700,f'Amount (INR): ₹ {amount:.2f}')
        p.showPage(); p.save()
        buffer.seek(0)
        return send_file(buffer, as_attachment=True, download_name='invoice.pdf', mimetype='application/pdf')
    return render_template('invoice_form.html')

# Mock escrow flows
@app.route('/escrow', methods=['GET','POST'])
def escrow():
    if request.method == 'POST':
        user_email = request.form.get('email')
        try:
            amount = float(request.form.get('amount',0))
        except:
            amount = 0.0
        conn = sqlite3.connect(DB_FILE)
        cur = conn.cursor()
        # find user id
        cur.execute('SELECT id FROM users WHERE email=?', (user_email,))
        r = cur.fetchone()
        user_id = r[0] if r else None
        cur.execute('INSERT INTO escrow (user_id,amount,status,created_at) VALUES (?,?,?,?)', (user_id,amount,'held',datetime.utcnow().isoformat()))
        conn.commit(); conn.close()
        flash('Mock escrow created (held)', 'success')
        return redirect(url_for('escrow'))
    # show escrow entries
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute('SELECT e.id,u.email,e.amount,e.status,e.created_at FROM escrow e LEFT JOIN users u ON e.user_id=u.id ORDER BY e.created_at DESC')
    rows = cur.fetchall(); conn.close()
    return render_template('escrow.html', rows=rows)

@app.route('/escrow_release/<int:escrow_id>')
def escrow_release(escrow_id):
    if not is_admin(): return redirect(url_for('login'))
    conn = sqlite3.connect(DB_FILE); cur = conn.cursor()
    cur.execute('UPDATE escrow SET status=? WHERE id=?', ('released', escrow_id))
    conn.commit(); conn.close()
    flash('Escrow released (mock)', 'success')
    return redirect(url_for('escrow'))

if __name__ == '__main__':
    app.run(debug=True)
