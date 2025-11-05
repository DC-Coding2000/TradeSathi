TradeSathi (TSCBSE) — Full Demo Quick Start

Prereqs:
- Python 3.9+ installed
- Internet access to install packages once

Steps:
1. Create folder TSCBSE and subfolders as shown.
2. Copy each file into exact path.
3. Open terminal and cd into TSCBSE.
4. (Optional but recommended) Create virtualenv:
   python3 -m venv venv
   source venv/bin/activate   (Windows: venv\Scripts\activate)
5. Install dependencies:
   pip install -r requirements.txt
6. Run:
   python app.py
7. Open browser: http://127.0.0.1:5000

Admin:
- Admin password (demo): adminpass
- Login at /login → then go to /admin to approve users.

Notes:
- Uploaded CSVs saved in model/uploads/ and combined with base training_data.csv for retraining.
- To reset model, delete model/scm_model.joblib and restart; it will rebuild from training_data.csv (and uploads).
- Escrow is mock-only; no real payments.
- Replace SECRET keys and admin password before any public deployment.
