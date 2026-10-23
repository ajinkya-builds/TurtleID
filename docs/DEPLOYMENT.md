# Deployment Guide: Mobile Web Application

The `docs/` folder contains a lightweight HTML/JS mobile web application that can be hosted statically on GitHub Pages.

Because GitHub Pages only supports static files, it cannot run the Python-based AI models (YOLOv8, BioCLIP) used by Chelonid-ID. Therefore, we use a decoupled architecture:

1. **Frontend**: Hosted on GitHub Pages.
2. **Image Storage**: Handled via Google Drive (using Google Apps Script).
3. **AI Inference Backend**: A FastAPI Python backend hosted on a cloud provider like **Render** or **Google Cloud Run**.

---

## 1. Hosting the Frontend on GitHub Pages

1. In your GitHub repository, go to **Settings**.
2. Scroll down to the **Pages** section on the left sidebar.
3. Under **Build and deployment**, set the Source to **Deploy from a branch**.
4. Select the `main` branch (or whichever branch you are using) and choose the `/docs` folder.
5. Click **Save**.
6. After a few minutes, your site will be live at `https://<your-username>.github.io/<repository-name>/`.

---

## 2. Setting up Google Drive Image Storage

The frontend allows users to take or upload photos. These photos can be automatically saved to a Google Drive folder.

1. Create a new folder in your Google Drive to store the incoming images (e.g., "Chelonid-ID-Images").
2. Look at the URL of that folder in your browser. It will look like `https://drive.google.com/drive/folders/ABC123xyz`. Copy the ID part (`ABC123xyz`).
3. Go to [Google Apps Script](https://script.google.com/) and create a new project.
4. Copy the entire contents of the `docs/google_drive_backend.js` file and paste it into the script editor (`Code.gs`).
5. In the script editor, replace `'YOUR_GOOGLE_DRIVE_FOLDER_ID_HERE'` with the Folder ID you copied in step 2.
6. Click **Deploy > New deployment**.
7. Select **Web app** as the type.
8. Set **Execute as** to `Me` and **Who has access** to `Anyone`.
9. Click **Deploy** and authorize the script when prompted.
10. Copy the resulting **Web app URL**.
11. Open `docs/index.html` in this repository and replace `YOUR_GOOGLE_APPS_SCRIPT_URL_HERE` with the URL you just copied.

---

## 3. Setting up the AI Inference Backend on Render

To actually run the species identification model, you must deploy `api.py` as a backend service that the frontend can call. The default backend relies on **BioCLIP**, a foundation model, which means it works immediately without requiring you to train YOLOv8 weights first.

### Prerequisites & Dependencies
The backend requires dependencies from both the main app and the offline matcher stack.

When deploying, your build process must install:
```bash
pip install -r requirements.txt
pip install -r requirements-matcher.txt
pip install fastapi uvicorn pydantic python-multipart
```

### Hugging Face Cache Notes
The first time `api.py` boots, the `BioClipMatcher` will download the necessary foundation models from Hugging Face.
* By default, the code caches these in `models/hf_cache/`.
* **Important:** Ensure your hosting environment has enough disk space (at least 2-3 GB) to store these weights.
* On some platforms (like Render Free Tier), disk state is ephemeral. The models will be downloaded on every cold boot, which can delay the first request. A paid tier with persistent disk is recommended for production.

### Deploying to Render
1. Create an account on [Render](https://render.com/).
2. Create a new **Web Service** and connect your GitHub repository.
3. **Build Command:**
   ```bash
   pip install -r requirements.txt && pip install -r requirements-matcher.txt && pip install fastapi uvicorn pydantic python-multipart
   ```
4. **Start Command:**
   ```bash
   uvicorn api:app --host 0.0.0.0 --port $PORT
   ```
5. Choose your instance type. (A tier with at least 2GB of RAM is strongly recommended to hold the BioCLIP models in memory).
6. Click **Create Web Service**.

### Connecting Frontend to the API
Once your API is live (e.g., `https://chelonid-api-xyz.onrender.com`):
1. Open `docs/index.html`.
2. Locate the line: `const PYTHON_API_URL = 'YOUR_PYTHON_API_URL_HERE/identify';`
3. Replace the placeholder string with your live Render URL, ensuring it ends with `/identify`. Example:
   ```javascript
   const PYTHON_API_URL = 'https://chelonid-api-xyz.onrender.com/identify';
   ```
4. Commit and push the changes to GitHub. Your mobile app will now perform real species identification!
