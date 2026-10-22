# Deployment Guide: Mobile Web Application

The `docs/` folder contains a lightweight HTML/JS mobile web application that can be hosted statically on GitHub Pages.

Because GitHub Pages only supports static files, it cannot run the Python-based AI models (YOLOv8, BioCLIP) used by Chelonid-ID. Therefore, we use a decoupled architecture:

1. **Frontend**: Hosted on GitHub Pages.
2. **Image Storage**: Handled via Google Drive (using Google Apps Script).
3. **AI Inference Backend**: A FastAPI python backend hosted on a free/low-cost cloud provider.

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

## 3. Setting up the AI Inference Backend (api.py)

To actually run the species identification model, you must deploy the `api.py` python backend that the frontend can call.

1. Ensure the backend dependencies are installed (`pip install -r requirements.txt fastapi uvicorn pydantic`).
2. Deploy the root directory (containing `api.py`) to a hosting provider like **Render**, **Heroku**, or **Google Cloud Run**.
3. *Note*: The backend requires the trained model files (e.g. `models/chelonid_cls.pt`, `models/gallery.npz`, etc) to be present or properly built/downloaded on the host.
4. Once your API is live (e.g., `https://my-turtle-api.onrender.com`), open `docs/index.html` and replace `YOUR_PYTHON_API_URL_HERE` with your API URL.

---
