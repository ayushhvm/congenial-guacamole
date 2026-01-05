# 📸 Face Recognition Attendance System

A robust, AI-powered attendance tracking system built with **Django** and **InsightFace**. It allows detailed student management, automated face-based attendance marking, and comprehensive reporting.

## 🌟 Features

*   **AI-Powered Recognition**: Uses **InsightFace (ArcFace)** for state-of-the-art accuracy (99%+).
*   **Anti-Dilution Logic**: Implements Cosine Similarity with dynamic thresholding to ensure accuracy doesn't drop as you add more students.
*   **Role-Based Access**:
    *   **Teachers**: Manage students, train models, create sessions, mark attendance, and view reports.
    *   **Students**: View their own attendance history and stats.
*   **Automated Attendance**: "Auto-Capture" mode for sessions to automatically snap and process photos at set intervals.
*   **Bulk Management**: Command-line tools to register hundreds of students and teachers instantly.
*   **Analytics**: Built-in tools to analyze model health, identifying lookalikes and bad data.

---

## 🛠️ Technology Stack

*   **Backend**: Django 5.x (Python 3.11+)
*   **AI Engine**: InsightFace (RetinaFace for detection, ArcFace for embedding) + Scikit-Learn (KNN)
*   **Database**: SQLite (Default) / PostgreSQL (Supported)
*   **Frontend**: HTML5, CSS3, JavaScript (Vanilla)
*   **Queueing**: (Optional) Redis/Celery for Async tasks.

---

## 🚀 Installation & Setup

### 1. Prerequisites
*   Python 3.10 or higher
*   C++ Build Tools (for InsightFace dependencies)

### 2. Clone and Install
```bash
git clone <repository-url>
cd Trackapp

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Mac/Linux
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### 3. Setup Database
```bash
# Apply migrations
python manage.py migrate
```

### 4. Create Admin/Teacher
You can create a teacher via command line:
```bash
python manage.py register_teacher --id "T001" --name "Ayush V Mangalgi" --password "securepass123" --email "ayush@example.com"
```

### 5. Run Server
```bash
python manage.py runserver
```
Access the app at `http://127.0.0.1:8000`.

---

## 📖 Usage Guide

### 🧑‍🏫 Teacher Workflow

1.  **Login**: Use your Teacher ID and Password.
2.  **Register Students**:
    *   **Manual**: Go to "Register Student", enter details, and upload 5-10 clear face photos.
    *   **Bulk**: See [Management Commands](#-management-commands) below.
3.  **Train Model**:
    *   Go to "Train V2" (or "Train Model" in dashboard).
    *   Click "Train Now". This processes all registered student faces into the recognition engine.
    *   *Note: Retrain whenever you add new students.*
4.  **Create Session**:
    *   Go to "Sessions" -> "New Session".
    *   Set Course Name (e.g., "CS101"), Date, and Time.
    *   **Auto-Capture**: Enable to have the system auto-record attendance every X seconds (requires camera feed).
5.  **Mark Attendance**:
    *   **Upload**: Upload a group photo of the class. The system detects faces and marks them present.
    *   **Live**: (If enabled) Use the camera interface.

### 🧑‍🎓 Student Workflow

1.  **Login**: Use Student ID (e.g., "S001") and Password.
2.  **Dashboard**: View attendance percentage, total present/absent classes.
3.  **My Attendance**: Filter history by course or date.

---

## ⚡ Management Commands

The system includes powerful CLI tools for administration.

### 1. Bulk Register Students
Registers students from a CSV/Excel or structured folder.
```bash
python manage.py register_bulk
```
*   Expects images in `yamages/` folder named by Student ID (e.g., `yamages/001/`).
*   Automatically detects faces, generates embeddings, and saves to DB.

### 2. Analyze Model Health 🏥
Checks your dataset quality. Critical for maintaining high accuracy.
```bash
python manage.py analyze_model_quality
```
*   **Separability**: Tells you how distinct students are from each other.
*   **Outliers**: Finds "bad images" that don't look like the student.
*   **Threshold**: Suggests the perfect strictness (e.g., 0.50) to avoid false positives.

### 3. Register Teacher
```bash
python manage.py register_teacher --id <ID> --name <NAME> --password <PASS>
```

### 4. Clean Corrupted Data
If necessary, remove empty embeddings.
```bash
python manage.py clear_corrupted_embeddings
```

---

## 🔧 Troubleshooting

**Q: "Model not loaded" error?**
A: You must train the model at least once. Log in as Teacher -> Train Model -> Click Train.

**Q: False Positives (Wrong person recognized)?**
A: 
1. Run `python manage.py analyze_model_quality`.
2. check for "Lookalikes" or bad data.
3. Increase the **Threshold** when marking attendance (e.g., from 0.5 to 0.6).

**Q: "No face detected" in valid image?**
A: Ensure the image is not upside down. The system uses a high-accuracy detector (RetinaFace), so it handles side profiles well, but extreme angles or extreme blur can fail.
