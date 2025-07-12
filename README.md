# TalkSim 🎙️
**A Full-Stack Multi-Speaker Text-to-Speech Web App using React, Flask & Google Cloud TTS**

TalkSim is a containerized web application that transforms multi-speaker dialogues into expressive, natural-sounding audio. It uses Google Cloud’s neural Text-to-Speech (TTS) API to synthesize realistic multi-speaker speech from textual dialogues and provides a clean web interface for interaction.

---

## 🚀 Features

- 🧠 **Multi-Speaker Voice Rendering**  
  Converts structured dialogue (e.g., "Speaker 1: Hi") into alternating speaker voices.

- 🎧 **Audio Synthesis with Google Cloud TTS**  
  Uses Google Cloud's `en-US-Studio-Multispeaker` voice with speaker tags `R` and `S`.

- 🔊 **Audio Preview + Download**  
  Users can listen to or download the final MP3 output.

- 📦 **Containerized Deployment**  
  Frontend (React + Nginx) and backend (Flask) are containerized with Docker.

---

## 🧠 How It Works

1. **User Input**  
   User pastes a structured conversation like:
   Speaker 1: Hello!
   Speaker 2: Hi, how are you?

2. **Request Sent**  
The frontend POSTs the text to `/process-text` on the backend.

3. **Text Parsing & Chunking**  
The backend splits text into speaker turns and chunks (<2000 chars each).

4. **TTS Synthesis**  
Google Cloud TTS synthesizes audio using multi-speaker markup.

5. **Concatenation**  
MP3 clips are stitched using MoviePy and returned to the frontend.

6. **Playback/Download**  
User can preview and download the resulting dialogue MP3.

---

## 🛠️ Technologies Used

### Frontend
- **React** (UI and state handling)
- **Axios** (API requests)
- **Nginx** (serving production build)
- **Docker** (multi-stage build)

### Backend
- **Flask** (REST API)
- **Flask-CORS** (CORS support)
- **Google Cloud Text-to-Speech** (speech synthesis)
- **MoviePy** (audio concatenation)
- **Docker** (Python container)

---

## 🧪 Setup & Development

### 🐳 Prerequisites

- Docker
- Google Cloud account with TTS API enabled
- Google service account key JSON

### 🔧 Local Setup (via Docker Compose)

```bash
# Clone the repo
git clone https://github.com/kalakh22/TalkSim.git
cd TalkSim

# Place your GCP key at backend/key.json

# Build and run the containers
docker-compose up --build

