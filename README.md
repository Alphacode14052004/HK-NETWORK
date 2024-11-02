# HKNETWORK App

This is a video replaying application built with React Native for the frontend and Flask for the backend. The app allows users to upload videos, which are combined into a single video playlist that can be played back within the application.

## Features

- Upload video files (supports mp4, avi, mov, mkv formats).
- Automatically combine uploaded videos into a single playlist.
- Stream combined video for playback.
- User-friendly interface for video management.

## Technologies Used

- **Frontend**: React Native, TypeScript
- **Backend**: Flask, SQLAlchemy
- **Video Processing**: MoviePy
- **Database**: SQLite

## Getting Started

### Prerequisites

- Node.js and npm installed on your system.
- Python 3.7 or higher installed.
- Flask and required Python packages.

### Frontend Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/video-replaying-app.git
   cd video-replaying-app/frontend

## Install dependencies:

bash

npm install

## Start the React Native app:

bash

    npm start

Backend Setup

    Navigate to the backend directory:

    bash

### cd video-replaying-app/backend

## Create a virtual environment and activate it:

bash

python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

Install required packages:

bash

pip install -r requirements.txt

Run the Flask app:

bash

    python app.py

API Endpoints

    GET /api/videos: Retrieve the list of uploaded videos.
    GET /api/playlist: Retrieve the combined video for playback.
    POST /upload: Upload a new video.

Running the App

    Ensure both the backend and frontend servers are running.
    Open your React Native app on a simulator or physical device.
    Use the app to upload videos and view the combined playback.

Usage

    Upload videos through the app.
    The app automatically combines the uploaded videos.
    Play the combined video directly in the app.

License

This project is licensed under the MIT License - see the LICENSE file for details.
Acknowledgments

    React Native Documentation
    Flask Documentation
    MoviePy Documentation
