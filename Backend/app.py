from flask import Flask, request, jsonify, send_file, render_template, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
import os
from moviepy.editor import VideoFileClip, concatenate_videoclips
import tempfile
from datetime import datetime
import shutil

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///videos.db'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['COMBINED_VIDEO_PATH'] = 'static/combined_video.mp4'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

db = SQLAlchemy(app)

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(os.path.dirname(app.config['COMBINED_VIDEO_PATH']), exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class Video(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    filepath = db.Column(db.String(255), nullable=False)
    upload_date = db.Column(db.DateTime, default=datetime.utcnow)
    order = db.Column(db.Integer, default=0)
    filesize = db.Column(db.Integer)  # in bytes

    def to_dict(self):
        return {
            'id': self.id,
            'filename': self.filename,
            'upload_date': self.upload_date.strftime('%Y-%m-%d %H:%M:%S'),
            'order': self.order,
            'filesize': self.filesize
        }

with app.app_context():
    db.create_all()

def combine_videos():
    videos = Video.query.order_by(Video.order).all()
    if not videos:
        return

    try:
        clips = [VideoFileClip(video.filepath) for video in videos]
        final_clip = concatenate_videoclips(clips, method="compose")

        # Write to a temporary file first, then move to the final path
        temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        final_clip.write_videofile(temp_output.name, codec="libx264")
        temp_output.close()
        
        # Move the temporary file to the combined video path
        shutil.move(temp_output.name, app.config['COMBINED_VIDEO_PATH'])
        
        # Close clips to release resources
        for clip in clips:
            clip.close()
        final_clip.close()
    except Exception as e:
        print(f"Error combining videos: {e}")

@app.route('/')
def index():
    videos = Video.query.order_by(Video.order).all()
    return render_template('index.html', videos=videos)

@app.route('/upload', methods=['POST'])
def upload():
    if 'video' not in request.files:
        flash('No video file uploaded', 'error')
        return redirect(url_for('index'))
    
    video = request.files['video']
    if video.filename == '':
        flash('No selected file', 'error')
        return redirect(url_for('index'))
    
    if not allowed_file(video.filename):
        flash('Invalid file type. Allowed types: mp4, avi, mov, mkv', 'error')
        return redirect(url_for('index'))
    
    try:
        filename = secure_filename(video.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        video.save(filepath)
        
        filesize = os.path.getsize(filepath)
        
        new_video = Video(
            filename=filename,
            filepath=filepath,
            order=Video.query.count(),
            filesize=filesize
        )
        db.session.add(new_video)
        db.session.commit()
        
        # Combine videos after adding a new one
        combine_videos()
        
        flash('Video uploaded and combined successfully!', 'success')
    except Exception as e:
        flash(f'Error uploading video: {str(e)}', 'error')
    
    return redirect(url_for('index'))

@app.route('/delete/<int:video_id>', methods=['POST'])
def delete_video(video_id):
    video = Video.query.get_or_404(video_id)
    try:
        # Delete file from filesystem
        if os.path.exists(video.filepath):
            os.remove(video.filepath)
        
        # Delete from database
        db.session.delete(video)
        db.session.commit()
        
        # Recombine videos after deletion
        combine_videos()
        
        flash('Video deleted and playlist updated!', 'success')
    except Exception as e:
        flash(f'Error deleting video: {str(e)}', 'error')
    
    return redirect(url_for('index'))

# API Routes (for mobile app)
@app.route('/api/videos', methods=['GET'])
def get_videos():
    videos = Video.query.order_by(Video.order).all()
    return jsonify({'videos': [video.to_dict() for video in videos]})

@app.route('/api/playlist', methods=['GET'])
def get_combined_video():
    if not os.path.exists(app.config['COMBINED_VIDEO_PATH']):
        return jsonify({'error': 'No combined video available'}), 404
    return send_file(app.config['COMBINED_VIDEO_PATH'], mimetype='video/mp4')

if __name__ == '__main__':
    app.run(debug=True)
