import cv2
import numpy as np
import logging
from ultralytics import YOLO
from insightface.app import FaceAnalysis
import onnxruntime as ort
from datetime import datetime
import configparser
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.models import Faces
from dotenv import load_dotenv
import os

config = configparser.ConfigParser()
config.read('/app/configs/config.ini')

FOOTAGE_DIR = config['Paths']['FOOTAGE_DIR']
MODEL_PATH = config['Paths']['MODEL_PATH']

logging.basicConfig(
    filename='/app/logs/kinwatch_facetrain.log',
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@172.17.0.1:5432/kinai")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    """Initialize the faces database (handled by init_db.sql)."""
    logger.info("Ensuring faces table exists via SQLAlchemy")
    # Table is created via init_db.sql or SQLAlchemy Base.metadata.create_all
    return SessionLocal()

def init_models():
    try:
        yolo_model = YOLO(MODEL_PATH)
        logger.info("YOLOv8 model loaded")
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        if 'CUDAExecutionProvider' not in ort.get_available_providers():
            logger.warning("CUDAExecutionProvider not available, using CPU")
            providers = ['CPUExecutionProvider']
        face_app = FaceAnalysis(name='buffalo_l', providers=providers)
        face_app.prepare(ctx_id=0, det_size=(640, 640))
        logger.info("InsightFace model initialized")
        return yolo_model, face_app
    except Exception as e:
        logger.error(f"Model initialization failed: {e}")
        raise

def process_video(video_path, name, yolo_model, face_app, session):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            raise ValueError("Video file could not be opened")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        embeddings = []

        for _ in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break

            results = yolo_model(frame)
            boxes = results[0].boxes.xyxy.cpu().numpy()

            for box in boxes:
                x1, y1, x2, y2 = map(int, box[:4])
                face_img = frame[y1:y2, x1:x2]
                if face_img.size == 0:
                    continue

                face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                faces = face_app.get(face_img_rgb)
                if faces:
                    embedding = faces[0].embedding
                    embeddings.append(embedding)

        cap.release()

        if embeddings:
            avg_embedding = np.mean(embeddings, axis=0).tobytes()
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            face = Faces(name=name, embedding=avg_embedding, last_updated=timestamp)
            session.add(face)
            session.commit()
            logger.info(f"Stored embedding for {name} at {timestamp}")
        else:
            logger.warning(f"No faces detected for {name}")

    except Exception as e:
        logger.error(f"Error processing {video_path}: {e}")
        raise

def main():
    session = None
    yolo_model = None
    face_app = None
    try:
        session = init_db()
        yolo_model, face_app = init_models()

        video_files = [
            (os.path.join(FOOTAGE_DIR, 'ila_footage.mp4'), 'ila'),
            (os.path.join(FOOTAGE_DIR, 'logu_footage.mp4'), 'logu')
        ]

        for video_path, name in video_files:
            if not os.path.exists(video_path):
                logger.error(f"Video file not found: {video_path}")
                continue
            process_video(video_path, name, yolo_model, face_app, session)

        logger.info("Training completed successfully")
    except Exception as e:
        logger.error(f"Main execution failed: {e}")
    finally:
        if session:
            session.close()

if __name__ == "__main__":
    main()