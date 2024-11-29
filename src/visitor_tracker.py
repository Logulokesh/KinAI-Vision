import logging
import numpy as np
from scipy.spatial.distance import cosine
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.models import UnknownVisitors
from dotenv import load_dotenv
import os

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@172.17.0.1:5432/kinai")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_unknown_visitors_db():
    """Initialize unknown visitors database (handled by init_db.sql)."""
    logger.debug("Ensuring unknown visitors table exists via SQLAlchemy")
    # Table is created via init_db.sql or SQLAlchemy Base.metadata.create_all
    logger.info("Unknown visitors database ready")

def check_previous_visitor(embedding, camera_id):
    """Check for previous visitor."""
    logger.debug("Checking for previous visitor")
    with SessionLocal() as db:
        visitors = db.query(UnknownVisitors).all()
        best_ulid, best_similarity, best_visit_count, best_first_seen, best_last_seen, best_camera_id, best_image_path = None, 0.0, 0, None, None, None, None
        for visitor in visitors:
            stored_embedding = np.frombuffer(visitor.embedding, dtype=np.float32)
            similarity = 1 - cosine(embedding, stored_embedding)
            if similarity > best_similarity and similarity > 0.5:
                best_ulid = visitor.ulid
                best_similarity = similarity
                best_visit_count = visitor.visit_count
                best_first_seen = visitor.first_seen
                best_last_seen = visitor.last_seen
                best_camera_id = visitor.camera_id
                best_image_path = visitor.image_path
        if best_ulid:
            logger.debug(f"Matched previous visitor: ULID {best_ulid}, similarity {best_similarity:.2f}")
        else:
            logger.debug("No matching previous visitor found")
        return best_ulid, best_similarity, best_visit_count, best_first_seen, best_last_seen, best_camera_id, best_image_path

def store_unknown_visitor(ulid, embedding, timestamp, camera_id, image_path):
    """Store a new unknown visitor."""
    logger.debug(f"Storing new unknown visitor: ULID {ulid}")
    with SessionLocal() as db:
        visitor = UnknownVisitors(
            ulid=ulid,
            embedding=embedding.tobytes(),
            first_seen=timestamp,
            last_seen=timestamp,
            visit_count=1,
            camera_id=camera_id,
            image_path=image_path
        )
        db.add(visitor)
        db.commit()
        logger.info(f"Stored new unknown visitor: ULID {ulid}")

def update_unknown_visitor(ulid, embedding, timestamp, camera_id, image_path, visit_count):
    """Update an existing unknown visitor."""
    logger.debug(f"Updating unknown visitor: ULID {ulid}")
    with SessionLocal() as db:
        visitor = db.query(UnknownVisitors).filter(UnknownVisitors.ulid == ulid).first()
        if visitor:
            visitor.embedding = embedding.tobytes()
            visitor.last_seen = timestamp
            visitor.visit_count = visit_count + 1
            visitor.camera_id = camera_id
            visitor.image_path = image_path
            db.commit()
            logger.info(f"Updated unknown visitor: ULID {ulid}, visit count: {visit_count + 1}")