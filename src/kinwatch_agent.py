import cv2
import numpy as np
import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.modules.block import C2f
from insightface.app import FaceAnalysis
import onnxruntime as ort
from scipy.spatial.distance import cosine
from threading import Thread, Lock
import ulid
import uuid
import configparser
import torch
import torch.nn.modules.container
from src.models import Faces, KnownPersons, Detections
from src.visitor_tracker import init_unknown_visitors_db, check_previous_visitor, store_unknown_visitor, update_unknown_visitor
from src.surveillance_agent import process_unknown, trigger_single_known, trigger_family_profile, trigger_unknown_with_known, trigger_no_detection
from src.family_profiles import init_family_profiles_db, check_family_profile, add_family_profile
import insightface.utils.transform
from dotenv import load_dotenv
import os
import time
from datetime import datetime

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    filename='/app/logs/kinwatch.log'
)

logger.info("Kinwatch agent starting, logging initialized")

# Patch insightface.utils.transform to suppress FutureWarning
original_lstsq = np.linalg.lstsq
def patched_lstsq(a, b, rcond=None):
    return original_lstsq(a, b, rcond=rcond)
np.linalg.lstsq = patched_lstsq

# Load environment variables
load_dotenv()

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@172.17.0.1:5432/kinai")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

config = configparser.ConfigParser()
config.read('/app/configs/config.ini')

BASE_DIR = config['Paths']['BASE_DIR']
DETECTED_DIR = config['Paths']['DETECTED_DIR']
MODEL_PATH = config['Paths']['MODEL_PATH']

COOLDOWN_PERIOD = int(config['Settings']['COOLDOWN_PERIOD'])
VERIFICATION_WINDOW = int(config['Settings']['VERIFICATION_WINDOW'])
MIN_DETECTIONS = int(config['Settings']['MIN_DETECTIONS'])
THRESHOLD = float(config['Settings']['THRESHOLD'])
SAVE_COOLDOWN = int(config['Settings']['SAVE_COOLDOWN'])
MIN_CONFIDENCE = float(config['Settings']['MIN_CONFIDENCE'])
NO_DETECTION_INTERVAL = int(config['Settings']['NO_DETECTION_INTERVAL'])

os.makedirs(DETECTED_DIR, exist_ok=True)

faces_db_lock = Lock()
kinwatch_db_lock = Lock()
unknown_visitors_db_lock = Lock()

def clean_old_images():
    cutoff_time = time.time() - 24 * 3600
    for root, dirs, files in os.walk(DETECTED_DIR):
        for file in files:
            file_path = os.path.join(root, file)
            if os.path.getmtime(file_path) < cutoff_time:
                logger.debug(f"Deleting old image: {file_path}")
                os.remove(file_path)

def init_faces_db():
    logger.debug("Ensuring faces table exists via SQLAlchemy")
    # Table is created via init_db.sql or SQLAlchemy Base.metadata.create_all
    logger.info("Faces database ready")

def init_kinwatch_db():
    logger.debug("Ensuring kinwatch tables exist via SQLAlchemy")
    # Tables (known_persons, detections) are created via init_db.sql
    logger.info("Kinwatch database ready")

def init_models():
    logger.debug("Initializing YOLO and FaceAnalysis models")
    original_torch_load = torch.load
    def patched_load(f, *args, **kwargs):
        kwargs['weights_only'] = False
        return original_torch_load(f, *args, **kwargs)
    torch.load = patched_load
    try:
        yolo_model = YOLO(MODEL_PATH)
        logger.info("Successfully loaded YOLO model with weights_only=False")
    finally:
        torch.load = original_torch_load
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    if 'CUDAExecutionProvider' not in ort.get_available_providers():
        logger.warning("CUDAExecutionProvider not available, falling back to CPU")
        providers = ['CPUExecutionProvider']
    face_app = FaceAnalysis(name='buffalo_l', root='/app/models/', providers=providers)
    face_app.prepare(ctx_id=0, det_size=(320, 320))
    logger.info("Models initialized successfully")
    return yolo_model, face_app

def compare_embedding(embedding, threshold=THRESHOLD):
    with faces_db_lock:
        logger.debug("Comparing embedding with known faces")
        with SessionLocal() as db:
            faces = db.query(Faces).all()
            if not faces:
                logger.debug("No known faces in database")
                return None, 0.0
            best_name, best_similarity = None, 0.0
            for face in faces:
                stored_embedding = np.frombuffer(face.embedding, dtype=np.float32)
                similarity = 1 - cosine(embedding, stored_embedding)
                if similarity > best_similarity:
                    best_name, best_similarity = face.name, similarity
            if best_similarity > threshold:
                logger.info(f"Match found: {best_name} with similarity {best_similarity:.2f}")
                return best_name, best_similarity
            logger.debug(f"No match found, best similarity: {best_similarity:.2f}")
            return None, best_similarity

def update_known_person(name, embedding, image_path, timestamp):
    with kinwatch_db_lock:
        logger.debug(f"Updating known person: {name}")
        with SessionLocal() as db:
            person = db.query(KnownPersons).filter(KnownPersons.name == name).first()
            if person:
                person.last_seen = timestamp
                person.detection_count += 1
                person.embedding = embedding.tobytes()
                person.reference_image = image_path
                logger.info(f"Updated known person {name}, detection count: {person.detection_count}")
            else:
                person = KnownPersons(
                    name=name,
                    embedding=embedding.tobytes(),
                    first_seen=timestamp,
                    last_seen=timestamp,
                    reference_image=image_path
                )
                db.add(person)
                logger.info(f"Inserted new known person: {name}")
            db.commit()

def log_detection(timestamp, device, status, image_path=None, camera_id=None, unknown_id=None):
    with kinwatch_db_lock:
        logger.debug(f"Logging detection: {status}, camera_id: {camera_id}, unknown_id: {unknown_id}")
        with SessionLocal() as db:
            detection = Detections(
                timestamp=timestamp,
                device=device,
                status=status,
                image_path=image_path,
                camera_id=camera_id,
                unknown_id=unknown_id
            )
            db.add(detection)
            db.commit()
            logger.info(f"Detection logged: {status} for camera {camera_id}")

def create_labeled_image(frame, detections):
    logger.debug("Creating labeled image with detections")
    labeled_frame = frame.copy()
    for detection in detections:
        x1, y1, x2, y2 = detection['box']
        identity = detection['identity']
        confidence = detection['confidence']
        label = f"{identity} ({confidence:.2f})"
        cv2.rectangle(labeled_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(labeled_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return labeled_frame

def process_video(video_path, camera_id):
    logger.info(f"[Camera {camera_id}] Starting video processing for {video_path}")
    DETECTED_DIR_CAM = os.path.join(DETECTED_DIR, f"camera_{camera_id}")
    os.makedirs(DETECTED_DIR_CAM, exist_ok=True)

    init_faces_db()
    init_kinwatch_db()
    init_unknown_visitors_db()
    yolo_model, face_app = init_models()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"[Camera {camera_id}] Failed to open video capture for {video_path}")
        return
    logger.info(f"[Camera {camera_id}] Video capture opened successfully")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 25)
    logger.debug(f"[Camera {camera_id}] Capture properties: width={cap.get(cv2.CAP_PROP_FRAME_WIDTH)}, height={cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}, fps={cap.get(cv2.CAP_PROP_FPS)}")

    window_name = f"Camera {camera_id}"
    try:
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        logger.info(f"[Camera {camera_id}] Display window initialized: {window_name}")
    except Exception as e:
        logger.error(f"[Camera {camera_id}] Failed to initialize display window: {str(e)}")
        logger.warning(f"[Camera {camera_id}] Continuing without video display")

    unknown_trackers = {}
    detected_persons = {}
    last_saved = {}
    frame_count = 0
    last_no_detection_trigger = 0

    try:
        while cap.isOpened():
            try:
                ret, frame = cap.read()
                if not ret:
                    logger.warning(f"[Camera {camera_id}] Failed to read frame from {video_path}")
                    time.sleep(1)
                    continue

                frame_count += 1
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                current_time = time.time()
                logger.debug(f"[Camera {camera_id}] Processing frame {frame_count} at {timestamp}")
                logger.debug(f"[Camera {camera_id}] Frame dimensions: {frame.shape}")

                for ulid_key in list(unknown_trackers.keys()):
                    if current_time - unknown_trackers[ulid_key]['timestamp'] > COOLDOWN_PERIOD:
                        logger.debug(f"[Camera {camera_id}] Removing expired tracker: {ulid_key}")
                        del unknown_trackers[ulid_key]

                frame_detections = []
                if frame_count % 2 == 0:
                    logger.debug(f"[Camera {camera_id}] Running YOLO detection on frame {frame_count}")
                    results = yolo_model(frame, verbose=False, conf=0.45)
                    boxes = results[0].boxes
                    person_boxes = [box for box in boxes if int(box.cls) == 0]
                    logger.debug(f"[Camera {camera_id}] Detected {len(person_boxes)} person boxes")
                    if not person_boxes:
                        logger.debug(f"[Camera {camera_id}] No person boxes detected in frame {frame_count}")

                    if person_boxes:
                        for box in person_boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            box_area = (x2-x1) * (y2-y1)
                            logger.debug(f"[Camera {camera_id}] Bounding box coordinates: ({x1},{y1},{x2},{y2}), area: {box_area}")
                            if box_area < 10000:
                                logger.debug(f"[Camera {camera_id}] Skipping small box: ({x1},{y1},{x2},{y2})")
                                continue
                            person_img = frame[y1:y2, x1:x2]
                            head_height = int((y2-y1) * 0.4)
                            padding = int(head_height * 0.6)
                            face_y2 = min(y1 + head_height*2, y2)
                            face_x1, face_y1 = max(0, x1 - padding), max(0, y1 - padding)
                            face_x2, face_y2 = min(frame.shape[1], x2 + padding), min(frame.shape[0], face_y2 + padding)
                            face_img = frame[face_y1:face_y2, face_x1:face_x2]

                            if face_img.size == 0:
                                logger.debug(f"[Camera {camera_id}] Empty face image, skipping")
                                continue
                            face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                            faces = face_app.get(face_img_rgb)
                            logger.debug(f"[Camera {camera_id}] Face detection result: {len(faces)} faces detected")
                            if faces:
                                logger.debug(f"[Camera {camera_id}] Face detection score: {faces[0].det_score:.2f}")

                            if not faces or faces[0].det_score <= 0.3:
                                identity = "Unknown"
                                confidence = 0.0
                                embedding = None
                            else:
                                embedding = faces[0].embedding
                                name, similarity = compare_embedding(embedding)
                                identity = name if name else "Unknown"
                                confidence = similarity
                            logger.debug(f"[Camera {camera_id}] Face detected: {identity}, confidence: {confidence:.2f}")

                            unknown_id = None
                            visit_count = 0
                            first_seen = None
                            last_seen = None
                            if identity == "Unknown" and embedding is not None:
                                with unknown_visitors_db_lock:
                                    logger.debug(f"[Camera {camera_id}] Checking previous visitor for unknown face")
                                    matched_ulid, match_similarity, match_visit_count, first_seen, last_seen, prev_camera_id, prev_image_path = check_previous_visitor(embedding, camera_id)
                                    if matched_ulid:
                                        unknown_id = matched_ulid
                                        visit_count = match_visit_count
                                        tracker = unknown_trackers.get(matched_ulid, {
                                            'embedding': embedding,
                                            'timestamp': current_time,
                                            'confidence': confidence,
                                            'image': person_img,
                                            'camera_id': camera_id,
                                            'processed': False,
                                            'detection_count': 0,
                                            'first_seen': current_time
                                        })
                                        tracker['detection_count'] += 1
                                        tracker['timestamp'] = current_time
                                        if confidence > tracker['confidence']:
                                            tracker['confidence'] = confidence
                                            tracker['image'] = person_img
                                        unknown_trackers[matched_ulid] = tracker
                                        logger.info(f"[Camera {camera_id}] Matched previous visitor: ULID {unknown_id}, visit count: {visit_count}")
                                    else:
                                        try:
                                            unknown_id = str(ulid.ULID())
                                        except Exception:
                                            unknown_id = str(uuid.uuid4())
                                        first_seen = timestamp
                                        last_seen = timestamp
                                        unknown_trackers[unknown_id] = {
                                            'embedding': embedding,
                                            'timestamp': current_time,
                                            'confidence': confidence,
                                            'image': person_img,
                                            'camera_id': camera_id,
                                            'processed': False,
                                            'detection_count': 1,
                                            'first_seen': current_time
                                        }
                                        logger.info(f"[Camera {camera_id}] New unknown visitor: ULID {unknown_id}")

                            if identity not in detected_persons:
                                detected_persons[identity] = {'count': 0, 'best_confidence': 0, 'last_image_path': None}
                            detected_persons[identity]['count'] += 1
                            save_image = False
                            image_path = None

                            can_save = identity == "Unknown" or confidence >= MIN_CONFIDENCE
                            in_cooldown = identity in last_saved and current_time - last_saved.get(identity, 0) <= SAVE_COOLDOWN
                            if can_save and not in_cooldown:
                                day_key = datetime.now().strftime("%Y%m%d")
                                daily_dir = os.path.join(DETECTED_DIR_CAM, day_key)
                                os.makedirs(daily_dir, exist_ok=True)
                                logger.debug(f"[Camera {camera_id}] Preparing to save image for {identity} at {daily_dir}")

                                if identity == "Unknown":
                                    target_dir = os.path.join(daily_dir, "Unknown")
                                    os.makedirs(target_dir, exist_ok=True)
                                    save_path = os.path.join(target_dir, f"unknown_{unknown_id or 'no_face'}_{confidence:.2f}.jpg")
                                    logger.debug(f"[Camera {camera_id}] Saving unknown image: {save_path}, confidence: {confidence:.2f}")
                                    if cv2.imwrite(save_path, person_img, [cv2.IMWRITE_JPEG_QUALITY, 80]):
                                        save_image = True
                                        image_path = save_path
                                        if unknown_id:
                                            unknown_trackers[unknown_id]['image_path'] = image_path
                                        logger.info(f"[Camera {camera_id}] Unknown image saved: {save_path}, ULID: {unknown_id or 'no_face'}, confidence: {confidence:.2f}")
                                        if unknown_id and embedding is not None:
                                            with unknown_visitors_db_lock:
                                                if visit_count > 0:
                                                    update_unknown_visitor(unknown_id, embedding, timestamp, camera_id, image_path, visit_count)
                                                    logger.debug(f"[Camera {camera_id}] Updated unknown visitor: ULID {unknown_id}")
                                                else:
                                                    store_unknown_visitor(unknown_id, embedding, timestamp, camera_id, image_path)
                                                    logger.debug(f"[Camera {camera_id}] Stored new unknown visitor: ULID {unknown_id}")
                                    else:
                                        logger.error(f"[Camera {camera_id}] Failed to save unknown image: {save_path}")
                                else:
                                    target_dir = os.path.join(daily_dir, identity)
                                    os.makedirs(target_dir, exist_ok=True)
                                    save_path = os.path.join(target_dir, f"{identity}_{confidence:.2f}.jpg")
                                    logger.debug(f"[Camera {camera_id}] Saving known person image: {save_path}, confidence: {confidence:.2f}")
                                    if cv2.imwrite(save_path, person_img, [cv2.IMWRITE_JPEG_QUALITY, 80]):
                                        image_path = save_path
                                        detected_persons[identity]['last_image_path'] = image_path
                                        update_known_person(identity, embedding, image_path, timestamp)
                                        save_image = True
                                        logger.info(f"[Camera {camera_id}] Known person image saved: {save_path}, Identity: {identity}, confidence: {confidence:.2f}")
                                    else:
                                        logger.error(f"[Camera {camera_id}] Failed to save known person image: {save_path}")
                            else:
                                logger.debug(f"[Camera {camera_id}] Skipping save for {identity}: can_save={can_save}, in_cooldown={in_cooldown}, confidence={confidence:.2f}, MIN_CONFIDENCE={MIN_CONFIDENCE}")

                            if save_image:
                                last_saved[identity] = current_time
                                logger.debug(f"[Camera {camera_id}] Updated last_saved timestamp for {identity}")

                            frame_detections.append({
                                'box': (x1, y1, x2, y2),
                                'identity': identity,
                                'confidence': confidence,
                                'unknown_id': unknown_id,
                                'visit_count': visit_count
                            })

                        if frame_detections:
                            logger.debug(f"[Camera {camera_id}] {len(frame_detections)} detections in frame")
                            if len(detected_persons) == 1 and identity != "Unknown":
                                trigger_single_known(identity, confidence, timestamp, image_path, camera_id)
                                log_detection(timestamp, f"camera_{camera_id}", "single_known", image_path, camera_id)
                            elif len(detected_persons) > 1:
                                family_ids = [id for id, data in detected_persons.items() if id != "Unknown" and data['count'] >= MIN_DETECTIONS]
                                if family_ids:
                                    family_profile = check_family_profile(family_ids)
                                    if family_profile:
                                        trigger_family_profile(family_profile, timestamp, image_path, camera_id)
                                        log_detection(timestamp, f"camera_{camera_id}", "family_profile", image_path, camera_id)
                                    else:
                                        trigger_unknown_with_known(family_ids, timestamp, image_path, camera_id)
                                        log_detection(timestamp, f"camera_{camera_id}", "unknown_with_known", image_path, camera_id)
                                else:
                                    for ulid_key, tracker in list(unknown_trackers.items()):
                                        if tracker['camera_id'] != camera_id or tracker['processed']:
                                            continue
                                        time_since_first_seen = current_time - tracker['first_seen']
                                        if time_since_first_seen > VERIFICATION_WINDOW and tracker['detection_count'] >= MIN_DETECTIONS:
                                            logger.info(f"[Camera {camera_id}] Processing unknown visitor: ULID {ulid_key}, detections: {tracker['detection_count']}")
                                            process_unknown(
                                                tracker['image'],
                                                tracker['embedding'],
                                                timestamp,
                                                camera_id,
                                                ulid_key,
                                                visit_count,
                                                first_seen,
                                                last_seen
                                            )
                                            log_detection(timestamp, f"camera_{camera_id}", "suspect_detected", tracker.get('image_path'), camera_id, ulid_key)
                                            tracker['processed'] = True
                                        elif time_since_first_seen > VERIFICATION_WINDOW and tracker['detection_count'] < MIN_DETECTIONS:
                                            logger.debug(f"[Camera {camera_id}] Removing tracker {ulid_key} due to insufficient detections: {tracker['detection_count']}")
                                            del unknown_trackers[ulid_key]
                        else:
                            logger.debug(f"[Camera {camera_id}] No detections in frame")
                            if current_time - last_no_detection_trigger >= NO_DETECTION_INTERVAL:
                                trigger_no_detection(timestamp, camera_id)
                                log_detection(timestamp, f"camera_{camera_id}", "no_detection", None, camera_id)
                                last_no_detection_trigger = current_time

                try:
                    display_frame = create_labeled_image(frame, frame_detections) if frame_detections else frame
                    cv2.imshow(window_name, display_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        logger.info(f"[Camera {camera_id}] User pressed 'q', stopping video processing")
                        break
                except Exception as e:
                    logger.error(f"[Camera {camera_id}] Failed to display frame: {str(e)}")
                    logger.warning(f"[Camera {camera_id}] Disabling video display for this camera")
                    cv2.destroyWindow(window_name)
                    window_name = None

                time.sleep(0.1)

            except Exception as e:
                logger.error(f"[Camera {camera_id}] Error processing frame: {str(e)}")
                time.sleep(1)
                continue

    finally:
        cap.release()
        if window_name:
            cv2.destroyWindow(window_name)
        cv2.destroyAllWindows()
        logger.info(f"[Camera {camera_id}] Video processing stopped for {video_path}")

def main():
    logger.info("Starting Kinwatch agent")
    clean_old_images()
    cameras = [config['Cameras']['CAMERA_0']]
    threads = []
    max_retries = 3

    for i, device in enumerate(cameras):
        for attempt in range(max_retries):
            logger.info(f"[Camera {i}] Attempt {attempt + 1} to start thread for {device}")
            t = Thread(target=process_video, args=(device, i))
            t.start()
            t.join(5)
            if not t.is_alive():
                logger.error(f"[Camera {i}] Thread failed to start for {device}")
                if attempt + 1 == max_retries:
                    logger.error(f"[Camera {i}] Max retries reached for {device}")
                    break
                time.sleep(5)
                continue
            threads.append(t)
            logger.info(f"[Camera {i}] Thread started successfully for {device}")
            break

    for t in threads:
        t.join()
    logger.info("Kinwatch agent stopped")

if __name__ == "__main__":
    main()