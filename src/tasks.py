from celery import Celery
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from chains import ResponseChain
import os
import json
import requests
from dotenv import load_dotenv
from models import EventLog, EventType, SystemState, MusicSchedule
from datetime import datetime, time
import pytz
from astral.sun import sun
from astral import LocationInfo
import re
import logging

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Celery configuration
app = Celery(
    'tasks',
    broker='redis://172.17.0.1:6379/0',
    backend='redis://172.17.0.1:6379/0'
)
app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='Australia/Melbourne',
    enable_utc=False,
    broker_connection_retry_on_startup=True
)

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@172.17.0.1:5432/kinai")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Sunset and weather logic
MELBOURNE = LocationInfo("Melbourne", "Australia", "Australia/Melbourne", -37.8136, 144.9631)

def is_after_sunset():
    now = datetime.now(pytz.timezone('Australia/Melbourne'))
    s = sun(MELBOURNE.observer, date=now.date(), tzinfo=MELBOURNE.timezone)
    return now.time() > s['sunset'].time()

def get_weather_action(temp):
    if temp is None:
        logger.warning("Temperature not found, skipping weather action")
        return None
    if temp < 18:
        return {"device": "air_conditioner", "action": "turn_on", "mode": "heat"}
    elif temp > 24:
        return {"device": "air_conditioner", "action": "turn_on", "mode": "cool"}
    return None

def update_no_detection_state(db, event_type):
    state = db.query(SystemState).first()
    if not state:
        state = SystemState(no_detection_count=0, no_detection_flag=0)
        db.add(state)
    
    if event_type == EventType.NO_DETECTION:
        state.no_detection_count += 1
        logger.info(f"NO_DETECTION count: {state.no_detection_count}")
        if state.no_detection_count >= 10:
            state.no_detection_flag = 1
            logger.info("Set no_detection_flag to True")
    else:
        state.no_detection_count = 0
        state.no_detection_flag = 0
        logger.info("Reset no_detection_count and no_detection_flag")
    
    db.commit()
    return state

@app.task
def process_event_task(event_id):
    logger.info(f"Processing event_id={event_id} in Celery task")
    db = SessionLocal()
    try:
        event = db.query(EventLog).filter(EventLog.id == event_id).first()
        if not event:
            logger.error(f"Event {event_id} not found")
            return {"status": "error", "event_id": event_id, "error": "Event not found"}

        # Update no_detection state
        state = update_no_detection_state(db, event.event_type)

        # Deserialize payload
        payload = event.payload
        if isinstance(payload, str):
            payload = json.loads(payload)

        webhook_url = os.getenv("WEBHOOK_URL")
        if not webhook_url:
            logger.error("WEBHOOK_URL not set")
            return {"status": "error", "event_id": event_id, "error": "WEBHOOK_URL not set"}

        webhook_payload = {"event_id": event_id, "actions": []}

        # If no_detection_flag is set, send turn_off actions for NO_DETECTION and skip other automations
        if state.no_detection_flag == 1 and event.event_type == EventType.NO_DETECTION:
            webhook_payload.update({
                "event_type": "NO_DETECTION",
                "status": "no_detection",
                "timestamp": payload.get("timestamp")
            })
            webhook_payload["actions"].extend([
                {"service": "home_assistant", "device": "family_room_lights", "action": "turn_off"},
                {"service": "home_assistant", "device": "air_conditioner", "action": "turn_off"}
            ])
        elif state.no_detection_flag == 1:
            logger.info(f"Skipping automation for {event.event_type} due to no_detection_flag")
            return {"status": "skipped", "event_id": event_id, "message": "Automations disabled due to no_detection_flag"}
        else:
            # Normal automation processing
            chain = ResponseChain()
            after_sunset = is_after_sunset()

            if event.event_type == EventType.SINGLE_KNOWN:
                name = payload.get("name")
                timestamp = payload.get("timestamp")
                response = chain.run(name, timestamp)
                logger.info(f"ResponseChain output for SINGLE_KNOWN: {response}")
                temp_match = re.search(r'(\d+\.?\d*)\s*°C', response)
                temp = float(temp_match.group(1)) if temp_match else None
                webhook_payload.update({
                    "event_type": "SINGLE_KNOWN",
                    "name": name,
                    "response": response,
                    "timestamp": timestamp
                })
                webhook_payload["actions"].extend([
                    {"service": "jellyfin", "action": "play_greeting", "name": name},
                    {"service": "telegram", "action": "send_message", "message": f"Welcome home, {name}."},
                    {"service": "smart_display", "action": "show_message", "message": f"Welcome Home, {name}"},
                    {"service": "home_assistant", "device": "family_room_lights", "action": "turn_on"}
                ])
                if after_sunset or name == "ila":
                    webhook_payload["actions"].append(
                        {"service": "home_assistant", "device": "family_room_lights", "action": "turn_on"}
                    )
                weather_action = get_weather_action(temp)
                if weather_action:
                    webhook_payload["actions"].append(
                        {"service": "home_assistant", **weather_action}
                    )

            elif event.event_type == EventType.FAMILY_PROFILE:
                names = payload.get("names")
                timestamp = payload.get("timestamp")
                response = chain.run(", ".join(names), timestamp)
                logger.info(f"ResponseChain output for FAMILY_PROFILE: {response}")
                temp_match = re.search(r'(\d+\.?\d*)\s*°C', response)
                temp = float(temp_match.group(1)) if temp_match else None
                webhook_payload.update({
                    "event_type": "FAMILY_PROFILE",
                    "names": names,
                    "response": response,
                    "timestamp": timestamp
                })
                webhook_payload["actions"].extend([
                    {"service": "jellyfin", "action": "play_greeting", "names": names},
                    {"service": "smart_display", "action": "show_message", "message": "Welcome Home"},
                    {"service": "home_assistant", "device": "family_room_lights", "action": "turn_on"}
                ])
                if after_sunset:
                    webhook_payload["actions"].append(
                        {"service": "home_assistant", "device": "family_room_lights", "action": "turn_on"}
                    )
                weather_action = get_weather_action(temp)
                if weather_action:
                    webhook_payload["actions"].append(
                        {"service": "home_assistant", **weather_action}
                    )

            elif event.event_type == EventType.UNKNOWN_WITH_KNOWN:
                name = payload.get("name")
                unknown_id = payload.get("unknown_id")
                image_base64 = payload.get("image_base64")
                timestamp = payload.get("timestamp")
                response = chain.run(name, timestamp)
                logger.info(f"ResponseChain output for UNKNOWN_WITH_KNOWN: {response}")
                webhook_payload.update({
                    "event_type": "UNKNOWN_WITH_KNOWN",
                    "name": name,
                    "unknown_id": unknown_id,
                    "image_base64": image_base64,
                    "response": response,
                    "timestamp": timestamp
                })
                webhook_payload["actions"].extend([
                    {"service": "telegram", "action": "send_message", "message": f"Guest with {name}", "image_base64": image_base64},
                    {"service": "smart_display", "action": "show_message", "message": "Welcome Home"},
                    {"service": "home_assistant", "device": "family_room_lights", "action": "turn_on"}
                ])

            elif event.event_type == EventType.SUSPECT:
                unknown_id = payload.get("unknown_id")
                image_base64 = payload.get("image_base64")
                timestamp = payload.get("timestamp")
                webhook_payload.update({
                    "event_type": "SUSPECT",
                    "unknown_id": unknown_id,
                    "image_base64": image_base64,
                    "timestamp": timestamp,
                    "emergency": True
                })
                webhook_payload["actions"].extend([
                    {"service": "telegram", "action": "send_message", "message": "Suspect detected!", "image_base64": image_base64},
                    {"service": "home_assistant", "device": "family_room_lights", "action": "turn_off"},
                    {"service": "home_assistant", "device": "alarm", "action": "trigger"}
                ])

            elif event.event_type == EventType.NO_DETECTION:
                webhook_payload.update({
                    "event_type": "NO_DETECTION",
                    "status": "no_detection",
                    "timestamp": payload.get("timestamp")
                })

        # Send webhook
        logger.info(f"Sending webhook to {webhook_url} with payload: {webhook_payload}")
        try:
            webhook_response = requests.post(webhook_url, json=webhook_payload, timeout=5)
            webhook_response.raise_for_status()
            logger.info(f"Successfully sent webhook to {webhook_url}")
        except requests.RequestException as e:
            logger.error(f"Failed to send webhook: {str(e)}")
            return {"status": "error", "event_id": event_id, "error": f"Webhook failed: {str(e)}"}

        return {"status": "success", "event_id": event_id, "response": webhook_payload.get("response", "")}
    except Exception as e:
        logger.error(f"Error processing event_id={event_id}: {str(e)}")
        return {"status": "error", "event_id": event_id, "error": str(e)}
    finally:
        db.close()

@app.task
def music_scheduler():
    now = datetime.now(pytz.timezone('Australia/Melbourne')).time()
    with SessionLocal() as db:
        state = db.query(SystemState).first()
        if state and state.no_detection_flag == 1:
            logger.info("Skipping music playback due to no_detection_flag")
            return
        schedules = db.query(MusicSchedule).filter(
            MusicSchedule.start_time <= now,
            MusicSchedule.end_time >= now
        ).all()
        if not schedules:
            logger.info("No active music schedule")
            return
        webhook_url = os.getenv("WEBHOOK_URL")
        for schedule in schedules:
            webhook_payload = {
                "event_type": "MUSIC_PLAYBACK",
                "actions": [
                    {
                        "service": "jellyfin",
                        "action": "play_playlist",
                        "playlist_id": schedule.playlist_id
                    }
                ]
            }
            try:
                requests.post(webhook_url, json=webhook_payload, timeout=5)
                logger.info(f"Started playback for playlist {schedule.playlist_id}")
            except requests.RequestException as e:
                logger.error(f"Failed to start playback: {str(e)}")