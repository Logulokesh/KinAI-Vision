from celery import Celery
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Celery configuration
celery_app = Celery(
    'kinai',
    broker=os.getenv('CELERY_BROKER_URL', 'redis://redis:6379/0'),
    backend=os.getenv('CELERY_RESULT_BACKEND', 'redis://redis:6379/0')
)

# Celery settings
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='Australia/Melbourne',
    enable_utc=False,
)

# Explicitly import tasks to register them
import tasks  # Ensure tasks.py is loaded