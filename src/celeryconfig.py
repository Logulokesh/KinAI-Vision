from celery.schedules import crontab

beat_schedule = {
    'music-scheduler': {
        'task': 'tasks.music_scheduler',
        'schedule': 60.0,  # Run every 60 seconds
    },
}