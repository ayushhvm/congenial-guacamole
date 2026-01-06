import os
import django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'attendance_system.settings')
django.setup()

from attendance.models import FaceRecognitionModel

active_model = FaceRecognitionModel.objects.filter(is_active=True).first()
if active_model:
    print(f"Active Model: {active_model.model_name}")
    print(f"Accuracy: {active_model.accuracy}")
    print(f"Created At: {active_model.created_at}")
else:
    print("No active model found.")
