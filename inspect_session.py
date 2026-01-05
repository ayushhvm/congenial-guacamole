
import os
import django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'attendance_system.settings')
django.setup()

from attendance.models import AttendanceSession
from django.utils import timezone

s = AttendanceSession.objects.filter(session_name="Theory 2").last()
if s:
    print(f"Session: {s.session_name}")
    print(f"Date: {s.session_date}")
    print(f"Start: {s.start_time}")
    print(f"End: {s.end_time}")
    print(f"Current Time (UTC): {timezone.now()}")
    print(f"Current Time (Local): {timezone.localtime(timezone.now())}")
else:
    print("Session not found")
