
import os
import django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'attendance_system.settings')
django.setup()

from attendance.models import AttendanceSession
from datetime import date, time

s = AttendanceSession.objects.filter(session_name="Theory 2").last()
if s:
    print(f"Updating session {s.id}...")
    s.session_date = date(2026, 1, 6)
    s.end_time = time(0, 45) # Extend to 00:45
    s.save()
    print("Session updated: 2026-01-06, 00:36 - 00:45")
else:
    print("Session not found")
