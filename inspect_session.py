
import os
import django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'attendance_system.settings')
django.setup()

from attendance.models import AttendanceSession
from django.utils import timezone

sessions = AttendanceSession.objects.all().order_by('-id')[:5]
print(f"{'ID':<5} | {'Name':<20} | {'Date':<12} | {'Active'}")
print("-" * 50)
for s in sessions:
    is_active = s.is_active
    print(f"{s.id:<5} | {s.session_name:<20} | {s.session_date} | {is_active}")
