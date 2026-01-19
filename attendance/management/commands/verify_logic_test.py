from django.core.management.base import BaseCommand
from django.conf import settings
from attendance.models import Student, AttendanceSession, AttendanceRecord, CaptureRecord, StudentCapture
from attendance.utils.automated_attendance import AutomatedAttendanceCapture
from django.utils import timezone
from datetime import timedelta
import shutil

class Command(BaseCommand):
    help = 'Test verification logic'

    def handle(self, *args, **options):
        self.test_verification_logic()

    def test_verification_logic(self):
        print("Testing Verification Logic...")
        
        # Setup Check: Verify we can access the class
        capture_system = AutomatedAttendanceCapture(session_id=9999)
    
        # we need to mock the models or use the real db but cleanup.
        # Let's try to unit test just the logic if possible, but the logic is tightly coupled to DB models.
        # So we will create temporary DB records.
        
        # Create a test student
        student_id = "TEST_STUDENT_001"
        student, _ = Student.objects.get_or_create(
            student_id=student_id,
            defaults={'first_name': 'Test', 'last_name': 'Student', 'email': 'test@example.com'}
        )
        
        # Create a test session
        session, _ = AttendanceSession.objects.get_or_create(
            session_name="Verification Test Session",
            defaults={
                'course_name': "TEST101", 
                'session_date': timezone.now().date(),
                'start_time': timezone.now().time(),
                'end_time': (timezone.now() + timedelta(hours=1)).time()
            }
        )
        
        # Clean up previous records for this session/student
        AttendanceRecord.objects.filter(session=session, student=student).delete()
        CaptureRecord.objects.filter(session=session).delete()
    
        # Helper to create captures
        def create_scenario(name, total_caps, present_indices, expected_status):
            print(f"\nScenario: {name}")
            
            # Cleanup
            AttendanceRecord.objects.filter(session=session, student=student).delete()
            CaptureRecord.objects.filter(session=session).delete()
            
            # Create Attendance Record (Pending)
            AttendanceRecord.objects.create(
                student=student, session=session, status='pending'
            )
            
            captures_objs = []
            for i in range(total_caps):
                cap = CaptureRecord.objects.create(
                    session=session,
                    capture_number=i+1,
                    image_path="test.jpg"
                )
                captures_objs.append(cap)
                
                if i in present_indices:
                    StudentCapture.objects.create(
                        capture=cap,
                        student=student,
                        confidence_score=0.9
                    )
            
            # Run Verification
            capture_system.verify_attendance(session)
            
            # Check Result
            record = AttendanceRecord.objects.get(student=student, session=session)
            print(f"Result: {record.status} (Expected: {expected_status})")
            print(f"Notes: {record.verification_notes}")
            
            if record.status == expected_status:
                self.stdout.write(self.style.SUCCESS("✅ PASS"))
            else:
                self.stdout.write(self.style.ERROR("❌ FAIL"))
                
        # Total 10 captures
        # Start range: 0, 1 (20% of 10 = 2)
        # End range: 8, 9 (20% of 10 = 2)
        # Majority: > 5
        # High: > 7.5 (so 8 or more)
        
        # Case A: Standard success (Start + End + Majority)
        # Present at 0, 1 (Start), 5, 6, 8, 9 (End) -> 6/10 = 60%
        create_scenario("Standard Success", 10, [0, 1, 5, 6, 8, 9], 'present')
        
        # Case B: High Participation Success (Missed Start, but 90% present)
        # Present 1-9 (Missed 0 - Start) -> 9/10 = 90%
        create_scenario("High Participation (Missed Start)", 10, [1, 2, 3, 4, 5, 6, 7, 8, 9], 'present')
        
        # Case C: High Participation Success (Missed End, but 80% present)
        # Present 0-7 -> 8/10 = 80%
        create_scenario("High Participation (Missed End)", 10, [0, 1, 2, 3, 4, 5, 6, 7], 'present')

        # Case D: Fail (Missed Start, High Participation Fail - 60%)
        # Present 2-7 -> 6/10 = 60%. Not present in Start (0,1). Majority OK, but Start/End FAIL. High Part FAIL (<75%).
        create_scenario("Fail: Good % but missed Start/End criteria", 10, [2, 3, 4, 5, 6, 7], 'absent')
        
        # Case E: Fail (Missed Majority)
        # Present 0, 9 (Start/End OK) but only 2/10
        create_scenario("Fail: Start/End OK but Low %", 10, [0, 9], 'absent')


