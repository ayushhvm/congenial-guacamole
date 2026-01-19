import cv2
import os
import threading
import time
import numpy as np
from datetime import datetime, timedelta
from django.utils import timezone
from django.conf import settings
from attendance.models import AttendanceSession, AttendanceRecord, FaceRecognitionModel, Student, CaptureRecord, StudentCapture
from .face_recognition import FaceRecognitionSystem


class AutomatedAttendanceCapture:
    """
    Automated attendance capture system that runs during active sessions
    Captures photos periodically and marks attendance automatically at 0.5 confidence
    """
    
    def __init__(self, session_id, camera_index=0, capture_interval=30):
        """
        Initialize automated attendance capture
        
        Args:
            session_id: ID of the attendance session
            camera_index: Camera device index (default 0)
            capture_interval: Seconds between captures (default 30)
        """
        self.session_id = session_id
        self.camera_index = camera_index
        self.capture_interval = capture_interval
        self.is_running = False
        self.thread = None
        self.fr_system = None
        self.threshold = 0.5  # Auto-mark at 0.5 confidence
        self.camera = None
        self.capture_count = 0  # Track number of captures
        
    def initialize_camera(self):
        """Initialize camera with optimal settings"""
        try:
            self.camera = cv2.VideoCapture(self.camera_index)
            
            if not self.camera.isOpened():
                print(f"Error: Could not open camera {self.camera_index}")
                return False
            
            # Set camera properties for better quality
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            self.camera.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # Enable autofocus
            self.camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # Enable auto exposure
            
            # Warmup camera (capture and discard a few frames)
            print("Warming up camera...")
            for i in range(10):
                ret, frame = self.camera.read()
                time.sleep(0.1)
            
            print("Camera initialized successfully")
            return True
            
        except Exception as e:
            print(f"Error initializing camera: {str(e)}")
            return False
    
    def release_camera(self):
        """Release camera resources"""
        if self.camera:
            self.camera.release()
            self.camera = None
        
    def initialize_face_recognition(self):
        """Initialize face recognition system with active model"""
        try:
            face_model = FaceRecognitionModel.objects.get(is_active=True)
            self.fr_system = FaceRecognitionSystem()
            self.fr_system.initialize_arcface()
            self.fr_system.load_model(face_model.model_file)
            return True
        except FaceRecognitionModel.DoesNotExist:
            print("No active face recognition model found")
            return False
        except Exception as e:
            print(f"Error initializing face recognition: {str(e)}")
            return False
    
    def adjust_brightness(self, image):
        """
        Adjust image brightness if too dark or too bright
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Adjusted image
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Calculate mean brightness
        mean_brightness = np.mean(l)
        
        # If too dark (mean < 80), apply CLAHE
        if mean_brightness < 80:
            print(f"Image too dark (brightness: {mean_brightness:.1f}), applying enhancement...")
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            enhanced_lab = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
            return enhanced, True
        
        # If too bright (mean > 200), reduce
        elif mean_brightness > 200:
            print(f"Image too bright (brightness: {mean_brightness:.1f}), reducing...")
            l = cv2.convertScaleAbs(l, alpha=0.7, beta=0)
            adjusted_lab = cv2.merge([l, a, b])
            adjusted = cv2.cvtColor(adjusted_lab, cv2.COLOR_LAB2BGR)
            return adjusted, True
        
        return image, False
    
    def check_image_quality(self, image):
        """
        Check if image quality is sufficient for face detection
        
        Args:
            image: Input image
            
        Returns:
            (is_good, reason) tuple
        """
        if image is None or image.size == 0:
            return False, "Empty image"
        
        # Check if image is too dark
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        
        if mean_brightness < 20:
            return False, f"Too dark (brightness: {mean_brightness:.1f})"
        
        # Check if image is too blurry (using Laplacian variance)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        if laplacian_var < 50:
            return False, f"Too blurry (variance: {laplacian_var:.1f})"
        
        return True, "OK"
    
    def capture_frame_with_retry(self, max_retries=3):
        """
        Capture frame with retry logic and quality checks
        
        Args:
            max_retries: Maximum number of retry attempts
            
        Returns:
            frame or None
        """
        for attempt in range(max_retries):
            if not self.camera or not self.camera.isOpened():
                print("Camera not initialized, reinitializing...")
                if not self.initialize_camera():
                    return None
            
            # Capture frame
            ret, frame = self.camera.read()
            
            if not ret or frame is None:
                print(f"Failed to capture frame (attempt {attempt + 1}/{max_retries})")
                time.sleep(0.5)
                continue
            
            # Check quality
            is_good, reason = self.check_image_quality(frame)
            
            if not is_good:
                print(f"Poor quality frame: {reason} (attempt {attempt + 1}/{max_retries})")
                time.sleep(0.5)
                continue
            
            # Adjust brightness if needed
            adjusted_frame, was_adjusted = self.adjust_brightness(frame)
            
            if was_adjusted:
                print("Image brightness adjusted")
            
            return adjusted_frame
        
        print("Failed to capture good quality frame after all retries")
        return None
    
    def capture_and_process(self, session, location_data=None):
        """
        Capture image from camera and process for attendance
        
        Args:
            session: AttendanceSession object
            location_data: Dict with latitude, longitude, location_name (optional)
        """
        # Capture frame with quality checks
        frame = self.capture_frame_with_retry(max_retries=3)
        
        if frame is None:
            print("Error: Could not capture valid frame")
            return
        
        # Increment capture count
        self.capture_count += 1
        
        # Save image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_filename = f"auto_capture_{session.id}_{timestamp}.jpg"
        image_path = os.path.join(settings.ATTENDANCE_IMAGES_DIR, image_filename)
        
        os.makedirs(settings.ATTENDANCE_IMAGES_DIR, exist_ok=True)
        
        # Save with high quality
        cv2.imwrite(image_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        print(f"Image saved: {image_filename} (Capture #{self.capture_count})")
        
        # Create CaptureRecord
        capture_record = CaptureRecord.objects.create(
            session=session,
            capture_number=self.capture_count,
            image_path=image_path,
            latitude=location_data.get('latitude') if location_data else None,
            longitude=location_data.get('longitude') if location_data else None,
            location_name=location_data.get('location_name') if location_data else None,
        )
        
        # Process image for face recognition
        self.process_attendance_from_image(image_path, session, location_data, capture_record)
    
    def process_attendance_from_image(self, image_path, session, location_data, capture_record):
        """
        Process captured image and track student appearances
        
        Args:
            image_path: Path to captured image
            session: AttendanceSession object
            location_data: Dict with latitude, longitude, location_name (optional)
            capture_record: CaptureRecord object
        """
        if not self.fr_system:
            print("Face recognition system not initialized")
            return
        
        try:
            # Recognize all faces in image
            recognitions = self.fr_system.recognize_all_faces_from_image(
                image_path, 
                self.threshold
            )
            
            if not recognitions:
                print(f"No faces detected in {os.path.basename(image_path)}")
                capture_record.is_processed = True
                capture_record.faces_detected = 0
                capture_record.save()
                return
            
            # Track detections for this capture
            detected_students = []
            capture_time = timezone.now()
            
            for name, confidence, bbox, message in recognitions:
                if name and confidence >= self.threshold:
                    try:
                        student = Student.objects.get(student_id=name)
                        
                        # Record this student's appearance in this capture
                        StudentCapture.objects.create(
                            capture=capture_record,
                            student=student,
                            confidence_score=confidence,
                            bbox_x=bbox[0] if bbox else None,
                            bbox_y=bbox[1] if bbox else None,
                            bbox_w=bbox[2] if bbox else None,
                            bbox_h=bbox[3] if bbox else None,
                        )
                        
                        detected_students.append(student)
                        print(f"✓ Detected {student.student_id} - {student.first_name} {student.last_name} (confidence: {confidence:.2f})")
                        
                        # Get or create attendance record (with pending status)
                        attendance, created = AttendanceRecord.objects.get_or_create(
                            student=student,
                            session=session,
                            defaults={
                                'status': 'pending',
                                'confidence_score': confidence,
                                'image_path': image_path,
                                'marked_by': 'auto_system',
                                'photo_captured_at': capture_time,
                                'latitude': location_data.get('latitude') if location_data else None,
                                'longitude': location_data.get('longitude') if location_data else None,
                                'location_name': location_data.get('location_name') if location_data else None,
                                'device_id': location_data.get('device_id') if location_data else None,
                            }
                        )
                        
                        if created:
                            print(f"  → Created pending attendance record for {student.student_id}")
                        
                    except Student.DoesNotExist:
                        print(f"✗ Student {name} not found in database")
                    except Exception as e:
                        print(f"✗ Error processing {name}: {str(e)}")
            
            # Update capture record
            capture_record.is_processed = True
            capture_record.faces_detected = len(detected_students)
            capture_record.save()
            
            print(f"Capture #{self.capture_count} complete: {len(detected_students)} student(s) detected")
            
        except Exception as e:
            print(f"Error processing image: {str(e)}")
    
    def run_session_capture(self, location_data=None):
        """
        Main loop for automated capture during session
        Runs from session start to end time
        
        Args:
            location_data: Dict with latitude, longitude, location_name (optional)
        """
        try:
            session = AttendanceSession.objects.get(id=self.session_id, is_active=True)
        except AttendanceSession.DoesNotExist:
            print(f"Session {self.session_id} not found or inactive")
            return
        
        # Initialize camera
        if not self.initialize_camera():
            print("Failed to initialize camera")
            return
        
        # Initialize face recognition
        if not self.initialize_face_recognition():
            print("Failed to initialize face recognition system")
            self.release_camera()
            return
        
        # Calculate session start and end times
        session_datetime_start = datetime.combine(session.session_date, session.start_time)
        session_datetime_end = datetime.combine(session.session_date, session.end_time)
        
        # Make timezone aware
        if settings.USE_TZ:
            import pytz
            tz = pytz.timezone(settings.TIME_ZONE)
            session_datetime_start = tz.localize(session_datetime_start)
            session_datetime_end = tz.localize(session_datetime_end)
        
        print("\n" + "="*60)
        print(f"🎥 AUTOMATED ATTENDANCE CAPTURE STARTED")
        print("="*60)
        print(f"Session: {session.session_name}")
        print(f"Course: {session.course_name}")
        print(f"Time: {session_datetime_start.strftime('%H:%M')} - {session_datetime_end.strftime('%H:%M')}")
        print(f"Capture Interval: {self.capture_interval} seconds")
        print(f"Confidence Threshold: {self.threshold}")
        print(f"Camera Index: {self.camera_index}")
        print("="*60 + "\n")
        
        self.is_running = True
        self.capture_count = 0
        
        try:
            while self.is_running:
                current_time = timezone.now()
                
                # Check if session has ended
                if current_time > session_datetime_end:
                    print("\n⏰ Session ended. Running final verification...")
                    self.verify_attendance(session)
                    break
                
                # Check if session has started
                if current_time >= session_datetime_start:
                    print(f"\n[{current_time.strftime('%H:%M:%S')}] 📸 Capturing and processing...")
                    self.capture_and_process(session, location_data)
                else:
                    wait_seconds = (session_datetime_start - current_time).total_seconds()
                    print(f"⏳ Waiting {int(wait_seconds)} seconds for session to start...")
                
                # Wait for next capture
                time.sleep(self.capture_interval)
        
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user. Running verification...")
            self.verify_attendance(session)
        
        finally:
            self.is_running = False
            self.release_camera()
            print("\n" + "="*60)
            print("🛑 Automated capture stopped")
            print("="*60 + "\n")
    
    def verify_attendance(self, session):
        """
        Verify attendance based on presence across multiple captures
        Students must be present:
        - In the beginning (first 20% of captures)
        - In the end (last 20% of captures)
        - In majority of captures (>50%)
        
        Args:
            session: AttendanceSession object
        """
        print("\n" + "="*60)
        print("🔍 VERIFYING ATTENDANCE")
        print("="*60)
        
        # Get all captures for this session
        captures = CaptureRecord.objects.filter(session=session).order_by('capture_number')
        total_captures = captures.count()
        
        if total_captures == 0:
            print("No captures found. Cannot verify attendance.")
            return
        
        print(f"Total captures: {total_captures}")
        
        # Define start and end ranges (20% of captures)
        start_range = max(1, int(total_captures * 0.2))
        end_range = max(1, int(total_captures * 0.2))
        majority_threshold = total_captures * 0.5
        
        print(f"Start range: First {start_range} captures")
        print(f"End range: Last {end_range} captures")
        print(f"Majority threshold: >{majority_threshold:.1f} captures\n")
        
        # Get all pending attendance records for this session
        pending_records = AttendanceRecord.objects.filter(session=session, status='pending')
        
        verified_count = 0
        rejected_count = 0
        
        for record in pending_records:
            student = record.student
            
            # Count total appearances
            total_appearances = StudentCapture.objects.filter(
                capture__session=session,
                student=student
            ).count()
            
            # Check if present in start captures
            start_captures = captures[:start_range]
            present_in_start = StudentCapture.objects.filter(
                capture__in=start_captures,
                student=student
            ).exists()
            
            # Check if present in end captures
            end_captures = captures[total_captures - end_range:]
            present_in_end = StudentCapture.objects.filter(
                capture__in=end_captures,
                student=student
            ).exists()
            
            # Calculate presence percentage
            presence_percentage = (total_appearances / total_captures) * 100
            
            # Verification logic
            # Modified: 
            # 1. Standard: Start + End + Majority (>50%)
            # 2. High Participation: >75% of captures (overrides Start/End requirement)
            
            high_participation_threshold = total_captures * 0.75
            
            is_verified = False
            verification_method = ""
            
            # Check Standard Criteria
            if present_in_start and present_in_end and total_appearances > majority_threshold:
                is_verified = True
                verification_method = "Standard (Start + End + >50%)"
                
            # Check High Participation Criteria (Relaxed)
            elif total_appearances > high_participation_threshold:
                is_verified = True
                verification_method = "High Participation (>75%)"
            
            # Update attendance record
            record.total_captures = total_appearances
            record.present_in_start = present_in_start
            record.present_in_end = present_in_end
            record.is_verified = is_verified
            
            if is_verified:
                record.status = 'present'
                record.verification_notes = f"Verified ({verification_method}): {total_appearances}/{total_captures} captures ({presence_percentage:.1f}%)"
                verified_count += 1
                print(f"✓ {student.student_id} - {student.first_name} {student.last_name}: PRESENT")
                print(f"  └─ Method: {verification_method}")
                print(f"  └─ {total_appearances}/{total_captures} captures ({presence_percentage:.1f}%)")
                print(f"  └─ Start: {'✓' if present_in_start else '✗'}, End: {'✓' if present_in_end else '✗'}")
            else:
                record.status = 'absent'
                reasons = []
                if not present_in_start:
                    reasons.append("not present at start")
                if not present_in_end:
                    reasons.append("not present at end")
                if total_appearances <= majority_threshold:
                    reasons.append(f"only {total_appearances}/{total_captures} captures (needs >{int(majority_threshold)})")
                
                record.verification_notes = f"Rejected: {', '.join(reasons)}"
                rejected_count += 1
                print(f"✗ {student.student_id} - {student.first_name} {student.last_name}: ABSENT")
                print(f"  └─ {total_appearances}/{total_captures} captures ({presence_percentage:.1f}%)")
                print(f"  └─ Start: {'✓' if present_in_start else '✗'}, End: {'✓' if present_in_end else '✗'}")
                print(f"  └─ Reason: {', '.join(reasons)}")
            
            record.save()
        
        print("\n" + "-"*60)
        print(f"✓ Verified Present: {verified_count}")
        print(f"✗ Marked Absent: {rejected_count}")
        print("="*60 + "\n")
    
    def start(self, location_data=None):
        """
        Start automated capture in a background thread
        
        Args:
            location_data: Dict with latitude, longitude, location_name (optional)
        """
        if self.is_running:
            print("Automated capture already running")
            return False
        
        self.thread = threading.Thread(
            target=self.run_session_capture,
            args=(location_data,),
            daemon=True
        )
        self.thread.start()
        return True
    
    def stop(self):
        """Stop automated capture"""
        self.is_running = False
        self.release_camera()
        if self.thread:
            self.thread.join(timeout=5)
        print("Automated capture stopped")


def start_automated_attendance(session_id, camera_index=0, capture_interval=30, location_data=None):
    """
    Helper function to start automated attendance capture for a session
    
    Args:
        session_id: ID of the attendance session
        camera_index: Camera device index (default 0)
        capture_interval: Seconds between captures (default 30)
        location_data: Dict with latitude, longitude, location_name (optional)
    
    Returns:
        AutomatedAttendanceCapture instance
    """
    capture = AutomatedAttendanceCapture(
        session_id=session_id,
        camera_index=camera_index,
        capture_interval=capture_interval
    )
    capture.start(location_data)
    return capture
