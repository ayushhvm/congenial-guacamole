from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.db.models import Q, Count
from django.utils import timezone
from datetime import date, datetime
import os
import shutil
from django.conf import settings

from .models import Student, Teacher, AttendanceSession, AttendanceRecord, FaceRecognitionModel, FaceEmbedding
from .forms import LoginForm, StudentRegistrationForm, AttendanceMarkingForm, SessionForm
from .decorators import student_required, teacher_required
from .utils.face_recognition import FaceRecognitionSystem


def login_view(request):
    """Handle login for both students and teachers"""
    if request.method == 'POST':
        form = LoginForm(request.POST)
        if form.is_valid():
            user_id = form.cleaned_data['user_id']
            password = form.cleaned_data['password']
            user_type = form.cleaned_data['user_type']
            
            if user_type == 'student':
                try:
                    student = Student.objects.get(student_id=user_id, is_active=True)
                    if student.check_password(password):
                        request.session['student_id'] = student.student_id
                        request.session['student_name'] = f"{student.first_name} {student.last_name}"
                        messages.success(request, f'Welcome back, {student.first_name}!')
                        return redirect('student_dashboard')
                    else:
                        messages.error(request, 'Invalid password.')
                except Student.DoesNotExist:
                    messages.error(request, 'Student ID not found.')
            else:  # teacher
                try:
                    teacher = Teacher.objects.get(teacher_id=user_id, is_active=True)
                    if teacher.check_password(password):
                        request.session['teacher_id'] = teacher.teacher_id
                        request.session['teacher_name'] = f"{teacher.first_name} {teacher.last_name}"
                        messages.success(request, f'Welcome back, {teacher.first_name}!')
                        return redirect('teacher_dashboard')
                    else:
                        messages.error(request, 'Invalid password.')
                except Teacher.DoesNotExist:
                    messages.error(request, 'Teacher ID not found.')
    else:
        form = LoginForm()
    
    return render(request, 'login.html', {'form': form})


def logout_view(request):
    """Handle logout"""
    request.session.flush()
    messages.success(request, 'You have been logged out successfully.')
    return redirect('login')


@student_required
def student_dashboard(request):
    """Student dashboard with overview"""
    student_id = request.session.get('student_id')
    student = get_object_or_404(Student, student_id=student_id)
    
    # Get attendance statistics
    total_records = AttendanceRecord.objects.filter(student=student).count()
    present_count = AttendanceRecord.objects.filter(student=student, status='present').count()
    absent_count = AttendanceRecord.objects.filter(student=student, status='absent').count()
    
    # Get recent attendance
    recent_attendance = AttendanceRecord.objects.filter(student=student).order_by('-marked_at')[:5]
    
    # Get unique courses
    courses = AttendanceSession.objects.filter(
        attendance_records__student=student
    ).distinct().values_list('course_name', flat=True)
    
    context = {
        'student': student,
        'total_records': total_records,
        'present_count': present_count,
        'absent_count': absent_count,
        'recent_attendance': recent_attendance,
        'courses': courses,
    }
    return render(request, 'student/dashboard.html', context)


@student_required
def student_attendance(request):
    """View student's own attendance records"""
    student_id = request.session.get('student_id')
    student = get_object_or_404(Student, student_id=student_id)
    
    # Get filter parameters
    course_filter = request.GET.get('course', '')
    session_filter = request.GET.get('session', '')
    
    # Build query
    records = AttendanceRecord.objects.filter(student=student)
    
    if course_filter:
        records = records.filter(session__course_name__icontains=course_filter)
    if session_filter:
        records = records.filter(session__session_name__icontains=session_filter)
    
    records = records.order_by('-session__session_date', '-marked_at')
    
    # Get all courses for filter dropdown
    all_courses = AttendanceSession.objects.filter(
        attendance_records__student=student
    ).values_list('course_name', flat=True).distinct()
    
    context = {
        'student': student,
        'records': records,
        'all_courses': all_courses,
        'course_filter': course_filter,
        'session_filter': session_filter,
    }
    return render(request, 'student/my_attendance.html', context)


@teacher_required
def teacher_dashboard(request):
    """Teacher dashboard with statistics"""
    # Get statistics
    total_students = Student.objects.filter(is_active=True).count()
    total_sessions = AttendanceSession.objects.filter(is_active=True).count()
    
    today = date.today()
    today_records = AttendanceRecord.objects.filter(
        session__session_date=today
    ).count()
    
    # Get active model
    try:
        active_model = FaceRecognitionModel.objects.get(is_active=True)
    except FaceRecognitionModel.DoesNotExist:
        active_model = None
    
    # Recent attendance records
    recent_records = AttendanceRecord.objects.order_by('-marked_at')[:10]
    
    context = {
        'total_students': total_students,
        'total_sessions': total_sessions,
        'today_records': today_records,
        'active_model': active_model,
        'recent_records': recent_records,
    }
    return render(request, 'teacher/dashboard.html', context)


@teacher_required
def register_student(request):
    """Register a new student with face images"""
    if request.method == 'POST':
        form = StudentRegistrationForm(request.POST, request.FILES)
        if form.is_valid():
            # Check if student already exists
            student_id = form.cleaned_data['student_id']
            if Student.objects.filter(student_id=student_id).exists():
                messages.error(request, f'Student {student_id} already exists!')
                return render(request, 'teacher/register_student.html', {'form': form})
            
            # Create student
            student = Student(
                student_id=form.cleaned_data['student_id'],
                first_name=form.cleaned_data['first_name'],
                last_name=form.cleaned_data['last_name'],
                email=form.cleaned_data['email'],
                phone=form.cleaned_data.get('phone') or '',
                department=form.cleaned_data.get('department') or '',
                year=form.cleaned_data.get('year'),
            )
            password = form.cleaned_data['password']
            student.set_password(password)
            student.save()
            
            # Process uploaded images
            images = request.FILES.getlist('images')
            if images:
                fr_system = FaceRecognitionSystem()
                fr_system.initialize_arcface()
                
                # Create directory for student faces
                student_faces_dir = settings.FACE_IMAGES_DIR / student_id
                os.makedirs(student_faces_dir, exist_ok=True)
                
                processed_count = 0
                for img_file in images:
                    # Save uploaded file temporarily
                    temp_path = os.path.join(settings.MEDIA_ROOT, 'temp', img_file.name)
                    os.makedirs(os.path.dirname(temp_path), exist_ok=True)
                    with open(temp_path, 'wb+') as destination:
                        for chunk in img_file.chunks():
                            destination.write(chunk)
                    
                    # Extract face and generate embedding (Single Pass)
                    embedding, _ = fr_system.process_image(temp_path)
                    
                    if embedding is None:
                        os.remove(temp_path)
                        continue
                    
                    # Copy image to student directory
                    dest_path = student_faces_dir / img_file.name
                    shutil.copy2(temp_path, dest_path)
                    os.remove(temp_path)
                    
                    # Save embedding to database
                    face_embedding = FaceEmbedding(
                        student=student,
                        image_path=str(dest_path)
                    )
                    face_embedding.set_embedding(embedding)
                    face_embedding.save()
                    
                    processed_count += 1
                
                messages.success(request, 
                    f'Student {student_id} registered successfully with {processed_count} face images!')
            else:
                messages.warning(request, 
                    f'Student {student_id} created but no face images were processed.')
            
            return redirect('view_students')
    else:
        form = StudentRegistrationForm()
    
    return render(request, 'teacher/register_student.html', {'form': form})


@teacher_required
def view_students(request):
    """List all students"""
    search_query = request.GET.get('search', '')
    students = Student.objects.filter(is_active=True)
    
    if search_query:
        students = students.filter(
            Q(student_id__icontains=search_query) |
            Q(first_name__icontains=search_query) |
            Q(last_name__icontains=search_query) |
            Q(email__icontains=search_query)
        )
    
    students = students.order_by('student_id')
    
    context = {
        'students': students,
        'search_query': search_query,
    }
    return render(request, 'teacher/view_students.html', context)


@teacher_required
def mark_attendance(request):
    """Mark attendance from uploaded image"""
    if request.method == 'POST':
        form = AttendanceMarkingForm(request.POST, request.FILES)
        if form.is_valid():
            session = form.cleaned_data['session']
            threshold = form.cleaned_data.get('threshold', 0.5)
            latitude = form.cleaned_data.get('latitude')
            longitude = form.cleaned_data.get('longitude')
            location_name = form.cleaned_data.get('location_name')
            
            # Save uploaded image
            image_file = request.FILES['image']
            image_path = os.path.join(settings.ATTENDANCE_IMAGES_DIR, image_file.name)
            os.makedirs(settings.ATTENDANCE_IMAGES_DIR, exist_ok=True)
            
            with open(image_path, 'wb+') as destination:
                for chunk in image_file.chunks():
                    destination.write(chunk)
            
            # Load active model
            try:
                face_model = FaceRecognitionModel.objects.get(is_active=True)
            except FaceRecognitionModel.DoesNotExist:
                messages.error(request, 'No active model found. Please train a model first.')
                return render(request, 'teacher/mark_attendance.html', {'form': form})
            
            # Initialize face recognition system
            fr_system = FaceRecognitionSystem()
            fr_system.initialize_arcface()
            fr_system.load_model(face_model.model_file)
            
            # Recognize all faces
            recognitions = fr_system.recognize_all_faces_from_image(image_path, threshold)
            
            if not recognitions:
                messages.warning(request, 'No faces detected in the image.')
                return render(request, 'teacher/mark_attendance.html', {'form': form})
            
            # Mark attendance for recognized students
            marked_count = 0
            results = []
            for name, confidence, bbox, message in recognitions:
                # Only process if we have a recognized name
                if name:
                    try:
                        student = Student.objects.get(student_id=name)
                        
                        # Check if already marked
                        existing = AttendanceRecord.objects.filter(
                            student=student,
                            session=session
                        ).first()
                        
                        if existing:
                            results.append({
                                'student': student,
                                'student_id': name,
                                'status': 'already_marked',
                                'confidence': confidence,
                            })
                        else:
                            # Mark attendance with location data
                            attendance = AttendanceRecord.objects.create(
                                student=student,
                                session=session,
                                status='present',
                                confidence_score=confidence,
                                image_path=image_path,
                                marked_by='teacher',
                                latitude=latitude,
                                longitude=longitude,
                                location_name=location_name,
                                photo_captured_at=timezone.now()
                            )
                            marked_count += 1
                            results.append({
                                'student': student,
                                'student_id': name,
                                'status': 'marked',
                                'confidence': confidence,
                                'attendance': attendance,
                            })
                    except Student.DoesNotExist:
                        # Student ID recognized but not found in database
                        results.append({
                            'student': None,
                            'student_id': name,
                            'status': 'not_found',
                            'confidence': confidence,
                        })
                else:
                    # Face detected but not recognized or confidence too low
                    results.append({
                        'student': None,
                        'student_id': None,
                        'status': 'unrecognized',
                        'confidence': confidence,
                        'message': message,
                    })
            
            messages.success(request, f'Attendance marked for {marked_count} student(s) at confidence threshold {threshold}.')
            context = {
                'form': form,
                'results': results,
                'session': session,
            }
            return render(request, 'teacher/mark_attendance.html', context)
    else:
        form = AttendanceMarkingForm()
    
    return render(request, 'teacher/mark_attendance.html', {'form': form})


@teacher_required
def view_records(request):
    """View all attendance records"""
    # Get filter parameters
    session_filter = request.GET.get('session', '')
    student_filter = request.GET.get('student', '')
    date_from = request.GET.get('date_from', '')
    date_to = request.GET.get('date_to', '')
    
    # Build query
    records = AttendanceRecord.objects.all()
    
    if session_filter:
        records = records.filter(session__id=session_filter)
    if student_filter:
        records = records.filter(student__student_id__icontains=student_filter)
    if date_from:
        records = records.filter(session__session_date__gte=date_from)
    if date_to:
        records = records.filter(session__session_date__lte=date_to)
    
    records = records.order_by('-session__session_date', '-marked_at')
    
    # Get all sessions for filter
    all_sessions = AttendanceSession.objects.filter(is_active=True).order_by('-session_date')
    
    context = {
        'records': records,
        'all_sessions': all_sessions,
        'session_filter': session_filter,
        'student_filter': student_filter,
        'date_from': date_from,
        'date_to': date_to,
    }
    return render(request, 'teacher/view_records.html', context)


@teacher_required
def view_sessions(request):
    """List and manage attendance sessions"""
    sessions = AttendanceSession.objects.all().order_by('-session_date', '-start_time')
    
    # Get statistics for each session
    for session in sessions:
        session.total_students = AttendanceRecord.objects.filter(session=session).count()
        session.present_count = AttendanceRecord.objects.filter(session=session, status='present').count()
        session.absent_count = AttendanceRecord.objects.filter(session=session, status='absent').count()
    
    context = {
        'sessions': sessions,
    }
    return render(request, 'teacher/sessions.html', context)


@teacher_required
def create_session(request):
    """Create a new attendance session with optional automated capture"""
    if request.method == 'POST':
        form = SessionForm(request.POST)
        if form.is_valid():
            session = form.save()
            
            # Check if automated capture is enabled
            auto_capture = form.cleaned_data.get('auto_capture', False)
            capture_interval = form.cleaned_data.get('capture_interval', 30)
            
            if auto_capture:
                # Prepare location data
                location_data = None
                latitude = form.cleaned_data.get('latitude')
                longitude = form.cleaned_data.get('longitude')
                location_name = form.cleaned_data.get('location_name')
                device_id = form.cleaned_data.get('device_id')
                
                # Build location data dict if any location info is provided
                if latitude or longitude or location_name or device_id:
                    location_data = {
                        'latitude': latitude,
                        'longitude': longitude,
                        'location_name': location_name,
                        'device_id': device_id
                    }
                
                # Start automated attendance capture with location data
                from .utils.automated_attendance import start_automated_attendance
                try:
                    start_automated_attendance(
                        session_id=session.id,
                        camera_index=0,
                        capture_interval=capture_interval,
                        location_data=location_data
                    )
                    location_msg = f" at {location_name}" if location_name else ""
                    messages.success(request, 
                        f'Session "{session.session_name}" created with automated attendance enabled{location_msg}! '
                        f'Captures will occur every {capture_interval} seconds.')
                except Exception as e:
                    messages.warning(request, 
                        f'Session created but automated capture failed to start: {str(e)}')
            else:
                messages.success(request, f'Session "{session.session_name}" created successfully!')
            
            return redirect('view_sessions')
    else:
        form = SessionForm()
    
    return render(request, 'teacher/create_session.html', {'form': form})


@teacher_required
def train_model(request):
    """Train face recognition model"""
    if request.method == 'POST':
        # Get classifier settings from form
        classifier_type = request.POST.get('classifier_type', 'knn')
        k_value = int(request.POST.get('k_value', 5))
        
        # Get all students with face embeddings
        students_with_faces = Student.objects.filter(
            face_embeddings__isnull=False
        ).distinct()
        
        if not students_with_faces.exists():
            messages.error(request, 'No students with face embeddings found. Please register students first.')
            return redirect('train_model')
        
        # Initialize face recognition system
        fr_system = FaceRecognitionSystem()
        fr_system.initialize_arcface()
        
        # Load embeddings from database
        embeddings_list = []
        labels_list = []
        
        for student in students_with_faces:
            face_embeddings = student.face_embeddings.all()
            for face_emb in face_embeddings:
                embedding = face_emb.get_embedding()
                embeddings_list.append(embedding)
                labels_list.append(student.student_id)
        
        import numpy as np
        embeddings = np.array(embeddings_list)
        Y = np.array(labels_list)
        
        # Create models directory if it doesn't exist
        os.makedirs(settings.FACE_MODELS_DIR, exist_ok=True)
        
        # Generate model name
        from datetime import datetime
        model_name = f'face_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        model_path = settings.FACE_MODELS_DIR / f'{model_name}.pkl'
        
        # Train model with selected classifier
        accuracy = fr_system.train_model(
            embeddings, Y, str(model_path),
            classifier_type=classifier_type,
            k=k_value
        )
        
        # Save model information to database
        encoder_path = str(model_path).replace('.pkl', '_encoder.pkl')
        
        # Deactivate all other models
        FaceRecognitionModel.objects.update(is_active=False)
        
        face_model = FaceRecognitionModel.objects.create(
            model_name=model_name,
            model_file=str(model_path),
            encoder_file=encoder_path,
            classifier_type=classifier_type,
            accuracy=accuracy,
            is_active=True
        )
        
        classifier_display = 'Centroid' if classifier_type == 'centroid' else f'k-NN (k={k_value})'
        messages.success(request, 
            f'Model trained successfully with {classifier_display}! Accuracy: {accuracy * 100:.2f}%')
        return redirect('train_model')
    
    # Get all models
    models = FaceRecognitionModel.objects.all().order_by('-created_at')
    
    context = {
        'models': models,
    }
    return render(request, 'teacher/train_model.html', context)
