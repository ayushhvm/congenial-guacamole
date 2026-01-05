from django.core.management.base import BaseCommand
from attendance.models import Student, FaceEmbedding
from attendance.utils.face_recognition import FaceRecognitionSystem
from django.conf import settings
import os
import shutil
import cv2 as cv

class Command(BaseCommand):
    help = 'Registers students in bulk from provided data and yamages folder'

    def handle(self, *args, **options):
        # Student Data (ID, Name, Email, Dept, Year, Unknown, Status)
        raw_data = [
            ("S027", "Atharv A", "ss@gmail.com", "Computer Science", 3),
            ("S028", "Avani S", "sss@gmail.com", "Computer Science", 3),
            ("S029", "Avni J", "avni@gmail.com", "Computer Science", 3),
            ("S032", "Srivatsa B", "Srivats@gmail.com", "Computer Science", 3),
            ("S034", "bharath B", "Bharath@gmail.com", "Computer Science", 3),
            ("S036", "Brahati J", "Brahati@gmail.com", "Computer Science", 3),
            ("S037", "Challa Sainadh", "dd@gmail.com", "Computer Science", 3),
            ("S040", "deep S", "deep@gmail.com", "Computer Science", 3),
            ("S041", "Deepak D", "deepak@gmail.com", "Computer Science", 3),
            ("S042", "Dev A", "gg@gmail.com", "Computer Science", 3),
            ("S043", "Dhanush F", "kk@gmail.com", "Computer Science", 3),
            ("S045", "Dheeraj D", "dheeraj@gmail.com", "Computer Science", 3),
            ("S046", "Dhruti D", "dhrutid@gmail.com", "Computer Science", 3),
            ("S047", "Dhruti R", "ff@gmail.com", "Computer Science", 3),
            ("S048", "Dhruva R", "Dhruva@gmail.com", "Computer Science", 3),
            ("S049", "Ekaksh ISE", "ajdka@gmail.com", "Computer Science", 3),
            ("S050", "Ganashree G", "khakd@gmail.com", "Computer Science", 3),
            ("S051", "Gauri G", "Gauri@gmail.com", "Computer Science", 3),
            ("S052", "Hamas S", "hamas@gmail.com", "Computer Science", 3),
            ("S053", "Harsh G", "harsh@gmail.com", "Computer Science", 3),
            ("S055", "Jaganmeya H", "jaganmeya@gmail.com", "Computer Science", 3),
        ]

        yamages_root = os.path.join(settings.BASE_DIR, 'yamages')
        
        # Initialize FR System
        self.stdout.write("Initializing Face Recognition System...")
        fr_system = FaceRecognitionSystem()
        fr_system.initialize_arcface()

        for s_id, name, email, dept, year in raw_data:
            self.stdout.write(f"Processing {s_id} - {name}...")
            
            # Split name
            parts = name.strip().split()
            first_name = parts[0]
            last_name = ' '.join(parts[1:]) if len(parts) > 1 else ''
            
            # Create or Get Student
            student, created = Student.objects.get_or_create(
                student_id=s_id,
                defaults={
                    'first_name': first_name,
                    'last_name': last_name,
                    'email': email,
                    'department': dept,
                    'year': year,
                    'is_active': True
                }
            )
            
            if created:
                student.set_password("student123")
                student.save()
                self.stdout.write(f"  Created student record for {s_id}")
            else:
                self.stdout.write(f"  Student {s_id} already exists")

            # Process Images
            # Logic: Last 3 digits of ID -> Folder name
            folder_suffix = s_id[-3:] # e.g., '027'
            source_dir = os.path.join(yamages_root, folder_suffix)
            
            if not os.path.exists(source_dir):
                self.stdout.write(self.style.WARNING(f"  Image folder {source_dir} not found!"))
                continue
                
            # Create dest dir
            student_faces_dir = settings.FACE_IMAGES_DIR / s_id
            os.makedirs(student_faces_dir, exist_ok=True)
            
            processed_count = 0
            for img_name in os.listdir(source_dir):
                if not img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    continue
                    
                src_path = os.path.join(source_dir, img_name)
                
                # We need to process the image to get embedding
                # 1. Read and extract face (Single Pass)
                embedding, _ = fr_system.process_image(src_path)
                
                if embedding is None:
                    self.stdout.write(f"    Failed to detect face/embedding in {img_name}")
                    continue
                    
                # 3. Copy image to media
                # Make unique name to avoid overwrite if filenames are same
                dest_filename = f"{folder_suffix}_{img_name}"
                dest_path = student_faces_dir / dest_filename
                shutil.copy2(src_path, dest_path)
                
                # 4. Save Embedding record
                # Check if this precise image/student combo exists to avoid duplicates
                # Ideally we check embedding similarity but for import we just check if record count is excessive?
                # For now, just add it.
                
                # 4. Save Embedding record
                # Create object, set embedding, then save.
                fe = FaceEmbedding(
                    student=student,
                    image_path=str(dest_path)
                )
                fe.set_embedding(embedding)
                fe.save()
                
                processed_count += 1
                
            self.stdout.write(self.style.SUCCESS(f"  Processed {processed_count} images for {s_id}"))

        self.stdout.write(self.style.SUCCESS("Bulk registration complete!"))
