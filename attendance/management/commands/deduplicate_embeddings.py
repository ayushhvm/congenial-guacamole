from django.core.management.base import BaseCommand
from attendance.models import FaceEmbedding, Student
from django.db.models import Count

class Command(BaseCommand):
    help = 'Removes duplicate face embeddings for students'

    def handle(self, *args, **options):
        students = Student.objects.annotate(num_embeddings=Count('face_embeddings')).filter(num_embeddings__gt=0)
        
        total_deleted = 0
        
        for student in students:
            embeddings = FaceEmbedding.objects.filter(student=student).order_by('id')
            
            seen_paths = set()
            duplicates = []
            
            for emb in embeddings:
                # We identify duplicates by image_path
                # If image_path is same, it's the same file registered twice
                if emb.image_path in seen_paths:
                    duplicates.append(emb.id)
                else:
                    seen_paths.add(emb.image_path)
            
            if duplicates:
                count = len(duplicates)
                FaceEmbedding.objects.filter(id__in=duplicates).delete()
                self.stdout.write(f"Removed {count} duplicate(s) for {student.student_id}")
                total_deleted += count
                
        self.stdout.write(self.style.SUCCESS(f"Total duplicates removed: {total_deleted}"))
