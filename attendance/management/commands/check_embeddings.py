from django.core.management.base import BaseCommand
from attendance.models import FaceEmbedding

class Command(BaseCommand):
    help = 'Checks shapes of all face embeddings'

    def handle(self, *args, **options):
        embeddings = FaceEmbedding.objects.all()
        
        shapes = {}
        corrupted_ids = []
        
        for emb in embeddings:
            try:
                arr = emb.get_embedding()
                shape = arr.shape
                
                if shape not in shapes:
                    shapes[shape] = 0
                shapes[shape] += 1
                
                if shape != (512,):
                    corrupted_ids.append(emb.id)
                    self.stdout.write(f"Bad shape {shape} for Embedding ID {emb.id} (Student: {emb.student.student_id})")
            except Exception as e:
                self.stdout.write(f"Error loading Embedding ID {emb.id}: {e}")
                corrupted_ids.append(emb.id)
        
        self.stdout.write("--- Summary ---")
        for shape, count in shapes.items():
            self.stdout.write(f"Shape {shape}: {count} records")
            
        if corrupted_ids:
            self.stdout.write(self.style.WARNING(f"Found {len(corrupted_ids)} corrupted or inconsistent embeddings."))
            # prompt to delete? For now just list.
            # actually, let's just delete them if there are few, or valid ones are the majority.
        else:
            self.stdout.write(self.style.SUCCESS("All embeddings have consistent shapes."))
