from django.core.management.base import BaseCommand
from attendance.models import FaceEmbedding

class Command(BaseCommand):
    help = 'Deletes corrupted face embeddings (shape (0,))'

    def handle(self, *args, **options):
        embeddings = FaceEmbedding.objects.all()
        
        deleted_count = 0
        
        for emb in embeddings:
            try:
                arr = emb.get_embedding()
                if arr.shape != (512,):
                    emb.delete()
                    deleted_count += 1
            except Exception:
                # If we cannot load it, it's bad
                emb.delete()
                deleted_count += 1
        
        self.stdout.write(self.style.SUCCESS(f"Successfully deleted {deleted_count} corrupted embeddings."))
