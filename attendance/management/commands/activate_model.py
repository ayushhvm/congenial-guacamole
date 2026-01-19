from django.core.management.base import BaseCommand
from attendance.models import FaceRecognitionModel

class Command(BaseCommand):
    help = 'Activate a specific face recognition model'

    def add_arguments(self, parser):
        parser.add_argument('model_name', type=str, help='Name of the model to activate')

    def handle(self, *args, **options):
        model_name = options['model_name']
        try:
            target_model = FaceRecognitionModel.objects.get(model_name=model_name)
            
            # Deactivate all
            FaceRecognitionModel.objects.update(is_active=False)
            
            # Activate target
            target_model.is_active = True
            target_model.save()
            
            self.stdout.write(self.style.SUCCESS(f'Successfully activated model: {model_name}'))
            self.stdout.write(f'Accuracy: {target_model.accuracy}')
            self.stdout.write(f'Created: {target_model.created_at}')
            
        except FaceRecognitionModel.DoesNotExist:
            self.stdout.write(self.style.ERROR(f'Model "{model_name}" not found in database'))
            
            # List available models to help user
            self.stdout.write('\nAvailable models:')
            for m in FaceRecognitionModel.objects.all().order_by('-created_at')[:5]:
                self.stdout.write(f'- {m.model_name} (Active: {m.is_active})')
