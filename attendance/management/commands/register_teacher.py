from django.core.management.base import BaseCommand
from attendance.models import Teacher
from django.core.exceptions import ValidationError

class Command(BaseCommand):
    help = 'Registers a new teacher'

    def add_arguments(self, parser):
        parser.add_argument('--id', type=str, help='Teacher ID', required=True)
        parser.add_argument('--name', type=str, help='Full Name', required=True)
        parser.add_argument('--password', type=str, help='Password', required=True)
        parser.add_argument('--email', type=str, help='Email (optional, auto-generated if not provided)', required=False)
        parser.add_argument('--phone', type=str, help='Phone (optional)', required=False)

    def handle(self, *args, **options):
        teacher_id = options['id']
        full_name = options['name']
        password = options['password']
        email = options.get('email')
        phone = options.get('phone')

        # Split name
        parts = full_name.strip().split()
        if len(parts) > 1:
            first_name = parts[0]
            last_name = ' '.join(parts[1:])
        else:
            first_name = parts[0]
            last_name = ''

        # Generate email if not provided
        if not email:
            safe_name = first_name.lower()
            email = f"{safe_name}.{teacher_id.lower()}@example.com"

        # Check if already exists
        if Teacher.objects.filter(teacher_id=teacher_id).exists():
            self.stdout.write(self.style.WARNING(f"Teacher with ID {teacher_id} already exists."))
            return

        # Create teacher
        try:
            teacher = Teacher(
                teacher_id=teacher_id,
                first_name=first_name,
                last_name=last_name,
                email=email,
                phone=phone
            )
            teacher.set_password(password)
            teacher.save()
            
            self.stdout.write(self.style.SUCCESS(f"Successfully registered teacher: {first_name} {last_name} ({teacher_id})"))
            self.stdout.write(f"Email: {email}")
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error registering teacher: {str(e)}"))
