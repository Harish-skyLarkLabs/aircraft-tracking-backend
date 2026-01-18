"""
Management command to create default admin user
"""
from django.core.management.base import BaseCommand
from accounts.models import User


class Command(BaseCommand):
    help = 'Create default admin user if it does not exist'

    def add_arguments(self, parser):
        parser.add_argument(
            '--email',
            type=str,
            default='orgadmin@skylarklabs.ai',
            help='Email for the default user'
        )
        parser.add_argument(
            '--password',
            type=str,
            default='skylark@123',
            help='Password for the default user'
        )
        parser.add_argument(
            '--role',
            type=str,
            default='admin',
            choices=['admin', 'operator', 'viewer'],
            help='Role for the default user'
        )

    def handle(self, *args, **options):
        email = options['email']
        password = options['password']
        role = options['role']

        if User.objects.filter(email=email).exists():
            self.stdout.write(
                self.style.WARNING(f'User with email {email} already exists')
            )
            return

        user = User.objects.create_user(
            email=email,
            password=password,
            role=role,
            first_name='Org',
            last_name='Admin',
            is_staff=True,
            is_superuser=True,
        )

        self.stdout.write(
            self.style.SUCCESS(f'Successfully created user: {email} with role: {role}')
        )







