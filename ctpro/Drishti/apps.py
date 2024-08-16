from django.apps import AppConfig

class DrishtiConfig(AppConfig):
    name = 'Drishti'
    label = 'Drishti'  # This should match the directory name

    def ready(self):
        print(f"Initializing {self.name} app")
        # Add any app-specific initialization code here
        print(f"{self.name} app initialized successfully")
