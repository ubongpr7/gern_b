from djoser import email

class CustomActivationEmail(email.ActivationEmail):
    template_name = "email/activation.html"

class CustomPasswordResetEmail(email.PasswordResetEmail):
    template_name = "email/password_reset.html"
