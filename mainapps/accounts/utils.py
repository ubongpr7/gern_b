from django.core.mail import send_mail
from email.message import EmailMessage
import threading    
from django.conf import settings
from django.core.mail import EmailMultiAlternatives
from django.template.loader import render_to_string
from django.utils.html import strip_tags
class EmailThread(threading.Thread):
    def __init__(self,email_message):
        self.email_message=email_message
        threading.Thread.__init__(self)
    def run(self):
        print("Initializing thread")
        self.email_message.send()
        if self.email_message.send():
            print('done')

def send_html_email(subject, message, from_email, to_email,html_file):
    html_content = render_to_string(html_file, {'subject': subject, 'message': message})
    text_content = strip_tags(html_content)

    msg = EmailMultiAlternatives(subject, text_content, from_email, [to_email])
    msg.attach_alternative(html_content, "text/html")
    print("initials")
    EmailThread(msg).start()



def send_html_email2(
  subject, message, from_email, to_email, html_file, context
):
  html_content = render_to_string(html_file, context)

  text_content = strip_tags(html_content)

  send_mail(
    subject, text_content, from_email, [to_email], html_message=html_content
  )

def send_confirmation_email(email, name):
  # HTML email content
  logi_url = settings.DOMAIN + "login"
  if name is None:
    name = "there"
  html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Welcome to HooksMaster.io</title>
</head>
<body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
    <div style="max-width: 600px; margin: 0 auto;">
    <h2 style="color: #2c3e50;">Hi {name},</h2>
<p>Welcome to <strong>HooksMaster.io</strong>! Your journey to creating high-converting video hooks effortlessly starts here. Your account has been successfully created, and you’re ready to optimize your ads.</p>

<h3 style="color: #2c3e50;">Next Steps:</h3>
<ul>
<li><strong>Log in:</strong> <a href="https://hooksmaster.io/login" style="color: #3498db;">Login to HooksMaster.io</a></li>
<li><strong>Get Started:</strong> Prepare your hooks and generate winning creatives.</li>
</ul>

<p>If you need support, we’re here to help. Feel free to reach out to us at <a href="mailto:support@hooksmaster.io" style="color: #3498db;">support@hooksmaster.io</a>.</p>

<p>Let’s create some high-converting hooks together!</p>

<a href="https://hooksmaster.io/login" style="display: inline-block; padding: 10px 20px; background-color: #3498db; color: white; text-decoration: none; border-radius: 5px; font-weight: bold;">Login Now</a>

<p>Best regards,</p>
<p><strong>The HooksMaster.io Team</strong></p>
</div>
</body>
</html>
    """

  email_message = EmailMessage(
    subject="Welcome to HooksMaster.io – Your Account is Ready!",
    body=html_content,
    from_email=settings.EMAIL_HOST_USER,
    to=[email],
  )
  email_message.content_subtype = "html"  # This is required to send the email as HTML
  email_message.send(fail_silently=True)
