import smtplib

from email.mime.text import MIMEText
# from email.mime.multipart import MIMEMultipart

from ..core.config import settings


def send_email(full_name, receiver_email, date, time):
    """Setup SMTP server"""

    # smtp_server = "smtp.gmail.com"
    smtp_server = "live.smtp.mailtrap.io"
    smtp_port = 587
    login = "api"
    password = settings.MAILTRAP_API_KEY
    sender_mail = "hello@demomailtrap.co"
    try:
        msg = MIMEText(
            f"Hi {full_name}, your interview is scheduled on {date} at {time}."
        )
        msg["Subject"] = "Interview Booking Confirmation"
        # msg["From"] = settings.SENDER_EMAIL
        msg["From"] = sender_mail
        msg["To"] = receiver_email

        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            # server.login(settings.SENDER_EMAIL, settings.SENDER_MAIL_PASSWORD)
            server.login(login, password)
            # server.send_message(msg)
            # server.sendmail(settings.SENDER_EMAIL, receiver_email, msg.as_string())
            server.sendmail(sender_mail, receiver_email, msg.as_string())

        print("[DEBUG] Email sent successfully")

    except Exception as e:
        return f"Error while sending mail: {str(e)}"

