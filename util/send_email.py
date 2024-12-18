import smtplib
from email.mime.text import MIMEText
from email.header import Header

sender = 'example@123.com'      # email address of sender
receivers = ['example@456.com']   # email address of receivers
subject = f'{__file__} Training Completed'  # email subject
smtpserver = 'smtp.server.com'  # smtp server
port = 465  # port number
username = 'example@123.com'    # username of server        
password = '123456'   # password of server

# send email notification
def send_email(file_path):
    msg = MIMEText(f'{file_path} Training Completed', 'plain', 'utf-8') # email content
    msg['Subject'] = Header(subject, 'utf-8')   # email subject
    msg['From'] = sender    # email sender
    msg['To'] = ','.join(receivers) # email receivers

    smtp = smtplib.SMTP_SSL(smtpserver) # smtp server
    smtp.connect(smtpserver, port=port) # connect to smtp server
    smtp.login(username, password)  # login to smtp server
    smtp.sendmail(sender, receivers, msg.as_string())   # send email
    smtp.quit() # quit smtp server

    print('Email notification sent successfully.')