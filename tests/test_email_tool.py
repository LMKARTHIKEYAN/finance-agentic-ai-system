from src.tools.email_tool import EmailConfig, EmailTool


class SMTP:
    sent = None

    def __init__(self, *args, **kwargs): pass
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def starttls(self): pass
    def login(self, username, password): pass
    def send_message(self, message): SMTP.sent = message


def test_email_tool_sends_plain_html_and_pdf_attachment(tmp_path) -> None:
    attachment = tmp_path / "report.pdf"
    attachment.write_bytes(b"%PDF-test")
    tool = EmailTool(EmailConfig("smtp.example.com", 587, "user", "secret", "finance@example.com"), smtp_factory=SMTP)
    assert tool.send(recipients=["manager@example.com"], subject="Report", markdown_body="# Finance", attachments=[attachment]) == "SENT"
    assert SMTP.sent["To"] == "manager@example.com"
    assert SMTP.sent.is_multipart()
    assert any(part.get_filename() == "report.pdf" for part in SMTP.sent.iter_attachments())
