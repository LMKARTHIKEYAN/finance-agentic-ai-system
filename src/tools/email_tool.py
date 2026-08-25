"""SMTP email delivery for validated scheduled reports."""

from __future__ import annotations

import html
import smtplib
from dataclasses import dataclass
from email.message import EmailMessage
from pathlib import Path
from typing import Iterable


class EmailDeliveryError(RuntimeError):
    """Raised when a report email cannot be delivered."""


@dataclass(frozen=True)
class EmailConfig:
    host: str
    port: int
    username: str
    password: str
    sender: str
    use_starttls: bool = True
    timeout_seconds: int = 30

    def __post_init__(self) -> None:
        if not all(isinstance(value, str) and value.strip() for value in (self.host, self.username, self.password, self.sender)):
            raise ValueError("SMTP host, username, password, and sender are required.")
        if not 1 <= self.port <= 65535:
            raise ValueError("SMTP port is invalid.")


class EmailTool:
    def __init__(self, config: EmailConfig, *, smtp_factory=smtplib.SMTP) -> None:
        self.config = config
        self._smtp_factory = smtp_factory

    def send(
        self, *, recipients: Iterable[str], subject: str, markdown_body: str,
        attachments: Iterable[str | Path] = (),
    ) -> str:
        addresses = tuple(dict.fromkeys(item.strip() for item in recipients if item.strip()))
        if not addresses:
            raise ValueError("At least one recipient is required.")
        if not subject.strip() or not markdown_body.strip():
            raise ValueError("Email subject and body are required.")
        message = EmailMessage()
        message["From"] = self.config.sender
        message["To"] = ", ".join(addresses)
        message["Subject"] = subject.strip()
        message.set_content(markdown_body)
        message.add_alternative(
            "<html><body><pre style='font-family:Arial,sans-serif;white-space:pre-wrap'>"
            + html.escape(markdown_body) + "</pre></body></html>", subtype="html",
        )
        for attachment in attachments:
            path = Path(attachment)
            if not path.is_file():
                raise ValueError(f"Email attachment does not exist: {path}")
            maintype, subtype = ("application", "pdf") if path.suffix.lower() == ".pdf" else ("application", "octet-stream")
            message.add_attachment(
                path.read_bytes(), maintype=maintype, subtype=subtype,
                filename=path.name,
            )
        try:
            with self._smtp_factory(self.config.host, self.config.port, timeout=self.config.timeout_seconds) as smtp:
                if self.config.use_starttls:
                    smtp.starttls()
                smtp.login(self.config.username, self.config.password)
                smtp.send_message(message)
        except Exception as exc:
            raise EmailDeliveryError("SMTP report delivery failed.") from exc
        return "SENT"
