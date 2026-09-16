import contextvars
import logging
import sys

from pythonjsonlogger import jsonlogger

correlation_id_ctx = contextvars.ContextVar("correlation_id", default="-")


class CorrelationIdFilter(logging.Filter):
    def filter(self, record):
        record.correlation_id = correlation_id_ctx.get()
        return True


def setup_logging():
    logger = logging.getLogger()

    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)

    handler.addFilter(CorrelationIdFilter())

    formatter = jsonlogger.JsonFormatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s %(correlation_id)s"
    )

    handler.setFormatter(formatter)

    logger.handlers.clear()
    logger.addHandler(handler)
