"""Export completed figure bytes before replacing a publication asset."""
from io import BytesIO
from pathlib import Path
import os
import tempfile


def save_figure(figure, destination, **kwargs):
    destination = Path(destination)
    with BytesIO() as buffer:
        figure.savefig(buffer, format=destination.suffix[1:], **kwargs)
        data = buffer.getvalue()
    if destination.suffix == '.pdf' and not data.rstrip().endswith(b'%%EOF'):
        raise ValueError(f'Incomplete PDF export: {destination.name}')
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as output:
            temporary = Path(output.name)
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
