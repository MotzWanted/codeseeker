import typing as typ

from rich import progress, text


class ProcessingSpeedColumn(progress.ProgressColumn):
    """Renders human readable processing speed."""

    def render(self, task: progress.Task) -> text.Text:
        """Show data transfer speed."""
        speed = task.fields.get("speed", None)
        if speed is not None and speed < 0:
            return text.Text("", style="progress.data.speed")
        if speed is None:
            speed = task.finished_speed or task.speed
            if speed is None:
                return text.Text("?", style="progress.data.speed")
        speed_str = f"{1 / speed:.2f} s/batch" if speed < 1 else f"{speed:.2f} batch/s"
        return text.Text(speed_str, style="progress.data.speed")


class IterProgressBar(progress.Progress):
    """Progress bar for batch processing."""

    def __init__(self, **kwarg: typ.Any):
        columns = [
            progress.TextColumn("[bold blue]{task.description}", justify="right"),
            progress.BarColumn(bar_width=None),
            ProcessingSpeedColumn(),
            "•",
            progress.TimeRemainingColumn(),
            "•",
            progress.TextColumn("{task.fields[info]}", justify="left"),
        ]
        super().__init__(*columns, **kwarg)
