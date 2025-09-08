from rich.console import Console
from rich.table import Table
import logging
from rich.logging import RichHandler

console = Console()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
      RichHandler(console=console),
      logging.FileHandler("data/logs/test.log")
    ]
)

log = logging.getLogger(__name__)

class RichFormat:
  def __init__(self, column_names, title_name):
    self.column_names = column_names
    self.title_name = title_name

  def optimization_metrics_comparison(self):
      log.info("")
      table = Table(
          title=f"\n{self.title_name}\n",
          title_style="bold italic",
          show_lines=True
      )

      self.add_columns(table, column_names=self.column_names, justify="center")
      self.add_rows(table)

      console.print(table)
      console.print()

      with open("data/logs/test.log", "a") as f:
        file_console = Console(file=f)
        file_console.print(table)
        file_console.print()

  def add_columns(self, table, column_names, **kwargs):
      for name in column_names:
          table.add_column(name, **kwargs)

  def add_rows(self, table, *args, **kwargs):
    table.add_row(*args, **kwargs)

if __name__ == "__main__":
  column_names = ["Method", "Precision", "Speed", "Memory", "Accuracy"]
    table.add_row("FP32 (Baseline)", "32-bit", "1x", "1x", "100%", "Training, high accuracy", style="cyan")
    table.add_row("FP16", "16-bit", "1.5-2x", "0.5x", "99.9%", "General inference", style="magenta")
    table.add_row("INT8", "8-bit", "2-4x", "0.25x", "98-99%", "Mobile, edge deployment", style="green")
    table.add_row("INT4", "4-bit", "3-6x", "0.125x", "95-98%", "Extreme compression",  style="yellow")
    table.add_row("Dynamic", "Mixed", "1.5-3x", "0.3-0.7x", "99%", "Easy deployment", style="blue")
  rich_format = RichFormat(column_names, title="Compute Test")
  rich_format.optimization_metrics_comparison()

