from rich.console import Console
from rich.table import Table

import logging
log = logging.getLogger(__name__)

def print_quantization_comparison():
    console = Console()

    table = Table(title="Quantization Techniques Comparison", expand=True)

    table.add_column("Method", style="cyan", no_wrap=True)
    table.add_column("Precision", style="magenta")
    table.add_column("Speed", style="green")
    table.add_column("Memory", style="yellow")
    table.add_column("Accuracy", style="blue")
    table.add_column("Use Case", style="white")

    table.add_row("FP32 (Baseline)", "32-bit", "1x", "1x", "100%", "Training, high accuracy")
    table.add_row("FP16", "16-bit", "1.5-2x", "0.5x", "99.9%", "General inference")
    table.add_row("INT8", "8-bit", "2-4x", "0.25x", "98-99%", "Mobile, edge deployment")
    table.add_row("INT4", "4-bit", "3-6x", "0.125x", "95-98%", "Extreme compression")
    table.add_row("Dynamic", "Mixed", "1.5-3x", "0.3-0.7x", "99%", "Easy deployment")

    log.info(table)

if __name__ == "__main__":
    print_quantization_comparison()
