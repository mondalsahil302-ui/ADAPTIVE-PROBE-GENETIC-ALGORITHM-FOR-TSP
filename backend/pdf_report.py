"""
Backward compatibility bridge forwarding to backend.reports and backend.storage.
"""
from backend.reports.pdf_generator import generate_pdf_report, render_report_charts
from backend.storage.file_store import persist_run_artifacts, create_run_zip, list_saved_runs

