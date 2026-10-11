"""Restrained dashboard spacing; native Streamlit responsiveness is preserved."""
CSS = """
<style>
  .block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1450px; }
  h1 { font-size: 1.7rem !important; font-weight: 600 !important; letter-spacing: -.025em; }
  h2 { font-size: 1.25rem !important; }
  h3 { font-size: 1.05rem !important; }
  [data-testid="stMetric"] { border: 1px solid #e5e7eb; border-radius: 6px; padding: 14px 16px; }
  [data-testid="stMetricValue"] { font-size: 1.65rem; }
  [data-testid="stSidebar"] { border-right: 1px solid #e5e7eb; }
  [data-testid="stTabs"] { margin-top: 1rem; }
  @media (max-width: 640px) {
    .block-container { padding-left: 1rem; padding-right: 1rem; }
    [data-testid="stMetric"] { padding: 10px; }
  }
</style>
"""
