"""Run with: python -m streamlit run app.py."""
import logging

from climate.ui import render

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

if __name__ == "__main__":
    render()
