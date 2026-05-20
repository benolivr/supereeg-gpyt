"""
pytest configuration for the supereeg test suite.

Sets the matplotlib backend to Agg (non-interactive) so that plot functions
can be exercised in headless CI environments without a display.
"""
import matplotlib
matplotlib.use('Agg')
