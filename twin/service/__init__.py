"""Serving layer: demo cohort, live replay and the FastAPI application."""

# Real-world recordings shown in the dashboard demo. They are excluded from the production
# models' training data so that what the demo shows is genuinely out-of-sample.
DEMO_REAL_HOLDOUT = ("CGM-049", "CGM-036", "SH-2094-0")
