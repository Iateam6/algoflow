VISA_TYPE = "n-400"
DISPLAY_NAME = "Naturalization (N-400)"
CACHE_NAMESPACE = "n_400_rag"
PIPELINE_VERSION = "n-400-rag-v1"

SUPPORTED_DOCUMENT_TYPES = frozenset(
    {
        "Petition Cover Letter",
        "Intent to Depart",
        "Support Letter",
        "Recommendation Letter",
        "Exhibit List",
        "Port-of-Entry Support Letter",
        "Job-Offer Analysis",
        "Credentials Memo",
        "Response Outline",
        "Demand Letter",
        "Assessment Report",
        "Visa Application Summary Report",
    }
)
