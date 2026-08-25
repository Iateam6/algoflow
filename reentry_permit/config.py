VISA_TYPE = "r-1"
DISPLAY_NAME = "Reentry Permit"
CACHE_NAMESPACE = "r_1_rag"
PIPELINE_VERSION = "r-1-rag-v1"

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
