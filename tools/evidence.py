from datetime import datetime, timezone


def create_evidence(
    evidence_type,
    source,
    object_name,
    facts,
):
    """
    Create a standardized evidence record.

    Parameters
    ----------
    evidence_type : str
        Category of evidence.

        Examples:
            "astronomical_database"
            "archive"
            "spectral_analysis"

    source : str
        Source of the evidence.

    object_name : str
        Astronomical object associated with the evidence.

    facts : dict
        Explicitly measured or retrieved facts.

    Returns
    -------
    dict
        Standardized evidence record.
    """

    return {
        "evidence_type": evidence_type,
        "source": source,
        "object": object_name,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "facts": facts,
    }
