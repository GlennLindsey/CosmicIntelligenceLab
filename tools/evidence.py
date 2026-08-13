from datetime import datetime, timezone


def create_evidence(source, object_name, facts):
    """
    Convert data returned by an astronomical source into
    a standardized evidence record.
    """

    return {
        "evidence_type": "astronomical_database",
        "source": source,
        "object": object_name,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "facts": facts,
    }
