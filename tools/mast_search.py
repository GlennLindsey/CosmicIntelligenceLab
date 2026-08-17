from astroquery.mast import Observations

from tools.evidence import create_evidence


def mast_search(name, radius="0.02 deg"):

    observations = Observations.query_object(name, radius=radius)

    if observations is None or len(observations) == 0:

        return create_evidence(
            evidence_type="archive",
            source="MAST",
            object_name=name,
            facts={
                "status": "no_observations",
                "total_observations": 0,
                "missions": [],
                "jwst_observations": 0,
                "hst_observations": 0,
                "search_radius": radius,
            },
        )

    missions = sorted(
        set(str(value) for value in observations["obs_collection"] if value is not None)
    )

    jwst_count = sum(
        1 for value in observations["obs_collection"] if str(value).upper() == "JWST"
    )

    hst_count = sum(
        1 for value in observations["obs_collection"] if str(value).upper() == "HST"
    )

    facts = {
        "status": "observations_found",
        "total_observations": len(observations),
        "missions": missions,
        "jwst_observations": jwst_count,
        "hst_observations": hst_count,
        "search_radius": radius,
    }

    return create_evidence(
        evidence_type="archive",
        source="MAST",
        object_name=name,
        facts=facts,
    )
