from astroquery.mast import Observations


def mast_search(name, radius="0.02 deg"):

    observations = Observations.query_object(
        name,
        radius=radius
    )

    if observations is None or len(observations) == 0:
        return {
            "source": "MAST",
            "object": name,
            "status": "no_observations",
            "total_observations": 0,
            "missions": []
        }

    missions = sorted(
        set(
            str(value)
            for value in observations["obs_collection"]
            if value is not None
        )
    )

    jwst_count = sum(
        1 for value in observations["obs_collection"]
        if str(value).upper() == "JWST"
    )

    hst_count = sum(
        1 for value in observations["obs_collection"]
        if str(value).upper() == "HST"
    )

    return {
        "source": "MAST",
        "object": name,
        "status": "observations_found",
        "total_observations": len(observations),
        "missions": missions,
        "jwst_observations": jwst_count,
        "hst_observations": hst_count
    }
