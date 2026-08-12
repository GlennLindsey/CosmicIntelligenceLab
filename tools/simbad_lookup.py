from astroquery.simbad import Simbad


def simbad_lookup(name):
    simbad = Simbad()

    result = simbad.query_object(name)

    if result is None:
        return {
            "source": "SIMBAD",
            "object": name,
            "status": "not_found",
        }

    row = result[0]

    return {
        "source": "SIMBAD",
        "object": str(row["main_id"]),
        "ra_deg": float(row["ra"]),
        "dec_deg": float(row["dec"]),
        "coo_bibcode": str(row["coo_bibcode"]),
        "matched_id": str(row["matched_id"]),
    }
