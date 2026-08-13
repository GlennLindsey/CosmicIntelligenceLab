from astroquery.simbad import Simbad

from tools.evidence import create_evidence


def simbad_lookup(name):

    simbad = Simbad()

    result = simbad.query_object(name)

    if result is None:
        return create_evidence("SIMBAD", name, {"status": "not_found"})

    row = result[0]

    facts = {
        "main_id": str(row["main_id"]),
        "ra_deg": float(row["ra"]),
        "dec_deg": float(row["dec"]),
        "coo_bibcode": str(row["coo_bibcode"]),
        "matched_id": str(row["matched_id"]),
    }

    return create_evidence("SIMBAD", name, facts)
