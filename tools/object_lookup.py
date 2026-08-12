def lookup_object(name):
    objects = {
        "M51": {
            "name": "Messier 51",
            "common_name": "Whirlpool Galaxy",
            "type": "Spiral galaxy",
            "constellation": "Canes Venatici",
            "distance_mly": 23.0,
        },
        "M31": {
            "name": "Messier 31",
            "common_name": "Andromeda Galaxy",
            "type": "Spiral galaxy",
            "constellation": "Andromeda",
            "distance_mly": 2.54,
        },
        "M87": {
            "name": "Messier 87",
            "common_name": "Virgo A",
            "type": "Elliptical galaxy",
            "constellation": "Virgo",
            "distance_mly": 53.5,
        },
    }

    return objects.get(
        name.upper(),
        {"error": f"No object named {name} is in the local database."}
    )
