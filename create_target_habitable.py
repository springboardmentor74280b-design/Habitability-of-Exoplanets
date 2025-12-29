df['habitable'] = df['pl_eqt'].apply
    (
    lambda x: 1 if 180 <= x <= 310 else 0
    )
