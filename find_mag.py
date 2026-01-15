from fetch_iso import IsochroneFetcher



def apparant_to_abs(m_app, distance_pc=None, parallax=None, A=0.0):
    m = np.asarray(m_app)
    if distance_pc is None:
        if parallax is None:
            raise ValueError("Provide distance or parallax")