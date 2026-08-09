from rtc.forcing import resolve_subcatchment_outlets


def test_resolve_subcatchment_outlets_accepts_pyswmm_string_connection() -> None:
    connection = {"Sub1544": "YS2001198"}

    assert resolve_subcatchment_outlets(connection, ("YS2001198",)) == {
        "Sub1544": "YS2001198"
    }
