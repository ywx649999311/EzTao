from nox_poetry import session


@session(python=["3.10", "3.11", "3.12", "3.13", "3.14"])
def tests(session):
    session.install("pytest", ".")
    session.install("joblib", ".")
    session.run("pytest")
