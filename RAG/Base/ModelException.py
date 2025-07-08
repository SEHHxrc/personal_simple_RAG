class ModelNotFoundError(Exception):
    def __init__(self, err_info: str=None):
        self.err_info = err_info if err_info is not None else 'Not Found Local model.'

    def __str__(self):
        return self.err_info


class ModelError(Exception):
    def __init__(self, err_info: str=None):
        self.err_info = err_info if err_info is not None else 'Model Type not Match'

    def __str__(self):
        return self.err_info


class UnKnowRoleError(Exception):
    def __init__(self, err_info: str=None):
        self.err_info = err_info if err_info is not None else 'Illegal Role.'

    def __str__(self):
        return self.err_info
