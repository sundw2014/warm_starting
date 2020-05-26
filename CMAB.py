class ContextualMAB(object):
    """docstring for ContextualMAB."""
    def __init__(self):
        super(ContextualMAB, self).__init__()

    def set_constants(self, nArms, nZ, nU):
        self.nArms = nArms
        self.nZ = nZ
        self.nU = nU

    def sample_context(self):
        raise NotImplementedError('sample_context')

    def play(self, arm):
        context = self.sample_context()
        return self.play(arm, context)

    def play_context(self, arm, context):
        raise NotImplementedError('play_context')
