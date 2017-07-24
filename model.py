from torch import nn

class DanceDanceConvolution(Module):
    """
    Takes input audio 
    """
    
    def __init__(self):
        super(DanceDanceConvolution, self).__init__()
        self.step_placement = nn.Sequential(
            nn.Conv3d((7, 3, 3)),
            nn.Conv3d((3, 3, 10)),
        )