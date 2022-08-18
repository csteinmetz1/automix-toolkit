import torch


class DifferentiableMixingConsole(torch.nn.Module):
    """Differentiable mixing console

    Notes:
        We do not use neural audio effect proxies as in the original publication.
        Instead we use a set of explicitly differentiable audio effects.

    Steinmetz et al. (2021). Automatic multitrack mixing with a differentiable mixing console of neural audio effects. ICASSP.
    """

    def __init__(self, num_params: int, d_embed: int) -> None:
        super().__init__()

        self.encoder = Encoder()
        self.post_processor = torch.nn.Sequential(
            torch.nn.Linear(d_embed, 256),
            torch.nn.PReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.PReLU(),
            torch.nn.Linear(256, num_params),
            torch.nn.Sigmoid(),
        )
        self.mixer = Mixer()  # also called the "Transformation Network"

    def forward(self, x: torch.Tensor):
        """Given a set of tracks, analyze them with a shared encoder, predict a set of mixing parameters,
        and use these parameters to generate a stereo mixture of the inputs.

        Args:
            x (torch.Tensor): Input tracks with shape (bs, num_tracks, seq_len)

        Returns:
            y (torch.Tensor): Final stereo mixture with shape (bs, 2, seq_len)
            p (torch.Tensor): Estimated mixing parameters with shape (bs, num_tracks, num_params)
        """
        bs, num_tracks, seq_len = x.size()

        # move tracks to the batch dimension to fully parallelize
        x = x.view(bs * num_tracks, -1)

        # generate single embedding for each track
        e = self.encoder(x)
        e = e.view(bs, num_tracks, -1)  # (bs, num_tracks, d_embed)

        # generate the "context" embedding
        c = e.mean(dim=1)  # (bs, 1, d_embed)
        c = c.repeat(1, num_tracks, 1)  # (bs, num_tracks, d_embed)

        # fuse the track emb and context emb
        ec = torch.cat((e, c), dim=-1)  # (bs, num_tracks, d_embed*2)

        # estimate mixing parameters for each track (in parallel)
        p = self.post_processor(ec)

        # generate the stereo mix
        y = render_mix(p, x)

        return y, p
