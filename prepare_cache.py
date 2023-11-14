import argparse
import sys

from torch.utils.data import DataLoader


class LunaPrepCacheApp:
    @classmethod
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser()
        parser.add_argument('--batch_size',
                            help='Batch size to use for training',
                            default=8,
                            type=int,
                            )
        parser.add_argument('--num_workers',
                            help='Number of worker processes for background data loading',
                            default=2,
                            type=int,
                            )
        parser.add_argument('--data_dir',
                            help='Directory of data',
                            default="data/",
                            type=str,
                            )

        self.args = parser.parse_args(sys_argv)

    def main(self):
        log.info("Starting {}, {}".format(type(self).__name__, self.args))

        prep_dl = DataLoader(
            LunaDataset(sortby_str='series_uid', data_dir=self.args.data_dir),
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
        )

        batch_iter = enumerateWithEstimate(
            prep_dl,
            "Stuffing cache",
            start_ndx=prep_dl.num_workers,
        )
        for _ in batch_iter:
            pass


if __name__ == '__main__':
    LunaPrepCacheApp().main()
